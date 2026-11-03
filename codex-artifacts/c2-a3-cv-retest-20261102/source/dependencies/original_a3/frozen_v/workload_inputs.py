"""Deterministic, fresh V inputs; import make_inputs from the benchmark reference.

The framework calls this full callback separately for each custom input. A case
therefore uses a local CPU RNG with a stable seed and never caches mutable state.
Requires existing PyTorch at execution time; this module installs nothing.
"""
import json
from pathlib import Path


def case_for_axes(axes):
    cases = json.loads(Path(__file__).with_name("workload_manifest.json").read_text(encoding="utf-8"))
    case = cases[int(axes["case_index"])]
    if any(int(axes[name]) != value for name, value in case["axes"].items()):
        raise ValueError("Axes disagree with the selected V case")
    return case


def make_inputs(axes, device):
    import torch

    case = case_for_axes(axes)
    t, d, p = (case["axes"][name] for name in ("T", "D", "P"))
    rng = torch.Generator(device="cpu").manual_seed(case["seed"])
    kv = torch.randn((t, d), generator=rng, dtype=torch.float32)
    scores = torch.randn((t, d), generator=rng, dtype=torch.float32)
    state = torch.zeros((p, 32, 2 * d), dtype=torch.float32)
    for start, used, _, _, page in zip(*case["metadata"]):
        if not used:
            continue
        for position in range(max(0, start - 32), start):
            state[page, position % 32] = torch.randn((2 * d,), generator=rng, dtype=torch.float32)
    norm_weight = (0.5 + torch.rand((d,), generator=rng)).to(dtype=torch.bfloat16)
    # Historical cases retain the identical RNG sequence and random values.
    # Apply each synthetic numerical policy to both current and cached values.
    mode = case.get("value_mode", "random")
    if mode == "zero_kv":
        kv.zero_()
        state[:, :, :d].zero_()
    elif mode == "equal_scores":
        scores.zero_()
        state[:, :, d:].zero_()
    elif mode == "extreme_scores":
        scores[0::2].fill_(10000.0)
        scores[1::2].fill_(-10000.0)
        for start, used, _, base, page in zip(*case["metadata"]):
            if used:
                for position in range(max(0, start - 32), start):
                    state[page, position % 32, d:].fill_(
                        10000.0 if (base + position - start) % 2 == 0 else -10000.0
                    )
    elif mode == "zero_norm":
        norm_weight.zero_()
    elif mode == "small_kv":
        kv.mul_(1e-12)
        state[:, :, :d].mul_(1e-12)
    elif mode != "random":
        raise ValueError(f"Unknown V numerical policy: {mode}")
    return {
        "kv": kv.to(device), "scores": scores.to(device), "state_cache": state.to(device),
        "metadata": torch.tensor(case["metadata"], dtype=torch.int32, device=device),
        "norm_weight": norm_weight.to(device=device),
        "epsilon": case["epsilon"], "max_query_len": case["max_query_len"],
    }


def fresh_inputs(axes, device):
    """Call before each independent reference/candidate replay if supported."""
    return make_inputs(axes, device)


def reset_state(inputs, pristine_state):
    """Runner hook: restore Ring before EACH replay, including timing repeats.

    The caller must snapshot inputs['state_cache'].clone() before the first run.
    Providing this helper does not install it into the benchmark runner.
    """
    inputs["state_cache"].copy_(pristine_state)
