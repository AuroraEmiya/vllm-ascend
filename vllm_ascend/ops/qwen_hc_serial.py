# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""A3 BF16 serial HC: public matmuls and owned Ascend C vector stages.

Load vllm_ascend_C or the HC-only vllm_ascend_hc_C extension before calling.
Forward does not compile code or retain a hidden cache.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def combine_and_mix(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    previous_injection: torch.Tensor | None,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    *,
    eps: float,
    hc_count: int,
    use_combine: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return (updated residual, next block input, next injection view).

    down_weight is [R+K+P,D] for intermediate connections and [R,D] for
    the final mixer; up_weight is [D,R]. P=(-(R+K)) % 16.
    Inputs are read-only. Injection retains the down output's storage and row
    stride, including padding that is otherwise discarded.
    """
    rank = up_weight.shape[1]
    dim = residual.shape[1]
    padding = (-(rank + hc_count)) % 16 if use_combine else 0
    expected_down = rank + hc_count + padding if use_combine else rank
    if down_weight.shape != (expected_down, dim) or up_weight.shape[0] != dim:
        raise ValueError("HC projection weights do not match the full path")
    if down_weight.dtype != residual.dtype or up_weight.dtype != residual.dtype:
        raise ValueError("HC projection dtype mismatch")
    if down_weight.device != residual.device or up_weight.device != residual.device:
        raise ValueError("HC projection device mismatch")
    combined, normalized = torch.ops._C_ascend.qwen_hc_combine_norm(
        residual, block_output, previous_injection, norm_weight, eps, hc_count
    )
    down = F.linear(normalized, down_weight)
    lowrank = torch.ops._C_ascend.qwen_hc_silu(down[:, :rank], hc_count)
    next_injection = down[:, rank : rank + hc_count] if use_combine else None
    gate = F.linear(lowrank, up_weight)
    block_input = torch.ops._C_ascend.qwen_hc_gate_mix(normalized, gate, hc_count)
    return combined, block_input, next_injection


def combine_and_mix_module(
    module,
    hidden_states: torch.Tensor,
    prev_block_output: torch.Tensor,
    prev_injection: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Explicit adapter for an unquantized replicated GatedResidual module."""
    down = module.input_mix_weight_down_block_inject if module.use_combine else module.input_mix_weight_down
    return combine_and_mix(
        hidden_states,
        prev_block_output,
        prev_injection,
        module.hc_norm.weight,
        down.weight,
        module.input_mix_weight_up.weight,
        eps=module.config.rms_norm_eps,
        hc_count=module.hc_count,
        use_combine=module.use_combine,
    )
