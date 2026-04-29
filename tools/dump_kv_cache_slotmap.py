#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Run vLLM once and dump the KV-cache slot mapping used by Ascend.

In vLLM Ascend, ``slot_mapping[token_idx]`` is the flattened KV-cache slot
where that token's K/V vectors are stored. For a paged cache with block size
``B``:

    block_idx = slot_mapping[token_idx] // B
    block_offset = slot_mapping[token_idx] % B

This is the same convention used by ``tests/ut/ops/test_store_kv_block.py``.
"""

import argparse
import contextlib
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch


Capture = dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Model id for online loading, or a local model directory when --source=local.",
    )
    parser.add_argument(
        "--source",
        choices=("local", "hf", "modelscope"),
        default="local",
        help="Where vLLM should load the model from. hf uses Hugging Face; modelscope sets VLLM_USE_MODELSCOPE=True.",
    )
    parser.add_argument(
        "--output-dir",
        default="kv_cache_dump",
        help="Directory where dump.pt and metadata.json are written.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="Prompt to run. Can be repeated. Defaults to one short prompt.",
    )
    parser.add_argument("--max-tokens", type=int, default=1, help="Number of generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Optional vLLM max_model_len override.")
    parser.add_argument("--dtype", default="auto", help="vLLM dtype, for example auto, float16, bfloat16.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to vLLM.")
    parser.add_argument(
        "--max-captures",
        type=int,
        default=1,
        help="Maximum cache writes to save. Use 0 to save every write; this can be very large.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Move captured tensors to CPU before saving. Enabled by default behavior; this flag is kept for clarity.",
    )
    return parser.parse_args()


def _configure_source(source: str) -> None:
    if source == "modelscope":
        os.environ["VLLM_USE_MODELSCOPE"] = "True"
    elif source == "hf":
        os.environ.pop("VLLM_USE_MODELSCOPE", None)


def _to_cpu(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    try:
        if tensor.device.type == "npu":
            torch.npu.synchronize()
    except Exception:
        pass
    return tensor.detach().cpu().clone()


def _tensor_summary(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _maybe_append_capture(
    captures: list[Capture],
    max_captures: int,
    *,
    source: str,
    key: torch.Tensor | None,
    value: torch.Tensor | None,
    key_cache: torch.Tensor | None,
    value_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor | None,
) -> None:
    if max_captures > 0 and len(captures) >= max_captures:
        return

    captures.append(
        {
            "source": source,
            "slot_mapping": _to_cpu(slot_mapping),
            "key": _to_cpu(key),
            "value": _to_cpu(value),
            "key_cache": _to_cpu(key_cache),
            "value_cache": _to_cpu(value_cache),
            "summaries": {
                "slot_mapping": _tensor_summary(slot_mapping),
                "key": _tensor_summary(key),
                "value": _tensor_summary(value),
                "key_cache": _tensor_summary(key_cache),
                "value_cache": _tensor_summary(value_cache),
            },
        }
    )


def _arg(args: tuple[Any, ...], index: int, kwargs: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in kwargs:
            return kwargs[name]
    return args[index] if index < len(args) else None


@contextlib.contextmanager
def _capture_kv_writes(captures: list[Capture], max_captures: int) -> Iterator[None]:
    import torch_npu

    from vllm_ascend.device import device_op

    original_device_reshape_attr = device_op.DeviceOperator.__dict__["reshape_and_cache"]
    original_device_reshape: Callable[..., Any] = device_op.DeviceOperator.reshape_and_cache
    original_npu_reshape: Callable[..., Any] = torch_npu._npu_reshape_and_cache
    original_scatter = getattr(torch_npu, "npu_scatter_pa_kv_cache", None)

    def wrapped_device_reshape(cls, *args, **kwargs):
        result = original_device_reshape(*args, **kwargs)
        _maybe_append_capture(
            captures,
            max_captures,
            source="DeviceOperator.reshape_and_cache",
            key=kwargs.get("key"),
            value=kwargs.get("value"),
            key_cache=kwargs.get("key_cache"),
            value_cache=kwargs.get("value_cache"),
            slot_mapping=kwargs.get("slot_mapping"),
        )
        return result

    def wrapped_npu_reshape(*args, **kwargs):
        result = original_npu_reshape(*args, **kwargs)
        _maybe_append_capture(
            captures,
            max_captures,
            source="torch_npu._npu_reshape_and_cache",
            key=_arg(args, 0, kwargs, "key"),
            value=_arg(args, 1, kwargs, "value"),
            key_cache=_arg(args, 2, kwargs, "key_cache"),
            value_cache=_arg(args, 3, kwargs, "value_cache"),
            slot_mapping=_arg(args, 4, kwargs, "slot_indices", "slot_mapping"),
        )
        return result

    def wrapped_scatter(*args, **kwargs):
        result = original_scatter(*args, **kwargs)
        _maybe_append_capture(
            captures,
            max_captures,
            source="torch_npu.npu_scatter_pa_kv_cache",
            key=_arg(args, 0, kwargs, "key"),
            value=_arg(args, 1, kwargs, "value"),
            key_cache=_arg(args, 2, kwargs, "key_cache"),
            value_cache=_arg(args, 3, kwargs, "value_cache"),
            slot_mapping=_arg(args, 4, kwargs, "slot_mapping"),
        )
        return result

    device_op.DeviceOperator.reshape_and_cache = classmethod(wrapped_device_reshape)
    torch_npu._npu_reshape_and_cache = wrapped_npu_reshape
    if original_scatter is not None:
        torch_npu.npu_scatter_pa_kv_cache = wrapped_scatter

    try:
        yield
    finally:
        device_op.DeviceOperator.reshape_and_cache = original_device_reshape_attr
        torch_npu._npu_reshape_and_cache = original_npu_reshape
        if original_scatter is not None:
            torch_npu.npu_scatter_pa_kv_cache = original_scatter


def _build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = {
        "model": args.model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    return kwargs


def _write_outputs(output_dir: Path, args: argparse.Namespace, captures: list[Capture], generations: list[Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_path = output_dir / "dump.pt"
    metadata_path = output_dir / "metadata.json"

    metadata = {
        "model": args.model,
        "source": args.source,
        "prompts": args.prompt or ["Hello, my name is"],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "num_captures": len(captures),
        "slot_mapping_semantics": "slot = block_idx * block_size + block_offset; -1 means padding/unused token.",
        "captures": [capture["summaries"] | {"source": capture["source"]} for capture in captures],
        "outputs": [
            {
                "prompt": output.prompt,
                "text": output.outputs[0].text if output.outputs else "",
            }
            for output in generations
        ],
    }

    torch.save({"metadata": metadata, "captures": captures}, dump_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote KV-cache dump to: {dump_path}")
    print(f"Wrote readable metadata to: {metadata_path}")


def main() -> None:
    args = _parse_args()
    _configure_source(args.source)

    from vllm import LLM, SamplingParams

    prompts = args.prompt or ["Hello, my name is"]
    captures: list[Capture] = []
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)

    llm = LLM(**_build_llm_kwargs(args))
    with _capture_kv_writes(captures, args.max_captures):
        generations = llm.generate(prompts, sampling_params)

    if not captures:
        raise RuntimeError("No KV-cache writes were captured. Check that this run used the Ascend attention path.")

    _write_outputs(Path(args.output_dir), args, captures, generations)


if __name__ == "__main__":
    main()
