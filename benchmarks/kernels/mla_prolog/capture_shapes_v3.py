# SPDX-License-Identifier: Apache-2.0
"""Observe public MlaPrologV3 shapes; no serial algorithm or timing changes."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import traceback
from pathlib import Path

import torch
import torch_npu

OP = "npu::npu_mla_prolog_v3"
INPUT_NAMES = (
    "token_x",
    "weight_dq",
    "weight_uq_qr",
    "weight_uk",
    "weight_dkv_kr",
    "rmsnorm_gamma_cq",
    "rmsnorm_gamma_ckv",
    "rope_sin",
    "rope_cos",
    "kv_cache",
    "kr_cache",
)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def matrix():
    cases = []
    for label, batch, hidden, rank, dim, heads, norm, blocks, source in (
        ("shared-b1", 1, 7168, 1536, 128, 32, False, None, "original public-contract point"),
        ("shared-b17", 17, 7168, 1536, 128, 32, True, None, "original public-contract point plus V3 optional output"),
        ("glm-b1", 1, 6144, 2048, 192, 32, True, 1325, "formal template TP2/DP2 case01"),
        ("glm-b17", 17, 6144, 2048, 192, 32, True, 1325, "formal template TP2/DP2 case04"),
        ("glm-b16-n8", 16, 6144, 2048, 192, 8, True, 1214, "formal template TP8/DP2 case12"),
        ("glm-b16-n64", 16, 6144, 2048, 192, 64, True, 1477, "formal template TP1/DP1 case08"),
        ("shared-b256", 256, 7168, 1536, 128, 32, False, None, "public API batch expansion; serial B limit remains128"),
        ("boundary-he1024", 17, 1024, 1536, 128, 32, False, None, "public API hidden-size boundary"),
        ("boundary-he8192", 17, 8192, 1536, 128, 32, False, None, "public API hidden-size boundary"),
    ):
        cases.append(
            dict(
                id=label,
                B=batch,
                S=1,
                He=hidden,
                Hcq=rank,
                D=dim,
                Hckv=512,
                Dr=64,
                N=heads,
                Nkv=1,
                block_size=128,
                block_num=blocks or math.ceil((3 * batch + 2) / 128),
                query_norm_flag=norm,
                source=source,
                seed=20261103 + len(cases),
            )
        )
    return {
        "kind": "finite_public_V3_operator_capture",
        "cases": cases,
        "not_full_model_capture": True,
        "not_performance_iteration": True,
        "numeric_equivalence_to_serial": "NOT_EVALUATED",
        "optimization_budget_used": 0,
    }


def tensor_meta(value):
    if value is None:
        return None
    result = dict(
        shape=list(value.shape),
        dtype=str(value.dtype),
        device=str(value.device),
        stride=list(value.stride()),
        storage_offset=value.storage_offset(),
        storage_bytes=value.untyped_storage().nbytes(),
        data_ptr=value.data_ptr(),
        is_contiguous=value.is_contiguous(),
    )
    if value.device.type == "npu":
        result["npu_format"] = int(torch_npu.get_npu_format(value))
    return result


def inputs(case):
    generator = torch.Generator().manual_seed(case["seed"])
    b, h, r, n, d = (case[k] for k in ("B", "He", "Hcq", "N", "D"))

    def normal(shape, scale=1.0):
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16).npu()

    x = normal((b, 1, h))
    wdq = torch_npu.npu_format_cast(normal((h, r), h**-0.5), 29)
    wuq = torch_npu.npu_format_cast(normal((r, n * (d + 64)), r**-0.5), 29)
    wuk = normal((n, d, 512), d**-0.5)
    wdkv = torch_npu.npu_format_cast(normal((h, 576), h**-0.5), 29)
    gq, gkv = normal((r,), 0.1) + 1, normal((512,), 0.1) + 1
    angles = torch.randn((b, 1, 32), generator=generator)
    sine = torch.cat((angles.sin(), angles.sin()), -1).to(torch.bfloat16).npu()
    cosine = torch.cat((angles.cos(), angles.cos()), -1).to(torch.bfloat16).npu()
    cache_shape = (case["block_num"], 128, 1)
    kv = torch.full((*cache_shape, 512), 0.125, dtype=torch.bfloat16, device="npu")
    kr = torch.full((*cache_shape, 64), -0.125, dtype=torch.bfloat16, device="npu")
    index_cpu = (torch.arange(b, dtype=torch.int64) * 3 + 1).reshape(b, 1)
    return [x, wdq, wuq, wuk, wdkv, gq, gkv, sine, cosine, kv, kr], index_cpu


def run(case, out):
    result = dict(
        case=case,
        status="STARTED",
        source_level="actual_A3_public_operator_capture",
        started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        performance="NOT_EVALUATED",
        numeric_equivalence_to_serial="NOT_EVALUATED",
        model_inference="NOT_RUN",
        optimization_iteration=False,
        tool="torch_npu.profiler.profile(record_shapes=True, CPU+NPU)",
    )
    dump(out / "result.json", result)
    try:
        torch.set_num_threads(4)
        torch.npu.set_device(0)
        torch.npu.config.allow_internal_format = True
        result["environment"] = dict(
            python=platform.python_version(),
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            device=torch.npu.get_device_name(0),
            visible_device_count=torch.npu.device_count(),
            schema=str(torch.ops.npu.npu_mla_prolog_v3.default._schema),
        )
        values, index_cpu = inputs(case)
        index = index_cpu.npu()
        kwargs = dict(
            cache_index=index,
            rmsnorm_epsilon_cq=1e-5,
            rmsnorm_epsilon_ckv=1e-5,
            cache_mode="PA_BSND",
            query_norm_flag=case["query_norm_flag"],
            weight_quant_mode=0,
            kv_cache_quant_mode=0,
            query_quant_mode=0,
            ckvkr_repo_mode=0,
            quant_scale_repo_mode=0,
            qc_qr_scale=1.0,
            kc_scale=1.0,
        )
        torch.npu.synchronize()
        result["inputs"] = dict(zip(INPUT_NAMES, map(tensor_meta, values)))
        result["cache_index"] = dict(
            metadata=tensor_meta(index), values=index_cpu.flatten().tolist(), distinct=True, in_range=True
        )
        result["scalar_kwargs"] = {k: v for k, v in kwargs.items() if k != "cache_index"}
        result["optional_tensor_kwargs"] = {
            k: None
            for k in (
                "dequant_scale_x",
                "dequant_scale_w_dq",
                "dequant_scale_w_uq_qr",
                "dequant_scale_w_dkv_kr",
                "quant_scale_ckv",
                "quant_scale_ckr",
                "smooth_scales_cq",
                "actual_seq_len",
                "k_nope_clip_alpha",
            )
        }
        handler = torch_npu.profiler.tensorboard_trace_handler(
            str(out / "profiler"), analyse_flag=True, async_mode=False
        )
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            on_trace_ready=handler,
        ):
            outputs = torch_npu.npu_mla_prolog_v3(*values, **kwargs)
            torch.npu.synchronize()
        result["synchronized_call_completed"] = True
        result["outputs"] = dict(
            zip(
                ("query", "query_rope", "dequant_scale_q_nope", "query_norm", "dequant_scale_q_norm"),
                map(tensor_meta, outputs),
            )
        )
        result["output_count"] = len(outputs)
        result["nonempty_output_finite"] = all(v is None or bool(torch.isfinite(v.float()).all()) for v in outputs)
        result["cache_state"] = {}
        slots = case["block_num"] * 128
        touched = torch.zeros(slots, dtype=torch.bool)
        touched[index_cpu.flatten()] = True
        for name, cache, width, fill in (("kv_cache", values[-2], 512, 0.125), ("kr_cache", values[-1], 64, -0.125)):
            current = cache.cpu().reshape(slots, width)
            result["cache_state"][name] = dict(
                metadata_after=tensor_meta(cache),
                untouched_exact=bool((current[~touched] == fill).all()),
                written_rows_changed=bool((current[touched] != fill).any()),
                written_finite=bool(torch.isfinite(current[touched].float()).all()),
            )
        traces = sorted((out / "profiler").rglob("trace_view.json"))
        matches, kernels = [], []
        for trace in traces:
            data = json.loads(trace.read_text(encoding="utf-8"))
            for event in data.get("traceEvents", []):
                name = event.get("name", "")
                if name == OP or name.startswith(OP + "."):
                    matches.append(event)
                if "mlaprologv3" in name.replace("_", "").lower():
                    kernels.append(dict(name=name, cat=event.get("cat"), args=event.get("args")))
        result["traces"] = [str(p) for p in traces]
        result["exact_dispatch_events"] = matches
        result["related_kernel_events"] = kernels
        result["input_shapes_recorded"] = any(
            any("shape" in str(k).lower() or "dim" in str(k).lower() for k in event.get("args", {}))
            for event in matches
        )
        result["exact_dispatch_hit"] = bool(matches)
        state_ok = all(
            v["untouched_exact"] and v["written_rows_changed"] and v["written_finite"]
            for v in result["cache_state"].values()
        )
        result["status"] = "CAPTURED" if matches and result["input_shapes_recorded"] and state_ok else "INCOMPLETE"
        result["artifact_sha256"] = {
            str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in traces
        }
        if result["status"] == "CAPTURED":
            columns = [
                "node_id",
                "name",
                "op_name",
                "ori_name",
                "source",
                "inputs_shape",
                "inputs_dtype",
                "outputs_shape",
                "outputs_dtype",
            ]
            ins = [*result["inputs"].values(), result["cache_index"]["metadata"]]
            outs = [m for m in result["outputs"].values() if m is not None]
            row = dict(
                node_id=case["id"],
                name="MlaPrologV3",
                op_name=OP,
                ori_name=OP,
                source="actual_synchronized_NPU_call_with_exact_dispatch_hit",
                inputs_shape=json.dumps([m["shape"] for m in ins]),
                inputs_dtype=json.dumps([m["dtype"] for m in ins]),
                outputs_shape=json.dumps([m["shape"] for m in outs]),
                outputs_dtype=json.dumps([m["dtype"] for m in outs]),
            )
            with (out / "output-forward-runtime.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerow(row)
    except Exception as exc:
        result.update(status="FAILED", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    result["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    dump(out / "result.json", result)
    print(json.dumps({"case": case["id"], "status": result["status"], "error": result.get("error")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = matrix()
    if args.write_manifest:
        dump(args.out / "capture-cases.json", manifest)
        return
    case = next(c for c in manifest["cases"] if c["id"] == args.case_id)
    if run(case, args.out)["status"] != "CAPTURED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
