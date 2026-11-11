"""Diagnose an unchanged fallback using actual public launches and cache binary IDs."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import runpy

import torch
from vllm_ascend.ops.triton.dcp import dcp_a2a

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True)
args = parser.parse_args()
fixture = runpy.run_path('/work/pr-combine-final-integration/benchmarks/benchmark_dcp_combine.py')
torch.npu.set_device(0)
recv, local, lse, expected = fixture['make_inputs']((41, 7, 128, 3, True, False))
identity = {}


def capture(fn, grid, *positional, **keywords):
    binary = fn[grid](*positional, **keywords)
    meta = dict(vars(binary.metadata)) if hasattr(binary.metadata, '__dict__') else binary.metadata._asdict()
    folder = Path('/work/.triton-cache') / base64.b32encode(bytes.fromhex(meta['hash'])).decode().rstrip('=')
    identity.update(
        kernel=fn.__name__, grid=grid, metadata=meta,
        binaries={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in folder.glob('*.npubin')},
        positional=[dict(dtype=str(v.dtype), shape=list(v.shape), stride=list(v.stride()), ptr_mod_2m=v.data_ptr() % (2**21))
                    if isinstance(v, torch.Tensor) else v for v in positional], keywords=keywords,
    )


dcp_a2a._launch_kernel = capture


def invoke():
    return dcp_a2a.fused_dcp_lse_combine(recv, 128, 1, local_output=local, local_lse=lse)


actual = invoke().cpu().double()
assert torch.isfinite(actual).all()
assert ((actual - expected).abs() <= 2e-5 + .004 * expected.abs()).double().mean() >= .99
windows = [fixture['measure'](invoke, 100, 10) for _ in range(3)]
out = dict(module=dcp_a2a.__file__, shape=[41, 7, 128, 3, True, False], windows=windows, identity=identity)
Path(args.out).write_text(json.dumps(out, indent=2, default=str) + '\n')
print(json.dumps(out, default=str))

