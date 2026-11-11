"""CANNBot IR workflow, adapted to the installed A3 compiler's exact options.

Replay the real installed backend's final stage on the selected cached bcmlir.
Only add IR printing. Compare the returned binary hash against the NPU replay.
This avoids substituting the skill's 950-oriented default pipeline on A3.
"""
import dataclasses
import hashlib
import json
import re
from pathlib import Path

import torch
import torch_npu

from triton.backends.ascend import compiler

root = Path('results/optimization/next/selected-ir-row-weights')
selected = json.loads((root / 'selection.json').read_text())
real_run = compiler.subprocess.run
records = []
for entry in selected:
    directory = root / (entry['case_id'] + '_' + entry['impl'])
    metadata = dict(entry['metadata'])
    options = compiler.NPUOptions(**{f.name: metadata[f.name] for f in dataclasses.fields(compiler.NPUOptions) if f.init and f.name in metadata})
    source = next(directory.glob('*.bcmlir'))
    original = next(directory.glob('*.npubin'))
    calls = []

    def run_with_print(args, *pos, **kwargs):
        cmd = list(args)
        if str(cmd[0]).endswith('bishengir-compile'):
            cmd.append('--mlir-print-ir-after-all')
            (directory / 'compiler-command.json').write_text(json.dumps(cmd, indent=2))
            result = real_run(cmd, *pos, **kwargs)
            text = ''.join(v.decode(errors='replace') if isinstance(v, bytes) else (v or '') for v in (result.stdout, result.stderr))
            (directory / 'full-pass-dump.log').write_text(text)
            # Keep the latest valid HIVM/LLVM block, and its exact pass marker.
            blocks = re.split(r'(?=// -----// IR Dump After)', text)
            useful = [s for s in blocks if 'hivm.hir.' in s or 'llvm.' in s]
            if useful:
                (directory / 'last_pass.mlir').write_text(useful[-1])
            calls.append(dict(command=cmd, dumped=bool(useful), bytes=len(text)))
            return result
        return real_run(cmd, *pos, **kwargs)

    compiler.subprocess.run = run_with_print
    try:
        result = compiler.linalg_to_bin_enable_npu_compile_A2_A3(source.read_text(), metadata, options)
        if isinstance(result, tuple): result = result[0]
        sha = hashlib.sha256(result).hexdigest()
        original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
        record = dict(case_id=entry['case_id'], impl=entry['impl'], binary_sha256=original_sha,
                      recompiled_binary_sha256=sha, byte_identical=sha == original_sha,
                      final_ir=(directory / 'last_pass.mlir').exists(), compiler_calls=calls)
    except Exception as exc:
        record = dict(case_id=entry['case_id'], impl=entry['impl'], error_type=type(exc).__name__, error=str(exc), compiler_calls=calls)
    finally:
        compiler.subprocess.run = real_run
    records.append(record)
    (root / 'lowering-audit.json').write_text(json.dumps(records, indent=2))
    print(json.dumps({k:v for k,v in record.items() if k != 'compiler_calls'}), flush=True)
