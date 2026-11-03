# Decode matmul interface trial

This is round 1 of the newly authorized ten-round decode budget. The historical
ten rounds, fixed 15 cases, accuracy thresholds and performance evidence remain
unchanged. No V3 parameter domain is implemented by this trial.

The only computation change is in `vllm_ascend/ops/mla_prolog_serial.py`: the two
ND projections use `torch.matmul(x, nd_weight.t())` in place of `F.linear`, and
the two NZ projections use `torch.matmul` in place of `torch.mm`. The static
B>=16 and B%16==0 route, prepared weights, head projection and Ascend C V are
unchanged. The old route strings describe the retained layout selector, not
the replacement API name. No native rebuild is required.

`baseline_serial_af0d5824.py` is the exact self-owned wrapper obtained with
`git show af0d582472abb826d3ea6cbe37c75e7f49f9481c:vllm_ascend/ops/mla_prolog_serial.py`.
Its canonical LF SHA256 is
`654bb8fe5761e19edfdaf2e8a595c9d599cf10a7e40f1f410d44008b2107af51`.
The frozen cases SHA256 is
`824f1b88598244182516c164dc9a369307ce89a2d1bf20b4887a8092a9528858`.
The harness verifies both before importing device libraries.

Run from the native repository root, with the published checkout as the first
Python source path and the existing native extension/custom OPP available.
Replace `PUBLISHED_SHA` and use a new result path for every attempt:

```bash
# Local/CPU contract plan; imports no torch or NPU libraries.
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --plan

# All 15 old/new cases: shared FP64 oracle and original BF16/cache thresholds.
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --mode verify \
  --source-commit PUBLISHED_SHA --out RESULTS/old-new-verify-attempt01.json

# Same condition complete Graph, 20 alternating AB/BA pairs, ten forwards/Graph.
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --mode paired \
  --case-id decode-b16 --source-commit PUBLISHED_SHA \
  --out RESULTS/old-new-b16-paired-attempt01.json

# C-only microbenchmark, same activation and physical ND/NZ storage per stage.
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --mode micro \
  --case-id decode-b16 --source-commit PUBLISHED_SHA \
  --out RESULTS/micro-b16-attempt01.json

# Run separately under the existing complete msprof collector, one case/process.
# Replace original with matmul for the trial. Last two Graph replays are marked.
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --mode profile \
  --implementation original --case-id decode-b16 \
  --source-commit PUBLISHED_SHA --out RESULTS/original-b16-profile-attempt01.json

# Candidate versus the public fused V1 baseline uses the unchanged validator.
python3 benchmarks/kernels/mla_prolog/validate.py --no-timing \
  --baseline-route aclnn --source-commit PUBLISHED_SHA \
  --out RESULTS/matmul-fused-verify-attempt01.json
python3 benchmarks/kernels/mla_prolog/validate.py --case-id decode-b16 \
  --baseline-route aclnn --source-commit PUBLISHED_SHA \
  --out RESULTS/matmul-fused-b16-paired-attempt01.json
python3 benchmarks/kernels/mla_prolog/validate.py --profile serial \
  --case-id decode-b16 --baseline-route aclnn --source-commit PUBLISHED_SHA \
  --out RESULTS/matmul-b16-profile-attempt01.json
```

Use `--profile fused` for the independent full fused trace. The full profiler
must retain every matrix task and other execution task, with Graph controls
excluded from C+V sums. `matmul_trial.py` keeps the original profile marker
(`implementation=serial`) and adds `trial_implementation=original|matmul`.
Determine actual OP Type/kernel routing from the trace; no V2/V3 assumption.

The old/new path uses independent same-value caches, validates eager and Graph
outputs, changes inputs/matrix values/gamma/RoPE/index and cache contents at the
same addresses, explicitly refreshes both prepared objects, then restores and
checks again. It reuses `validate.build_case`, `oracle` and `check_outputs`.
Preparation and refresh are recorded outside Graph timing. The same input
weights feed both objects; each owns its original NZ and ND copies.

For each micro stage, `linear(x, rhs.t())`, `mm(x, rhs)` and `matmul(x, rhs)` use
the same input tensor and underlying weight storage without new format casts.
ND `rhs` is the view of the existing prepared transposed weight; NZ `rhs` is
the existing prepared NZ weight. Shapes, strides, formats and storage pointers
are recorded for both calling conventions. The up-stage activation is a seeded
BF16 tensor of normalized-Q shape, shared across APIs. Each supported pair has
20 AB/BA samples with ten matrix calls/Graph. Graph/precision/interface failures
are retained with null comparisons, including possible NZ-linear failures.
Micro matrix-call latency is distinct from complete-forward latency.

Before any remote execution, the parent must recheck live A3 health, process
ownership, disk and device/container mappings using the remote skill. Use only
the existing project container on physical12; do not run concurrent formal
timing/profile or change other task containers. Historical full CI remains an
environment timeout; the new CPU contract tests and own Ruff checks are separate.
