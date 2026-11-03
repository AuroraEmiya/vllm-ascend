# Decode head projection and contiguous output trial

This is round 2 of the newly authorized ten-round decode budget. Round 1's
matmul spelling is retained as a control condition, without a performance
claim. The historical ten rounds and their evidence remain unchanged.

Only the B>1 head route changes in the candidate wrapper:
`torch.bmm(head_q, weight_uk).permute(1, 0, 2).contiguous()`. B1 keeps its
previous bmm/view route. All C inputs, weights and outputs stay BF16; the
down/up projections, prepared weights, static ND/NZ route, Ascend C V,
cache semantics and accuracy thresholds stay unchanged. No native rebuild
is required.

The public `aclnnMlaProlog` contract requires contiguous query outputs.
A view of a contiguous `[N,B,512]` bmm output has batch-major strides
`[512,B*512,1]`, which is noncontiguous when B>1 and N>1. The candidate
therefore includes the explicit `contiguous()` operation inside every
forward and Graph. It may add execution tasks; retain and classify all of
them in the full profiler. No copy is excluded from complete latency.

The round-1 trace measured head costs of 9.974/10.004/11.403 us for
B2/B16/B128. Historical independently authored round-6 traces measured bmm
at 6.935/6.468/7.553 us. This suggests only about 3.04/3.54/3.85 us of
layout overhead can be tolerated, before any current-runtime differences.
The historical custom Restore kernel cost 8.693 us at B128; it is not used
here. This round tests whether the public contiguous operation is cheaper.
These task times motivate the experiment and are not complete-Graph scores.
Eliminating the entire head stage would give only about 1.16-1.28 times
speedup in the task-sum model, so this change alone does not promise 1.5x.

The pinned control `baseline_serial_592d3963.py` is the exact round-1 wrapper
from published commit `592d3963e3ef59fc147ef88ab0f114f35b731776` (same tree as
local `370ce5e4f8c3aa0e67fc14a0eca38f9911664901`). Canonical LF SHA256:
`0ee2fe6008ea62b83d1b815249d2517b3d39550e587850858a60ad1f07e7ba8c`.
The unchanged 15-case manifest SHA256 is
`824f1b88598244182516c164dc9a369307ce89a2d1bf20b4887a8092a9528858`.
No new V3 shape is treated as an implemented serial case.

Run from the native checkout root with the existing native extension/custom
OPP on the Python/library paths. Replace PUBLISHED_SHA and use distinct output
paths for every attempt. The parent owns publication and all remote work.

```bash
# CPU-only plan and pinned workload check.
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --plan \
  --comparison head-layout --mode head-micro --case-id decode-b128

# Full original precision, Graph, independent caches and same-address updates.
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison head-layout \
  --mode verify --source-commit PUBLISHED_SHA --out RESULTS/control-candidate-verify.json

# Paired complete Graph: 20 AB/BA groups and ten complete forwards per Graph.
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison head-layout \
  --mode paired --case-id decode-b16 --source-commit PUBLISHED_SHA \
  --out RESULTS/control-candidate-b16-paired.json

# Diagnostic only: identical BF16 head Q/weight for both complete head routes.
# Candidate contiguous operation is inside the Graph. Records input/output
# strides, raw/permuted views, copy requirement and BF16 FP64-reference accuracy.
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison head-layout \
  --mode head-micro --case-id decode-b16 --source-commit PUBLISHED_SHA \
  --out RESULTS/head-micro-b16.json

# Optional control trace; current full candidate uses unchanged validate below.
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison head-layout \
  --mode profile --implementation control --case-id decode-b16 \
  --source-commit PUBLISHED_SHA --out RESULTS/control-profile-b16.json

# All 15 fixed cases versus the unchanged fused V1 black box.
python3 benchmarks/kernels/mla_prolog/validate.py --no-timing \
  --baseline-route aclnn --source-commit PUBLISHED_SHA --out RESULTS/verify.json

# Repeat for the eight acceptance IDs, with full serial and fused profiling.
python3 benchmarks/kernels/mla_prolog/validate.py --case-id decode-b16 \
  --baseline-route aclnn --source-commit PUBLISHED_SHA --out RESULTS/paired-b16.json
python3 benchmarks/kernels/mla_prolog/validate.py --profile serial --case-id decode-b16 \
  --baseline-route aclnn --source-commit PUBLISHED_SHA --out RESULTS/profile-b16.json
```

Full-profile markers retain `implementation=serial` and distinguish the control
and candidate with `trial_implementation`. Classify actual matrix tasks as C
and every other execution task, including layout copies, as V. Graph control
and wait tasks stay excluded from C+V. Determine kernel names and matrix dtypes
from the actual trace. Complete fused/serial Graph latency remains the target
metric. Reject this candidate on any accuracy, cache, output-layout or Graph
failure, or if the added layout operation erases the complete-Graph benefit.

The historical projection harness defaults remain available for CPU plans and
projection microbenchmarks. Device execution of its complete projection trial
requires its round-1 wrapper, so a newer head candidate cannot be mislabeled
as a projection-only comparison. CPU contract checks are separate from the
required parent-owned A3 system checks.
