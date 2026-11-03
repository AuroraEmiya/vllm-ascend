# Decode RMS inverse broadcast trial

This is round4 of the additional ten-round decode budget. Round3 GMM passed
all15 fixed cases but made affected ND merged-down tasks approximately
40.5-49.6us, versus the round1 path's approximately16.6-25.9us. The candidate
restores the round1 matrix path; this rollback does not count as a new gain.
Round2's rejected bmm plus contiguous head copy also stays rolled back.
The direct control remains round1 commit
`592d3963e3ef59fc147ef88ab0f114f35b731776`.

The only new hypothesis concerns applying the inverse inside RMS Normalize.
Round1 B128 norm/cache/RoPE cost14.032us and round3 cost13.312us. The existing
code waits for V, reads the first FP32 inverse into a scalar register with
`GetValue(0)`, then calls `Muls`. Round4 keeps the same reduction, epsilon,
`Rsqrt`, FP32 multiply order and BF16 rounding. It replaces that scalar path
with `Brcb<float>` into the already allocated, no-longer-needed squared-input
buffer, followed by broadcast `Mul`:

```cpp
Brcb(f4, f3, 1, {1, 8});
// PIPE_V barriers surround the producer/consumer dependency.
const BinaryRepeatParams inverseBroadcast(1, 1, 0, 8, 8, 0);
Mul(f0, f0, f4, uint64_t(64), uint8_t(width / 64), inverseBroadcast);
```

Brcb puts the exact first inverse value in the first eight FP32 lanes.
The binary strides reuse this first32-byte block for every block and repeat.
Widths512/1536 use8/24 repeats and have no tail. The other Brcb output blocks
are not consumed. Source and destination buffers are distinct, sufficiently
sized and32-byte aligned; no new UB buffer, GM tensor, cast or transfer exists.
The [official Brcb API](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0089.html)
documents A3 float support, eight source elements per repeat, block expansion
and disjoint input/output. The
[official binary stride structure](https://www.hiascend.com/doc_center/source/zh/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0013.html)
identifies the block and repeat parameters used here.

Mode3 retains the original scalar-read normalization in the same rebuilt own
custom package. New internal mode4 uses the broadcast and otherwise has the
same merged-input shapes, work units and cache writes. The candidate Python
wrapper differs from round1 only by passing mode4 for its first V stage;
the pinned control snapshot still passes mode3. This avoids accidentally
changing both control and candidate when a new native package is loaded.
The Torch registration and adapter ABI stay the same. The own custom package
must be rebuilt and its loaded binary hashes recorded; do not reuse the old
mode0-3 package or claim that its mode4 failure is a candidate correctness result.

All public matrix activation/weight/output tensors remain BF16. All matrix
interfaces, ND/NZ preparation and refresh, row rule, head outputs, second V
stage, cache semantics, original15 cases and precision thresholds are unchanged
from round1. FP32 work is internal to the V UB. No new V3 domain is implemented.

The prediction is a smaller norm stage and a repeatable complete-Graph
improvement, particularly when each core handles several normalization units.
The trace does not isolate the scalar wait, so its cost is a hypothesis.
Even eliminating the entire B128 norm stage gives only about1.20x in the
round1 task-sum model, not1.5x. That bound is not Graph latency. Reject on any
precision, Graph, output/cache, stable refresh-address failure or no stable
complete-Graph improvement against the round1 algorithm. Reject rather than
relaxing thresholds. Keep the failed GMM and layout-copy evidence unchanged.

Run from the same CANN-enabled native checkout after rebuilding/installing the
own custom package. The parent handles publication, Docker Git, build, resource
checks and all device execution. No local NPU result is claimed.

```bash
# CPU plan; no torch import.
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --plan \
  --comparison norm-broadcast --mode paired --case-id decode-b128

# Smoke B1, then all15 correctness/Graph/state checks.
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison norm-broadcast \
  --mode verify --case-id decode-b1 --source-commit PUBLISHED_SHA --out NEW.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison norm-broadcast \
  --mode verify --source-commit PUBLISHED_SHA --out NEW.json

#20 AB/BA pairs,10 complete forwards per Graph. Capture both algorithms in
# the same rebuilt binary: fixed control mode3, candidate mode4.
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison norm-broadcast \
  --mode paired --case-id CASE_ID --source-commit PUBLISHED_SHA --out NEW.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison norm-broadcast \
  --mode profile --implementation candidate --case-id CASE_ID \
  --source-commit PUBLISHED_SHA --out NEW.json
# Repeat profile with --implementation control where needed.

# Unchanged public fused V1 baseline and original fixed cases.
python3 benchmarks/kernels/mla_prolog/validate.py --no-timing \
  --baseline-route aclnn --source-commit PUBLISHED_SHA --out NEW.json
python3 benchmarks/kernels/mla_prolog/validate.py --case-id CASE_ID \
  --baseline-route aclnn --source-commit PUBLISHED_SHA --out NEW.json
```

Use the generic full-task analyzer for actual matrix versions, all extra
execution tasks, BF16 shape-aware boundary audit and separate C/V diagnostic
sums. Full synchronized Graph Event latency remains the performance criterion.
Opaque fused/MIX internal arithmetic remains unknown.
