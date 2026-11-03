# Decode within-core gamma reuse trial

This is round5 of the additional ten-round decode budget. Round4 is retained:
all15 correctness/Graph/cache checks passed, and control/candidate complete
Graph ratios improved at all eight acceptance points (1.0070-1.0327x).
Round5's direct control is its exact published wrapper:
`8b0d4da8d4897b3099736689c63ae7baea1dff3e`, tree
`5ad8290ba72476b98d193c36c2fd6561e6e230e6`.
The pinned snapshot remains mode4; do not use round1 as this trial's direct control.

Round4 full traces show norm/cache/RoPE8.735/10.632/14.451us at B32/64/128.
Normalize currently reloads and casts BF16 gamma for each work unit. On the
measured A3, blockDim40 strides preserve query/KV parity and let each core
process several units. The single round5 hypothesis reuses that core's FP32
gamma in UB across those units. Expected avoided gamma DMA at B32/64/128 is
49,152/180,224/442,368 bytes, plus the corresponding repeated BF16-to-FP32 UB
casts. All matrix GM inputs, weights and outputs remain BF16.

New mode5 stores gamma in the existing `f1[64:64+width]`, not at element0.
Query width1536 reaches element1599, within the2048-element allocation; KV
width512 reaches575. KR RoPE overwrites only `f1[0:64]` with sine values,
so its temporary calculations cannot corrupt the cached gamma. Other scratch
buffers and the norm inverse broadcast do not write this cached region.
A scalar width tag starts at0 in every kernel Init, then reloads on a width
change. A platform with an odd block count can alternate Q/KV and simply
reloads, preserving correctness. The tag never survives a kernel launch:
same-address gamma changes must be observed by the next Graph replay.
No allocation, persistent tensor, preparation cost or cross-core dependency
is added. Existing V/MTE buffer-reuse events and math barriers are retained.

The candidate selects mode5 only for B>=32; B<32 keeps the retained mode4.
Mode4's load/cast algorithm and original math remain the control in the same
rebuilt package. Reduction, epsilon, Rsqrt, inverse broadcast, gamma multiply
and BF16 RINT order are unchanged. All C interfaces and layouts, prepared
weight/refresh contracts, head route, second V stage, cache/output contracts,
original15 cases and thresholds remain those of round4. B31/B32/B33 protect
the explicit threshold.

The expected benefit is a smaller norm stage and complete Graph at B>=32.
Keep or reject that specific range from the measured AB/BA pairs and full
profiles; any frozen correctness failure rejects the path. The complete Graph
must include every execution task. Do not count earlier GMM or layout-copy
rollbacks as a new gain. There is no new performance result in this local source
handoff and no new V3 domain implementation.

The own custom `x_mla_vector` package must be rebuilt/installed to accept mode5.
The Torch adapter ABI is unchanged; record the newly loaded package hashes.
The parent handles GitHub publication, Docker Git fetch/pull, build and A3 runs.

```bash
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --plan \
  --comparison norm-gamma-cache --mode paired --case-id decode-b128

# First affected smoke, followed by the complete15 fixed cases.
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison norm-gamma-cache \
  --mode verify --case-id decode-b32 --source-commit PUBLISHED_SHA --out NEW.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison norm-gamma-cache \
  --mode verify --source-commit PUBLISHED_SHA --out NEW.json

#20 AB/BA pairs,10 complete forwards/Graph: fixed round4 mode4 versus candidate.
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison norm-gamma-cache \
  --mode paired --case-id CASE_ID --source-commit PUBLISHED_SHA --out NEW.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison norm-gamma-cache \
  --mode profile --implementation candidate --case-id CASE_ID \
  --source-commit PUBLISHED_SHA --out NEW.json
# Repeat profile with --implementation control as needed.

python3 benchmarks/kernels/mla_prolog/validate.py --no-timing \
  --baseline-route aclnn --source-commit PUBLISHED_SHA --out NEW.json
python3 benchmarks/kernels/mla_prolog/validate.py --case-id CASE_ID \
  --baseline-route aclnn --source-commit PUBLISHED_SHA --out NEW.json
```

Use the generic shape-aware full-task analyzer. Classify observed matrix routes
and every other execution task; C/V sums remain separate from Graph Event time.
