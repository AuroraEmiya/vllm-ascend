# Additional round10: extend retained norm7 to B16..31

This final candidate lowers only `NORM_OUTPUT_OVERLAP_MIN_BATCH` from32 to16.
The control is the exact round7 wrapper published as
`7ada452c33c5748fe141a786250c2b30345b637f`, stored in
`baseline_serial_7ada452c.py` with canonical SHA256
`1030da1293222b46d337840a496acc87d0bae152c532007121368da507096d56`.
Candidate Python source equals that snapshot except for this one constant.

Round9's row-major up-ND was rejected: control/candidate complete-Graph ratios
at B16/32/64/128 were0.945248/0.954453/0.986482/0.964643. Its full profile
passed with480 observed BF16 matrix tasks, no independent conversion/copy, and
up15.968/16.395/17.640/20.499us, all actual MatMulV2. Both round10 classes
restore exact round7 transposed-up storage, refresh and forward. They also
retain norm7>=32 and split6 only32..64 before applying the single new threshold.
Neither rejected-layout nor rejected-split rollback counts as round10 gain.
Round9 setup/refresh allocation and peak telemetry remains in the common harness.

The round7 norm stage took6.767us at B16. Even when a core owns just one
KV work unit, that unit executes512-element KV normalization/output and then
64-element KR RoPE/output. Mode4's `MTE3_MTE2` waits for the KV output before
loading independent KR/sin/cos inputs into b0/b1/b2. Existing mode7 instead
lets these loads and FP32 arithmetic proceed, then protects shared b3 with
`MTE3_V` immediately before the KR output Cast. This is a concrete dependency
window; no general claim of overlap speedup is assumed. Normalization already
uses mode7 at B>=32 with measured modest benefits. Its gamma cache and FP32
arithmetic remain exactly the existing algorithm, including cache invalidation
on every launch. This trial evaluates its application to a smaller batch domain.

| B | Exact control norm | Candidate norm | Split in both |
| --- | --- | --- | --- |
| 1..15 | mode4 | mode4 | mode1 |
| 16..31 | mode4 | mode7 | mode1 |
| 32..64 | mode7 | mode7 | mode6 |
| 65..128 | mode7 | mode7 | mode1 |

All public matrix APIs, input/output BF16 boundaries, operand strides, prepared
weight formats/ownership, head route, FP32 V arithmetic, UB allocation and
output/cache semantics are exact round7. The existing host treats mode7 as
merged norm, accepts rows1..128, sets units=2*B and blockDim=min(units,AIV cores).
There is no host mode7 restriction at B32. B16 therefore has32 norm units;
B31 has62. Every launched core has at least one unit. If a core gets multiple
units, existing V_MTE2 input reuse, MTE3_V output reuse and gamma-width tagging
remain active. No mode8 is selected, and no host/kernel/native source changes.

The following single-KV-unit lifetime is unchanged source code. Calls to
`Sync<EVENT>()` enqueue `SetFlag<EVENT>(EVENT_ID0)` and matching
`WaitFlag<EVENT>(EVENT_ID0)` together. No wait relies on a previously signaled
event or a completion from an earlier Graph replay.

| Phase | Buffers and dependency |
| --- | --- |
| Entry | Init resets cachedGammaWidth=0. Process V_MTE2 protects input reuse, even with no prior input work. |
| KV input | DMA writes b0/b1; MTE2_V precedes their FP32 Cast. Gamma is read from f1[64:] in mode7. |
| KV output | Initial MTE3_V is balanced even with an empty prior MTE3 queue. Cast writes b3; V_MTE3 precedes the512-element output DMA reading b3. |
| KR input | V_MTE2 orders previous V consumers before b0/b1/b2 overwrite. KR DMA can proceed while MTE3 reads b3; these buffers are disjoint. MTE2_V precedes KR Cast. |
| KR arithmetic | RoPE uses f1[:64], preserving cached gamma at f1[64:]. Existing vector barriers order all FP32 producers/consumers. |
| KR output | MTE3_V waits for the KV DMA to finish reading b3 before KR Cast overwrites b3. V_MTE3 then orders Cast before KR output DMA. |
| Next unit or exit | Next unit repeats V_MTE2 and MTE3_V at exact reuse points. Process final MTE3_S waits for the last KR/query DMA before returning. No pending output requires a next unit to drain it. |

The CPU dependency model checks this exact single-unit window with per-engine
FIFO ordering and independent cross-engine edges. Removing input reuse, b3
reuse or output-ready edges must expose a buffer hazard; removing the final
drain must leave output unordered with return. Existing multi-unit and gamma
region tests remain. Boundary source checks cover B15/16/17 and B31/32 as part
of B1..128, without adding or replacing frozen device cases. These are source
and CPU checks, not an A3 execution claim. Prior synchronization scanners still
have exit1 findings; original findings and manual explanations stay retained.

No rebuild is required. Reuse and verify the same round8 V binary SHA256
`a49ab3ef782d648ff44a204f0c2469df38977883df2f72753f17ebe2abf05dad`
and Torch extension SHA256
`c1328531de39083bac8094010c6e47cda8f766d32b8c6856d54eb9bb27203ef4`
before and after the parent-operated run. All remote health, ownership and
device gates remain mandatory. Local child work does not publish or use NPU.

```bash
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison norm-overlap-b16 --plan --mode paired
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison norm-overlap-b16 --mode verify --case-id decode-b16 --source-commit PUBLISHED_SHA --out NEW_SMOKE.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison norm-overlap-b16 --mode verify --source-commit PUBLISHED_SHA --out NEW_ALL15.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison norm-overlap-b16 --mode paired --case-id CASE_ID --source-commit PUBLISHED_SHA --out NEW_PAIRED.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison norm-overlap-b16 --mode profile --implementation candidate --case-id CASE_ID --source-commit PUBLISHED_SHA --out NEW_PROFILE.json
python3 benchmarks/kernels/mla_prolog/validate.py --baseline-route aclnn --no-timing --source-commit PUBLISHED_SHA --out NEW_FUSED_VERIFY.json
python3 benchmarks/kernels/mla_prolog/validate.py --baseline-route aclnn --case-id CASE_ID --source-commit PUBLISHED_SHA --out NEW_FUSED_PAIRED.json
```

Run the unchanged15 cases including acceptance B16 and guard-tail-b31, original
FP64 thresholds, separate initial caches, untouched cache-region checks and
changed/restored same-address replay. Timing retains20 AB/BA pairs of10 complete
forwards per Graph. Full profiles retain all execution tasks and actual matrix
types/dtypes/formats. KEEP requires a useful consistent B16 full-Graph benefit,
no material guard31 regression and all15 accuracy/state checks. Otherwise
restore the exact r7 wrapper. The budget ends at this candidate; no round11.
