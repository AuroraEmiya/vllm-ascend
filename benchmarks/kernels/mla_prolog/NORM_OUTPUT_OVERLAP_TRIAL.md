# Decode norm output overlap trial

This is round7 of the additional ten-round decode budget. The source control
is published round6 commit `f1f0dba2d1202579b8bea9b2d03f480d7fb003ed`, tree
`a6fe84eb3d95474d34e56f2d5b559998d361a4b2`. Its wrapper SHA256 is
`21279de14dfd20af8078b15c680982104de9b29aaee67fbee2cd9cabe8dd6508`.
Round6 passed all15 checks, but its split DMA overlap improved B32/64 and
regressed B128. The retained control therefore changes only that selection
to32<=B<=64. The derived snapshot `baseline_serial_f1f0dba2_retained.py` has
SHA256 `a2c2e333f94e9e62f3f13bdaa644fc1658cc9fffbf3777a71061d27f23d1fc44`.
The candidate makes the same retention change. Restoring mode1 at B128 is
not counted as a new round7 gain.

The round6 full profiles measured norm/cache/RoPE8.866/10.779/13.815us at
B32/64/128. Mode5 waits for output MTE3 before the next input MTE2, both at
loop entry and between KV normalization and KR RoPE. Those input transfers
write `b0`/`b1`/`b2`, while the preceding output transfer only reads `b3`.
The round7 hypothesis moves this output-reuse wait to the exact `b3` overwrite.
New norm mode7, selected only for B>=32, uses `MTE3_V` just before the output
Cast, permitting input DMA and FP32 UB arithmetic to overlap the prior output.
It retains mode5's math, gamma cache, units and allocations. The measured
norm stage is an upper bound on the saving; no substage wait time is assumed.

| Producer / storage | Next consumer or reuse | Mode7 ordering |
| --- | --- | --- |
| Previous V reads of `b0`/`b1`/`b2` | Next input MTE2 writes those buffers | Existing loop-entry `V_MTE2`; also retained at Rope entry |
| Input DMA writes `b0`/`b1`/`b2` | FP32 Cast reads input | Existing `MTE2_V` |
| Previous output MTE3 reads `b3` | Next input MTE2 writes `b0`/`b1`/`b2` | Distinct buffers; mode7 removes this input-blocking `MTE3_MTE2` wait |
| Previous output MTE3 reads `b3` | Next norm output Cast writes `b3` | New `MTE3_V` at the exact overwrite |
| KV output MTE3 reads `b3` | KR input DMA and FP32 work | Distinct input/output buffers; Rope keeps `V_MTE2` and `MTE2_V` |
| KV output MTE3 reads `b3` | KR output Cast writes `b3` | New Rope `MTE3_V` before the output Cast |
| FP32 writes `f0`/`f4` | Norm/RoPE output Cast reads them | Original `PIPE_V` barriers and arithmetic order |
| BF16 output Cast writes `b3` | Output DMA reads `b3` | Existing `V_MTE3` |
| Gamma Cast writes `f1[64:64+width]` | RMS multiply reads cached gamma | Original vector barriers; mode7 uses mode5's width-tagged branch |
| KV gamma remains in `f1[64:576]` | KR writes sine to `f1[0:64]` | Disjoint windows; gamma ends at1600 for query width1536 |
| Last MTE3 output reads `b3` | Process returns | Mode7 `MTE3_S` drain |

The same rules protect two norm units on one core, the KV-to-KR transition,
odd core counts that change gamma width, and the final unit. Init clears the
width tag on each invocation, preserving same-address gamma updates. A new
output's `V_MTE3` follows its `MTE3_V`; the old output has already received its
own `V_MTE3`, so the two waits do not introduce a cyclic queue dependency.
Mode5 retains the old waits exactly. Split mode6 remains a separate dispatch;
host shape inference and tiling recognize7 as merged norm and6 as split.

UB remains59,424 bytes. Public BF16 down/up/head matrices, ND/NZ preparation
and refresh, strides, continuous outputs, cache mutation, all15 frozen cases
and precision thresholds are unchanged. B<32 stays on norm mode4. The parent
will rebuild/install only the own `x_mla_vector` package; the Torch adapter ABI
is unchanged. No compiled-device result is claimed by this local handoff.

The sync scanners retain every candidate in project output. The template
`Sync<EVENT>` helper is mechanically expanded in a separate inspection copy.
Manual buffer reasoning and CPU lifetime checks supplement the raw output;
SDK headers and compiled behavior are outside the Windows static check.
Queue-event semantics follow the official
[SetFlag/WaitFlag control-unit description](https://www.hiascend.com/doc_center/source/zh/canncommercial/800/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0011.html).

Keep or reject a concrete B>=32 range using20 AB/BA pairs of10 complete
forwards per Graph, plus all execution tasks in full profiles. A frozen
correctness or cache/state-update failure rejects the candidate. Use the
same rebuilt binary for the pinned mode5 control and new mode7 candidate.
No V3 shape domain or new performance case is introduced.

```bash
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --plan \
  --comparison norm-output-overlap --mode paired --case-id decode-b128

python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison norm-output-overlap \
  --mode verify --case-id decode-b32 --source-commit PUBLISHED_SHA --out NEW.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison norm-output-overlap \
  --mode verify --source-commit PUBLISHED_SHA --out NEW.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison norm-output-overlap \
  --mode paired --case-id CASE_ID --source-commit PUBLISHED_SHA --out NEW.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison norm-output-overlap \
  --mode profile --implementation candidate --case-id CASE_ID \
  --source-commit PUBLISHED_SHA --out NEW.json
# Repeat profile with --implementation control for retained round6.

python3 benchmarks/kernels/mla_prolog/validate.py --no-timing \
  --baseline-route aclnn --source-commit PUBLISHED_SHA --out NEW.json
python3 benchmarks/kernels/mla_prolog/validate.py --case-id CASE_ID \
  --baseline-route aclnn --source-commit PUBLISHED_SHA --out NEW.json
```

The parent owns GitHub publication, Docker Git synchronization, all device
work and round email. Record actual matrix dtypes and operation types rather
than assuming a MatMul version. Graph latency remains the primary measure;
C/V task sums stay separate, and fused MIX internals remain unknown.
