# Round9: isolate the up projection's ND orientation

Round9 tests one layout hypothesis in the additional ten-round decode budget.
The retained round7 profile places public matrix stages at roughly69..84% of
complete execution-task time. The up projection takes12.763/13.345/15.195/18.424us
at B16/32/64/128, with logical K1536 and N6144 for the acceptance head count32.
These are diagnostic stage sums; complete NPU Graph latency remains the metric.

The control is the exact round7 wrapper published as
`7ada452c33c5748fe141a786250c2b30345b637f`, whose snapshot is
`baseline_serial_7ada452c.py`, canonical SHA256
`1030da1293222b46d337840a496acc87d0bae152c532007121368da507096d56`.
Round8 split coefficient packing passed15 correctness cases but was rejected
after full Graph ratios of0.997226/1.002018/0.987537 at B32/64/128. Both round9
classes select norm mode7 at B>=32 and split mode6 only for32<=B<=64, otherwise
mode4 and mode1 respectively. Restoring this selection is not round9 gain.

For B>=16 and16-aligned, only the up-ND right operand changes:

| Property | Exact round7 control | Round9 candidate |
| --- | --- | --- |
| Prepared up-ND slot | `[N,K]`, contiguous, format2 | `[K,N]`, contiguous, format2 |
| Matmul right operand | `.t()`, shape`[K,N]`, stride`[1,K]` | direct, shape`[K,N]`, stride`[N,1]` |
| Preparation | transpose + contiguous + format cast | clone + format cast |
| Refresh | copy transposed source into fixed target | copy logical source into fixed target |
| Source alias | independent prepared copy | independent prepared copy, checked at setup and refresh |

Both inputs, the right operand and output remain BF16. The down projection's
physical stride, fixed ND/NZ batch selector, NZ up path, head projection,
weight values, all V math and public output/cache contracts are unchanged.
There are still three public matrix stages. All matrix/layout tasks that the
runtime actually launches count in the full Graph and full msprof. Kernel
versions are observed; this trial does not assume MatMulV3 or a benefit from
contiguous storage. Earlier all-ND or both-stage routing trials did not isolate
this up orientation. The rejected grouped-down route is not used.

The up-ND slot replaces the old slot; no fifth resident prepared matrix is
introduced. That does not establish equal allocated storage or free setup.
Actual shapes, strides, pointers, formats, storage bytes and dtype are recorded.
`prepare` and every changed/restored-weight refresh report synchronized host
latency, allocated bytes before/after, peak bytes above the entry allocation,
and transient peak above the exit resident allocation. These NPU allocator
statistics include clone/conversion temporaries and exclude reserved cache.
Allocator diagnostics run outside the measured setup/refresh interval, after
completion. All preparation and refresh remain outside inference Graph.
Call `refresh_weights` after same-address source updates; it copies into fixed
ND/NZ targets before replay. Reconstruction/recapture is required after source
shape/address changes. Direct gamma and final-head source updates retain their
existing contract. The source weight remains immutable during forward.

The legacy `matrix_route` string `nd_transposed_linear` remains the original
batch selector label for trace compatibility. It does not describe every
operand's current orientation. Inspect `wuq_row_nd` and `nd_layouts[1]` for the
candidate's actual `[K,N]` shape and row-major stride.

No native rebuild is required. Own V/host sources are byte-identical to round8;
mode8 remains compiled but the default wrapper never selects it. The retained
mode1/6 split branches and mode4/7 norm branches are unchanged. Parent verifies
the loaded round8 V binary SHA256
`a49ab3ef782d648ff44a204f0c2469df38977883df2f72753f17ebe2abf05dad`
and Torch extension SHA256
`c1328531de39083bac8094010c6e47cda8f766d32b8c6856d54eb9bb27203ef4`
before and after execution. Round8 synchronization scanner findings and manual
buffer-dependency evidence remain retained; unchanged sources do not turn the
previous scanner exit1 into PASS. Only public matrix weight layout is new.

Reproduce after local DCO commit, identical-tree GitHub publication and Docker
Git fetch/pull. Use the published checkout's exact HEAD for`PUBLISHED_SHA`:

```bash
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison up-nd-row-major --plan --mode paired
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison up-nd-row-major --mode verify --case-id decode-b32 --source-commit PUBLISHED_SHA --out NEW_SMOKE.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison up-nd-row-major --mode verify --source-commit PUBLISHED_SHA --out NEW_ALL15.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison up-nd-row-major --mode paired --case-id CASE_ID --source-commit PUBLISHED_SHA --out NEW_PAIRED.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison up-nd-row-major --mode profile --implementation candidate --case-id CASE_ID --source-commit PUBLISHED_SHA --out NEW_PROFILE.json
python3 benchmarks/kernels/mla_prolog/validate.py --baseline-route aclnn --no-timing --source-commit PUBLISHED_SHA --out NEW_FUSED_VERIFY.json
python3 benchmarks/kernels/mla_prolog/validate.py --baseline-route aclnn --case-id CASE_ID --source-commit PUBLISHED_SHA --out NEW_FUSED_PAIRED.json
```

All15 original cases, frozen precision thresholds, independent caches, untouched
cache regions, contiguous BF16 outputs and changed/restored same-address replay
checks remain mandatory. Paired timing uses20 AB/BA groups of10 complete forwards
per Graph. Collect every execution task with actual dtype/format and matrix
type for both sides at applicable ND points. KEEP/REJECT uses complete Graph
results for concrete B conditions. This is candidate9, not a new budget; no V3
shape-generalization result is presented as validated by this serial wrapper.
Local checks are CPU/source evidence; A3 correctness, allocation statistics and
performance remain pending parent execution at local handoff.
