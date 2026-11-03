# Final retained MLA Prolog: exact round7

All ten additional optimization candidates have been consumed. The selected
wrapper is byte-identical to retained round7, published as
`7ada452c33c5748fe141a786250c2b30345b637f`, with canonical SHA256
`1030da1293222b46d337840a496acc87d0bae152c532007121368da507096d56`.
It is also identical to the committed `baseline_serial_7ada452c.py` snapshot.
Norm mode7 is selected only for B>=32; split mode6 only for32<=B<=64. The public
matrix route remains transposed ND for B>=16 and16-aligned, otherwise NZ, with
head bmm at B1 and public TransposeBatchMatMul above B1. All matrix boundaries
remain BF16. Preparation/refresh and peak allocation diagnostics are retained.

Round8 coefficient packing and round9 row-major up-ND were rejected. Round10's
norm7 extension to16<=B<32 passed all15 A3 accuracy/state checks and35 remote CPU
checks, but its performance result was rejected as inconclusive. B16 pooled
control/candidate ratios were1.012446 initially and1.004904 on confirmation.
Unchanged anchors showed larger differences; B128 reversed0.926790 to1.058892
despite identical selected algorithms and equal non-address metadata. Initial
B16 order-group pair ratios0.954855/1.038808 exceeded the small pooled effect.
All original samples and pooled metrics remain retained. The system cause of
unchanged-path differences is unassigned; it is not a threshold algorithm
regression. See `NORM_OVERLAP_B16_TRIAL.md` and the independent Event analysis
under the project's optimization2 outputs. The1.5x fused/serial target remains
unmet; no rollback gain is credited.

`matmul_trial.py --comparison final-retained-r7` is explicitly validation-only:
`new_budget_round=null`, `optimization_candidates_consumed=10`,
`candidate_count_increment=0`, and `selected_candidate_round=7`. Both harness
labels `control` and `candidate` execute this same retained algorithm with
independent prepared tensors/caches/Graphs. Their paired timing is a measurement
negative control, not an optimization result. The CLI checks the current wrapper
SHA before device imports; historical r9 and r10 CLIs retain their own source
guards and cannot silently execute this rollback as their candidate.

Fresh final-source validation must record this final checkout's published SHA
and tree. It reuses the same fixed native binaries with no source or build change:

- V SHA256 `a49ab3ef782d648ff44a204f0c2469df38977883df2f72753f17ebe2abf05dad`.
- Torch extension SHA256 `c1328531de39083bac8094010c6e47cda8f766d32b8c6856d54eb9bb27203ef4`.

Parent owns GitHub publication, fresh resource/ownership gates, Docker Git sync,
binary before/final comparison and every A3 execution. Own V/host sources are
unchanged from round8; mode8 is compiled but never selected. Original scanner
exit1 findings/manual explanations remain intact, not converted into PASS.

Use the exact published final HEAD for`FINAL_PUBLISHED_SHA` and new output paths:

```bash
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison final-retained-r7 --plan --mode paired
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison final-retained-r7 --mode verify --case-id decode-b16 --source-commit FINAL_PUBLISHED_SHA --out NEW_FINAL_SMOKE.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison final-retained-r7 --mode verify --source-commit FINAL_PUBLISHED_SHA --out NEW_FINAL_ALL15.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison final-retained-r7 --mode paired --case-id CASE_ID --source-commit FINAL_PUBLISHED_SHA --out NEW_FINAL_NEGATIVE_CONTROL.json
python3 benchmarks/kernels/mla_prolog/matmul_trial.py --comparison final-retained-r7 --mode profile --implementation candidate --case-id CASE_ID --source-commit FINAL_PUBLISHED_SHA --out NEW_FINAL_PROFILE.json
python3 benchmarks/kernels/mla_prolog/validate.py --baseline-route aclnn --no-timing --source-commit FINAL_PUBLISHED_SHA --out NEW_FINAL_FUSED_VERIFY15.json
python3 benchmarks/kernels/mla_prolog/validate.py --baseline-route aclnn --case-id CASE_ID --source-commit FINAL_PUBLISHED_SHA --out NEW_FINAL_FUSED_PAIRED.json
```

The fixed15 cases, independent FP64 oracle and precision thresholds remain
unchanged. Source/index changes, restored values, same prepared addresses,
contiguous BF16 outputs, exact untouched cache regions and Graph replay must
all pass. The actual fusion comparison is `validate.py --baseline-route aclnn`
against the installed public V1 MlaProlog black box. Eight acceptance cases are
B1/2/4/8/16/32/64/128 at He7168 and heads32; seven guards are retained. V3 shape
generalization is independent and is not claimed by this serial implementation.

Full Graph Event is the primary latency; full msprof retains all execution
tasks with actual kernel types, BF16 metadata and matrix/V stage diagnostics.
Preparation and weight refresh costs are separate explicit static-weight setup
costs, including transient allocator peaks. Final fresh artifacts belong to the
parent's final retained validation output, not an iteration11 directory. Source
selection does not turn the unresolved1.5x target into PASS.
