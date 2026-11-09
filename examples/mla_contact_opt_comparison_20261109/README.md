# Frozen Contact opt comparison harness

This package contains only the new test harness. It does not republish Contact
source. Supply explicit frozen own-source and Contact-package paths and record
their manifests. All operands at matrix calls remain BF16. Input flattening for
Contact rank2 is setup metadata; output shape normalization is validation only.
Every timed replay contains one full original call and is measured by NPU Events.
Twenty randomized paired samples each contain ten replays after ten warmups.

Primary labels: installed aclnnMlaProlog, installed MlaPrologV3, frozen own tuned
V3-NZ (S0 and optionally S2), original Contact and each V1 opt default entry.
The two compose_norm_*_opt4 labels are diagnostic compositions of existing Contact
ops and remain separate from those unchanged default entries. They do not consume
an optimization round. The documented original V0 T>=128 hang is retained as a
coverage exclusion for unchanged full entries, not silently replaced.

CPU FP64 independent oracle, whole-output fused comparison, untouched cache exact,
and return/alias checks are outside timing. Profiling is also outside Event timing.
No new production kernel or matrix implementation is introduced.
