# Logical ND packed down weight trial

Additional round10, the last candidate in the new budget, retains round9.
Packed down C is20.761/21.422us at B32/64, compared with9.765us at B16.
The single hypothesis changes the physical right-hand ND matrix from a
transpose view of [336,10240] to contiguous logical [10240,336], preserving
every BF16 value and public matmul semantics. It may change transpose/tiling
dispatch; only actual full traces determine the resulting MatMul version.

Candidate uses this prepared buffer only for32<=tokens<=64 with gate. Control
sets `_use_logical_down=False`; all other stages, layouts and data are equal.
Preparation adds6,881,280 persistent bytes per gated object and a transient
same-size logical buffer during refresh, outside Graph. Source updates are
repacked at the captured address before replay. No activation cast, copy or
preparation occurs outside the counted forward.

Run original15 cases with `matrix_trial.py --down-layout-control`, unchanged
CPU FP64 oracle and eager/Graph/source-update checks. Record prepared actual
format2, BF16, strides, storage and stable pointers. Full Graph uses20 AB/BA
pairs with10 forwards; fresh fused comparison and full-task profile follow.
Reject the route if affected Graph latency regresses or any correctness,
update, dtype or hidden-activation-conversion requirement fails.
