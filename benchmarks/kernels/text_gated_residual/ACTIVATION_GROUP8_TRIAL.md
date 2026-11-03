# Eight-row packed activation trial

Additional round9 keeps round8 and changes only packed activation at tokens>=256.
At B256 the previous group4 has64 units across40 cores, with two units on some
cores. Group8 uses32 units with one unit per active core. The hypothesis trades
core count for fewer DMA and vector-fence sequences. Round8 activation7.939us
is the current diagnostic basis; acceptance is full Graph, not this task alone.

Mode9 pins previous group4 in the same rebuilt binary; default mode5 uses8.
The two BF16 and two FP32 scratch tensors grow from2560 to2688 elements on
the new path: mode5 UB41024->42560 bytes, below the existing51232-byte floor.
No GM tensor or matrix interface changes. Batched vector math and strided
stores preserve all live320 activation values and16 padded gate values.

Run original15 cases via `matrix_trial.py --activation-group-control` and
separate B255/256/257 guards via the same flag plus `--activation-boundaries`.
All guards retain FP64 oracle, eager/Graph and same-address changes/restoration.
Full timing remains20 AB/BA pairs,10 complete forwards/Graph, with fresh fused
comparison and full all-task profile. Keep only if affected B256 complete Graph
improves and all correctness guards pass. B<256 uses its prior group sizes.
