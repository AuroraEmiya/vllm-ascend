# DCP sharded indexer candidates and lengths

## Description

- **Function**: Pack local QLI candidates, select exact global Top-K or owner-local winners, and build local causal key lengths.
- **Formula**: For local ID x, global ID is `(x // I * P + r) * I + x % I`. Owner-local output keeps global winners satisfying `(id // I) % P == r`. Local prefix length is `L // (P*I)*I + clamp(L % (P*I) - r*I, 0, I)`.
- **Algorithm flow**: Mask invalid candidates; optionally sort by score descending and ID ascending; map IDs and pack score bits; AllGather one tile; merge globally or search the owner's exact winning prefix; write every output slot including -1 padding. Local lengths derive from global causal prefixes on device.
- **Supported modes**: NPU eager and NPUGraph. A3/A5 share the implementation. Native FP32 and BF16 comparator inputs retain their original precision.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| indices, scores | Input | Native local QLI candidates `[T,1,K_local]`, positive strides | INT32; FP32/BF16 | ND |
| world_size, rank, interleave | Attribute | DCP group geometry and token stripe width | int | Scalar |
| sort_candidates, fuse_sort | Attribute | Establish comparator order; permit workload-based fused preparation | bool | Scalar |
| packed | Input/Output | Exact global ID and FP32 score-bit pairs `[P*T,K_local,2]` after gather | INT32 | Contiguous ND |
| rows, topk | Attribute | Query count and global winner count | int | Scalar |
| local_rank | Optional attribute | Owner-local output; None selects global coordinates | int/None | Scalar |
| presorted, owner_sharded | Attribute | Explicit sorted-order and per-source owner contracts | bool | Scalar |
| out | Optional output | Destination `[T,1,topk]`; may be a contiguous row slice | INT32 | ND |
| seq_lens, query_ends | Input | Global sequence lengths and cumulative query endpoints | INT32 | ND |
| key_lengths, visible_lengths | Output | Per-request local length and per-query local causal prefix | INT32 | ND |
| query_start_loc | Input | Cumulative query bounds including leading zero for native candidate normalization | INT32 | ND |

## Constraints

- DCP/stripe sizes are positive; all ranks use identical collective tile boundaries. The owning backend requires complete replicated indexer heads and `max_model_len <= 2**24` for exact FP32 ID comparison.
- K2048 has fused preparation/merge. Other supported K values retain native stable sorting. BF16 digit sorting uses its 16 meaningful comparator bits; FP32 scores are never rounded into that path.
- Owner-prefix merge requires `presorted=True`, `owner_sharded=True`, `K_local=topk=2048`, and a valid local rank. Other calls use full selection.
- Non-power-of-two rank tiles clamp padded peer addresses before masked loads; this prevents A3 GM faults without changing any valid peer or selection result.
- Invalid/nonfinite candidates do not enter the valid output prefix. Empty requests and graph padding produce zero lengths or -1 IDs. Normalization writes only the native QLI-owned output buffers in place.
- ROWS is runtime and does not specialize transport/merge kernels. K, topology, strides and sorting tiles are fixed compilation/layout parameters. Length kernels keep row/request counts dynamic; power-of-two tiles and binary-search step counts select compilation buckets. Input values update on replay. Black-box compilation counts and hardware/model validation belong in the PR evidence.

## Origin and Differences

- **Origin**: DCP candidate exchange developed for token-sharded QLI.
- **Differences**: Exchanges exact candidates rather than token-history KV. Owner-local output removes a later remap; occupancy-based preparation and grouped merge bound work without chip-model dispatch.

## Test Cases

Actual K2048 deployment shapes include T48/DCP16 and T128/DCP8. Shared tests also cover generic K, partial tiles, non-power-of-two DCP, ties, near-ULP scores, invalid owners, strided input, exact score bits, large mapped IDs and changing-input graph replay.

```bash
pytest -q tests/ut/attention/test_sharded_indexer.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dcp_candidates.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dcp_pre_post.py -k local_lengths
```

## Owner-prefix implementation

`merge_packed_candidates` selects the owner-prefix kernel when `topk == 2048`,
`local_k == topk`, `presorted=True`, `owner_sharded=True`, and `local_rank` names
a valid rank. Production DCP callers provide rank-major gathered candidate pairs.
The public wrapper allocates the output unless `out` is supplied; that buffer is
written in place, including every `-1` padding slot. Inputs are not modified.

One `_merge_owner_prefix` implementation handles both peer-count strategies.
A3's hardware policy enables parallel peers for `world_size > 4` and more rows
than vector cores. Other policies and smaller workloads use serial peers. A
skewed row also uses serial scans within the same kernel. Each sorted peer stops
after its first partial tile. Bounds always use exact better-candidate counts.

For the first owner candidate (`pivot == 0`), finding at least K better entries
proves that the owner contributes no winners. Remaining peers may then be
skipped. For any other pivot the count is completed, because interpolation and
boundary updates require the exact value. This preserves score-descending,
ID-ascending ordering, including tied scores and empty owner prefixes.

### Specialization and warmup

`WORLD`, `OWNER`, `INTERLEAVE`, `LOG_BLOCK`, `TILE`, `BLOCK_RANKS` and
`PARALLEL_PEERS` are compile-time parameters: they determine fixed lane shapes,
rank geometry, ID conversion or peer strategy. `num_rows` is a runtime integer
with `do_not_specialize`; row count alone is not an explicit compile-time axis.
Tensor dtype, alignment classes, compiler options and cache state still matter.
The two interpolation updates and four spare probes are named search constants.

This PR does not introduce an explicit warmup registration for the owner kernel.
Uncached variants may compile during their first invocation, including a dummy
run or graph capture. Serving-phase compilation counts have not been measured.

### Owner-prefix validation

The added test uses independent CPU global Top-K and owner compaction with exact
integer comparison. It covers real logical row counts 144/152, core boundaries,
non-power-of-two worlds, ties, skew, short prefixes, invalid entries, and changed
inputs under graph replay. A3 is the measured optimization target; A5 and model
accuracy/performance validation remain outstanding for this revision.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_owner_prefix_adaptive.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dcp_candidates.py
pytest -q tests/ut/device/test_hardware_profile.py
```
