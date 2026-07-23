# SparseKvGather SFA-style V1

## Implemented changes

1. **16-row ping-pong staging**
   - Each UB stage holds 16 rows of CTKV and KPE.
   - A full-valid chunk performs all GM->UB loads first, then one synchronization, then two contiguous UB->GM writes.
   - Output layout and slot order remain unchanged.

2. **Typed index paths**
   - All-INT32 and all-INT64 inputs enter separate templated device paths.
   - The index dtype decision is made once per kernel instead of in every `ReadIndex` call.
   - Mixed index dtypes retain the original generic fallback path.

3. **BF16 + INT32 support**
   - The OpDef support matrix now contains:
     - FP16 + INT32
     - BF16 + INT32
     - BF16 + INT64
   - `block_table`, `topk_indices`, and `cur_pos` are required to use the same dtype.

4. **Full-valid fast path**
   - Up to 16 contiguous output slots in one query are resolved first.
   - If all slots are valid, the kernel uses the batched staging path.
   - If any slot is invalid, the chunk falls back to the original pair/single/zero semantics.

## Preserved semantics

- `block_size` remains fixed at 128.
- Output shapes and dtypes are unchanged.
- Top-k order is preserved.
- Invalid slots remain in place and are zero-filled.
- Descending physical-token pairs are not reordered.

## Required validation

Rebuild the custom operator before testing.

Run existing correctness tests:

```bash
pytest ../../test.py -v
```

Add or run a BF16 + INT32 case explicitly. The three index tensors must all be `torch.int32`.

Profile the real shape:

```text
paged_ctkv   [9840, 128, 1, 512] BF16
paged_kpe    [9840, 128, 1, 64]  BF16
block_table  [16, 547]           INT32 or INT64
topk_indices [16, 2048]          INT32 or INT64
cur_pos      [16]                INT32 or INT64
```

Recommended profiling command:

```bash
rm -rf ./prof_sparse
msprof \
  --output=./prof_sparse \
  --runtime-api=on \
  --task-time=on \
  --ai-core=on \
  pytest ../../test.py::TestSparseKvGather::test_perf_large_case -s -v
```

Compare at least:

- Wall Duration
- `aiv_scalar_ratio`
- `aiv_mte2_ratio`
- `aiv_mte3_ratio`
- stable-run variance after warmup

## Expected profiler behavior

The first expected improvement is fewer MTE2/MTE3 synchronization and output-copy transactions. Total output bytes are unchanged, so MTE3 ratio may remain high even if wall duration falls. The typed INT32 path should primarily reduce scalar overhead.

## Validation status

This environment does not contain the AscendC/CANN build toolchain or NPU runtime. The source was structurally checked, but it has not been compiled or executed here.
