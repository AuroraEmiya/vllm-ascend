# Independent MLA Prolog V stages

This operator implements the V stages for BF16 decode MLA Prolog. Its companion
Python wrapper uses public `torch.mm` and `torch.bmm` for all matrix products.
The implementation was authored from the public mathematical and API contract;
the fused kernel, tiling, golden and computation tests were not read.

Mode 0 normalizes query down-projection and KV down-projection, multiplies gamma
directly, writes KV at valid distinct cache indices, and rotates/writes KR.
Mode 1 produces head-major non-RoPE queries and rotated query-RoPE output.
Mode 2 restores batch-major final query layout. For B=1, that final change is a
contiguous view, so mode 2 is omitted. All remaining layouts count in Graph time.
RoPE first places even positions before odd positions, then uses rotate-half;
this convention was confirmed by a structured black-box probe.

The supported subdomain is `[B,1,He]`, B1..128, public A3 hidden sizes,
Hcq1536/Hckv512/D128/Dr64, head counts1,2,4,8,16,32,64,128; BF16 nonquantized
PA_BSND caches with block16 or128. The caller must validate distinct in-range
index metadata before copying/replaying it. A CPU helper checks this precondition;
the hot path performs no host reads or synchronization. The device also guards
slot bounds. Quantization, PA_NZ, empty batches and prefill are outside this scope.

The kernel uses complete per-row FP32 RMS reductions, `Sqrt` plus `Div`, and BF16
RINT writes. One V unit owns an entire 512-element KV or64-element KR row, each
at least128 bytes, so distinct valid slots have no shared output cache line.
All GM accesses use DMA; the int64 slot is read through a32-byte UB buffer.
Mode 0/1 UB allocation is44,064 bytes: one2048-element BF16 buffer, three
1536-element BF16 buffers, five1536-element FP32 buffers, and32 index bytes.
Mode 2 allocates only one heads*512-element BF16 DMA buffer. V barriers order dependent vector operations; balanced hard events
order MTE2, V, scalar and MTE3, including buffer reuse. No cross-core dependency
or barrier is required.

Build inside the repository's CANN-enabled A3 Docker:

```bash
cd csrc
bash build.sh --pkg --ops=x_mla_vector --soc=ascend910_93 --vendor_name=mla_serial --ccache false -j4
```

Install the resulting package under `vllm_ascend/_cann_ops_custom` and build the
repository's standard `vllm_ascend_C` target. The native adapter uses the existing
`EXEC_NPU_CMD` queue and registers PrivateUse1 and a preallocated-output Meta
implementation. Benchmark/Graph validation is in
`benchmarks/kernels/mla_prolog/validate.py`; NPU ST and CPU guard UT accompany it.

The wrapper explicitly prepares three NZ matrix-weight copies at construction.
After changing source matrix data, call `refresh_weights()` on an ordered stream
before replay. Refresh copies into the existing buffers, checks actual format29,
and preserves captured data pointers. Gamma and the final head matrix remain
direct references. Reconstruct and recapture after weight addresses or
shape/layout metadata changes. Preparation, refresh latency and extra storage
are reported separately from the complete inference Graph. The wrapper returns the same
cache tensors it updates; only the documented input-cache mutation and values
are required for comparison, not an undocumented baseline return alias.

Candidate 2 batches sixteen heads per mode-1 tile. Strided DMA packs/scatters
non-RoPE query data; vector repeats broadcast the same per-row coefficients over
packed even and odd head halves. Two strided stores restore each head's RoPE
halves without a scalar per-head loop. Mode 2 transposes a whole row using a
strided load and contiguous write. Mode 0 is unchanged. Mode 0/1 UB uses44,064
bytes; mode 2 allocates only heads*512*2 bytes (at most131,072 bytes), checked
against the runtime platform UB size. Matrix calls and weight semantics stay
unchanged. All batching and layout work remains inside the complete Graph.

Candidate 3 prepares `WDQ`, `WUQ_QR` and `WDKV_KR` in actual FRACTAL_NZ format29
for the same three public `torch.mm` calls. The caller enables
`torch.npu.config.allow_internal_format=True` before preparation and inference.
No copy, format conversion or weight refresh is omitted from a per-call weight
update workload: this path assumes weights are prepared between updates.
The benchmark explicitly refreshes after same-address source updates, checks
NZ target addresses and physical formats again, and validates all outputs and
cache state against the independent FP64 reference after replay.
