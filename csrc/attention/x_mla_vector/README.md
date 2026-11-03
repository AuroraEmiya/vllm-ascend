# Independent MLA Prolog V stages

This operator implements the V stages for BF16 decode MLA Prolog. Its companion
Python wrapper currently uses public `torch.matmul` for down/up projections,
selected by the fixed candidate10 ND/NZ batch rule below. The additional round1
matmul spelling trial did not establish a speedup. The
head projection uses public `torch.bmm` for B=1 and
`torch_npu.npu_transpose_batchmatmul` for B>1.
The implementation was authored from the public mathematical and API contract;
the fused kernel, tiling, golden and computation tests were not read.

Mode 0 normalizes query down-projection and KV down-projection, multiplies gamma
directly, writes KV at valid distinct cache indices, and rotates/writes KR.
Mode 1 produces head-major non-RoPE queries and rotated query-RoPE output.
Mode 2 is retained for earlier candidates that restore batch-major query layout
after bmm. Candidate10 does not call it: the public transposed-bmm interface
returns the final layout for B>1, and B=1 uses a contiguous view. All layouts
executed by the selected matrix interface count in complete Graph time.
RoPE first places even positions before odd positions, then uses rotate-half;
this convention was confirmed by a structured black-box probe.
Mode 3 performs the same normalization and cache work as mode0 while reading
the query and KV/KR projections from one contiguous `[B,2112]` input.
Mode 4 has mode3's same shape, units, UB allocation, reduction, Rsqrt, gamma and
cache semantics. It broadcasts the FP32 inverse in UB with `Brcb` and strided
`Mul`, replacing only mode3's `V_S` event, scalar read and `Muls`. Mode3 keeps
the original scalar-read algorithm as a same-binary control. The additional
round4 candidate calls mode4; this requires rebuilding the own custom package,
without changing the Torch adapter ABI. The round4 A3 run passed all15 checks,
and all eight acceptance Graph points improved modestly. Mode4 is retained.

Mode5 keeps mode4's broadcast/math and caches gamma in existing `f1[64:]`
within one core's work-unit loop. Query gamma ends at element1599 of2048;
RoPE only overwrites `f1[:64]`. A width tag reloads gamma when query/KV type
changes, and Init clears the tag for every kernel invocation so same-address
gamma updates are observed on the next replay. The additional round5 wrapper
uses mode5 only for B>=32 and retains mode4 below32; its pinned round4 control
always uses mode4. No extra UB allocation or GM tensor is introduced.

Round5 passed all15 checks and is retained for B>=32. Round6 keeps that exact
wrapper as its control and changes only the second V call to mode6 for B>=32.
Mode6 produces the same split/RoPE outputs as mode1. It keeps Q in `b0` while
loading RoPE into the existing `b3`, removing the shared-b0 reuse wait within
one unit. Q output DMA can overlap the RoPE input DMA and FP32 UB computation.
The RoPE input is consumed before the later vector barrier and BF16 output
cast reuse `b3`. Both next-unit reuse events remain; mode6 additionally drains
the last MTE3 output through `MTE3_S`. Mode1 keeps its original algorithm.
The norm modes5/4, all public BF16 matrix calls, layouts, math order, UB size,
prepared weights and cache contracts remain those of round5. Round6 passed
all15 checks. Its Graph control/candidate ratios were1.006923 at B32,
1.018394 at B64 and0.989799 at B128. Mode6 is retained only for32<=B<=64;
mode1 is restored outside that interval in both round7 classes. This rollback
is not a new round7 performance gain.

Round7 selects norm mode7 only for B>=32, with retained round6 as its control.
Mode7 keeps mode5's gamma cache, FP32 arithmetic, buffers and BF16 results.
It permits input DMA into `b0`/`b1` while an earlier output DMA reads `b3`,
then waits with `MTE3_V` immediately before each output Cast overwrites `b3`.
This applies to both norm outputs and the KV-to-KR transition. Existing
`V_MTE2` protects the next input load; the final `MTE3_S` drains output before
return. The mode5 control keeps its original input-blocking waits. Mode6
remains a split mode, and the host explicitly treats mode7 as merged norm.
See `benchmarks/kernels/mla_prolog/NORM_OUTPUT_OVERLAP_TRIAL.md` for the
control derivation and buffer dependencies. Round7 passed all15 A3 checks;
its complete-Graph control/candidate ratios at B32/64/128 were
1.011370/1.029969/1.020738. Mode7 is retained for B>=32.

Round8 keeps that exact wrapper as its control and tests split mode8 for
B>=32. After GatherMask consumes the original input, four sin/cos coefficient
pairs occupy disjoint existing `f0[0:256]` windows. Independent cos products,
sine products and final Sub/Add share their respective barriers, retaining
all true dependencies and the original per-element arithmetic. This reduces
dynamic PIPE_V barriers per split tile from12 to6 without extra UB. Mode8
keeps the retained32..64 DMA overlap route and the mode1 route above64; norm7
is unchanged. See `benchmarks/kernels/mla_prolog/SPLIT_COEFFICIENT_PACK_TRIAL.md`
for addressing, synchronization and the pinned control. Round8 awaits A3.

The supported subdomain is `[B,1,He]`, B1..128, public A3 hidden sizes,
Hcq1536/Hckv512/D128/Dr64, head counts1,2,4,8,16,32,64,128; BF16 nonquantized
PA_BSND caches with block16 or128. The caller must validate distinct in-range
index metadata before copying/replaying it. A CPU helper checks this precondition;
the hot path performs no host reads or synchronization. The device also guards
slot bounds. Quantization, PA_NZ, empty batches and prefill are outside this scope.

The kernel uses complete per-row FP32 RMS reductions, `Rsqrt` for the reciprocal, and BF16
RINT writes. One V unit owns an entire 512-element KV or64-element KR row, each
at least128 bytes, so distinct valid slots have no shared output cache line.
All GM accesses use DMA; the int64 slot is read through a32-byte UB buffer.
Mode 0/1/3/4/5/6/7/8 UB allocation is59,424 bytes: one4096-element BF16 buffer, two
1536-element BF16 buffers, one2048-element BF16 buffer, five2048-element FP32
buffers, and32 index bytes.
Mode 2 allocates only one heads*512-element BF16 DMA buffer. V barriers order dependent vector operations; balanced hard events
order MTE2, V, scalar and MTE3, including buffer reuse. No cross-core dependency
or barrier is required.

Build inside the repository's CANN-enabled A3 Docker:

```bash
cd csrc
bash build.sh --pkg --ops=x_mla_vector --soc=ascend910_93 --vendor_name=mla_serial --ccache false -j4
```

The build suffix produces vendor directory `mla_serial_transformer`.
Install the resulting package under `vllm_ascend/_cann_ops_custom` and build the
repository's standard `vllm_ascend_C` target. The native adapter uses the existing
`EXEC_NPU_CMD` queue and registers PrivateUse1 and a preallocated-output Meta
implementation. Benchmark/Graph validation is in
`benchmarks/kernels/mla_prolog/validate.py`; NPU ST and CPU guard UT accompany it.

The wrapper explicitly prepares NZ and transposed ND matrix-weight copies at construction.
After changing source matrix data, call `refresh_weights()` on an ordered stream
before replay. Refresh copies into the existing buffers, checks actual formats29 and2,
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

Candidate4 concatenates WDQ and WDKV_KR by output columns during explicit
preparation and refresh, then uses a single public `torch.mm` to produce
`[B,1536+512+64]`. Mode3 reads the correct row stride and column offsets directly;
no sliced tensor is presented as contiguous and no layout copy is omitted from
Graph. The up projection and final head bmm are unchanged. Preparation includes
the concatenation and NZ conversion; persistent and peak extra allocation are
recorded. All BF16 outputs and cache state after same-address weight refresh
must pass the original independent FP64 thresholds, including any matmul
rounding differences caused by the wider output matrix.

Candidate5 (rejected after complete-Graph timing) gives query normalization, KV normalization and KR RoPE independent
work units. Every unit owns its UB and writes a separate output tensor; valid
distinct indices preserve at least128-byte separation between KR rows and
1024-byte separation between KV rows. No communication or cross-core barrier
is required. This removes the sequential normalization/RoPE dependency within
one KV work unit while preserving the exact same arithmetic and buffer sizes.

Candidate6 restores candidate4 two-owner normalization/cache and uses32-head RoPE/DMA tiles when B>=32 and16-head tiles for smaller
batches. The32-head tile amortizes vector coefficient broadcasts, events and
strided stores across more heads. Working FP32 and BF16 buffers are enlarged
to hold2048 RoPE values; total UB remains59,424 bytes and is checked against
the runtime platform. B31/B32/B33 protect the explicit workload threshold.

Candidate7 uses the public `torch_npu.npu_transpose_batchmatmul` for B>1,
with input `[heads,B,128]`, weight `[heads,128,512]`, identity input permutations,
and output permutation `[1,0,2]`. Its output is directly `[B,heads,512]`, removing
the separate mode2 layout kernel. B=1 keeps public bmm plus a contiguous view.
All actual tasks launched inside this public matrix call remain part of Graph
and full profiling, including any implementation layout work. No head weight
preparation or hidden intermediate rounding is changed. Public schema/API:
[op-plugin declaration](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/op_plugin/config/op_plugin_functions.yaml),
[ACLNN contract](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnTransposeBatchMatMul.md).

Candidate8 (rejected for severe small-batch regressions) prepares the two merged-down/up weights as contiguous transposed ND
format2 and uses public `torch.nn.functional.linear` for these two matrix stages.
Preparation and refresh include transpose, contiguous storage and format cast;
the stable-address refresh contract is unchanged. This is an actual matrix-route
candidate: kernel versions and all internal tasks are determined from complete
profiling, not inferred from the API name or layout. Candidate7 head projection
and every own V stage are unchanged.

Candidate9 restores the retained candidate7 NZ matrix route and replaces only the FP32 `Sqrt` plus `Div` RMS reciprocal with public
Ascend C `Rsqrt`. Reduction, gamma multiplication and BF16 RINT outputs are
unchanged. No FP32 inverse-RMS auxiliary value is part of this public operator's
observable outputs. Acceptance still uses the frozen independent FP64 reference
and all same-address Graph/cache checks; approximate-instruction precision is
not assumed to pass and the thresholds are not relaxed.

Candidate10 selects public transposed-ND `F.linear` only when B>=16 and B is
16-aligned; otherwise it retains public NZ `torch.mm`. This is a newly tested
candidate, based on candidate8's shape-dependent regressions. It does not select
historical best result rows. Both weight layouts are prepared explicitly, with
stable-address refresh and physical-format checks for all four buffers. Their
extra resident and peak allocation, preparation and update latency are reported.
The selected route is captured in each result. All15 original cases, including
B31/B32/B33, are rerun on this single source tree. The head-matrix route and own
V arithmetic remain as candidate9.
