# TextGatedResidual serial V stages

This operator is part of the serial TextGatedResidual pipeline in
`vllm_ascend/ops/text_gated_residual.py`. Matrix work uses public `torch.mm`.
The independently written Ascend C operator implements three vector stages:
group RMSNorm with optional raw gamma, SiLU of down projection divided by four,
and gated branch mixing with optional write gate.

The implementation targets A3 (`ascend910_93`), BF16, four branches of width
2560 and projection rank 320. The input is contiguous `[B,S,10240]` with
`1 <= B*S <= 2048`. Gamma is optional contiguous FP32 `[10240]` and scales
the normalized input by `1 + gamma`. Weight layouts are `[320,10240]`,
`[10240,320]` and optional `[4,10240]`.

The prepared Python class returns five tensors, matching the public low-level
CANN schema. Missing outputs are empty tensors. The convenience function uses
the public package's optional return convention. `need_backward=True` requests
forward auxiliary tensors; autograd backward is outside this implementation.
No inputs or caller-owned state are modified.

With a gate, the prepared class packs the down and gate weights into BF16
`[336,10240]`: 320 down rows, 4 gate rows, and 12 zero padding rows. A single
public `torch.mm` computes these projections, then owned V computes SiLU and
the write gate. Preparation allocates 6,881,280 extra bytes and is measured by
the harness. The up projection remains a separate public `torch.mm`.

After changing down/gate weight contents, the caller must invoke
`prepared.refresh_weights_()` before replay, on the same stream or with an
explicit stream dependency. The refresh copies only weights into the existing
prepared address; input-dependent computation remains inside every Graph.
New source weight addresses/metadata require a new object and recapture.
Same-address update tests invoke and time this explicit refresh outside the
forward latency, as allowed for the weight preparation stage. Down auxiliary
output is a strided view of the live projection columns.

The mix stage loads the four branch segments with a strided DMA, evaluates
their sigmoids together, then accumulates branch products in FP32 branch order.
Each core owns disjoint token/hidden tiles. Its UB demand is `68*tile + 32`
bytes, at most 174,112 bytes; tiling queries the available UB and reduces the
tile if needed. Other stages use 51,232 bytes. DMA/vector dependencies use
paired hardware events, with vector barriers between dependent operations.
Inference without auxiliary outputs uses the same precise normalization but
omits the unused inverse-RMS allocation and scalar global-memory stores.
Requests for auxiliary outputs retain the FP32 inverse-RMS output and checks.
The normalization scale stays on the vector pipeline through `Brcb` and a
zero-stride multiply. The internal write-gate buffer uses 16 BF16 values per
row to separate each core's small store into its own 32-byte block. The API
returns the live `[B,S,4]` view; its row stride is 16 and unused padding is not
exposed. This adds 24 temporary bytes per token, with no extra runtime task.

The dimensions come from the public Qwen3.8-Flash-Next configuration. The token
counts in the benchmark are synthetic and do not represent a captured model
trace. Performance acceptance is decode only: `S=1`, with batch sizes
`B=1,2,4,8,16,32,64,128,256`. No prefill performance claim is made.
The fused package is used only as an opaque runtime baseline. Its
implementation, tiling, golden tests and disassembly are not consulted.

Public semantic references:

- [Qwen3.8 paper, section 2.2](https://arxiv.org/html/2608.30320v1#S2.SS2)
- [Model configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)

Build this native operator through the repository's existing CANN build entry:

```bash
cd csrc
bash build.sh --pkg --ops=x_tgr_vector --soc=ascend910_93 --vendor_name=tgr_serial -j4
bash build_out/cann-ops-transformer-tgr_serial_linux-aarch64.run \
  --quiet --install-path="$PWD/../vllm_ascend/_cann_ops_custom"
```

The repository appends `_transformer` to the requested vendor name. Include
`vllm_ascend/_cann_ops_custom/vendors/tgr_serial_transformer` in
`ASCEND_CUSTOM_OPP_PATH` before importing the native extension. When comparing
with the opaque package, include its separate vendor directory as well.

The standard A3 operator list also includes `x_tgr_vector`. Its adapter and Meta
implementation register `_C_ascend.npu_tgr_vector_out` through the existing
`csrc/torch_binding.cpp` and `csrc/torch_binding_meta.cpp` translation units.
There is no separate operator wheel or alternate stream management framework.

The benchmark in `benchmarks/kernels/text_gated_residual/validate.py` preserves
FP64 oracle checks, Graph updates at stable addresses, and paired complete
Graph event samples. Performance and full-Graph profiler evidence are separate
outputs. No performance target is claimed before runtime validation.
