# TextGatedResidual serial C+V (experimental)

This independently written candidate targets Ascend A3 / Ascend910_93 and the
Qwen3.8-Flash-Next dimensions: 4 branches, hidden size 2560, bottleneck 320.
It is under development; correctness and performance acceptance are pending.
No 1.5x speedup is claimed.

The implementation uses public `torch.mm` for all matrix products and the
registered Ascend C operator `XTgrVector` for RMSNorm, SiLU and gated mixing.
The `native/` skeleton was generated with CANN's `msopgen`; the host tiling and
kernel computations are independent. `torch_ext/` provides an ACLNN adapter,
PrivateUse1 dispatch and Meta registration through `NpuExtension`.

Inputs are contiguous BF16 `x[B,S,10240]`, `w_down[320,10240]`,
`w_up[10240,320]`, optional `w_gate[4,10240]`, and optional FP32 raw
`gamma[10240]`. B and S are positive and B*S is at most 2048. Channels is 4;
eps is finite and positive, default 1e-6. There is no autograd implementation.
`need_backward=True` exposes forward intermediates only.

The independent mathematical reference is the
[Qwen3.8-Next paper, section 2.2 equations 30-33](https://arxiv.org/html/2608.30320v1).
The raw gamma convention, public output shapes and optional outputs follow
the supplied package's public ACLNN header and Python schema/Meta.
No fused native implementation or compute reference was read or reused.
The shape list in `cases.json` is agent-designed from the official
[model configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json),
and is not a model execution trace.

`TextGatedResidualSerial` owns references to caller-provided weights. Construct
it once and keep the weights unchanged through graph capture/replay. Rebuild
the object and recapture when weight addresses or prepared layouts change.
All per-forward work belongs inside the measured graph; weight preparation
is recorded separately. The complete graph's `fused_us / serial_us` is the
performance metric. Per-kernel profiler durations are a separate measurement.

The upstream fused wheel and `.run` baseline are supplied separately by the
user and are intentionally absent from this source tree.
