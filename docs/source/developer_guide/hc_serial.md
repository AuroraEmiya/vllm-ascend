# Qwen3.8 Flash Next HC serial Ascend C path

The implementation provides three Ascend C Vector kernels and two independent
public PyTorch linear calls on A3. It implements the Qwen4Exp-named HC path used
by Qwen3.8 Flash Next. Down and Up stay separate, with a BF16 SiLU stage between
them. This development implementation does not claim the 1.5x performance goal.

## Functional boundary

Let K be the HC stream count, H the per-stream width, D=K*H, and R the low rank.
The complete inputs are:

| Input | Shape |
| --- | --- |
| residual | [M,D] |
| previous block output | [M,H] |
| previous injection | [M,K] or None |
| norm weight | H or D contiguous elements |
| down weight | [R+K+P,D] intermediate; [R,D] final |
| up weight | [D,R] |
| configuration | positive finite epsilon, K, final/intermediate flag |

P=(-(R+K))%16 for intermediate mixers. The previous injection is consumed even
by a final mixer. Inputs are read only. The result has three values:

- Updated residual [M,D], freshly allocated.
- Input to the next block [M,H], freshly allocated.
- Next injection [M,K], a view into down projection storage, or None for a final
  mixer. Its row stride includes the discarded alignment padding.

Combine is `BF16(residual + block * scale)`, where scale is
2*sigmoid(previous_injection/K), or one without injection. This BF16 boundary
occurs before grouped RMSNorm. For each H group, z=combined*rsqrt(mean(square)
+epsilon), and the Gemma affine is BF16(z+z*norm_weight).

Down is BF16(normalized @ down_weight.T). Its first R columns feed
BF16(silu(down[:,:R]/K)); the following K columns supply the next injection.
Up is BF16(lowrank @ up_weight.T). Gate mixing returns
BF16(sum(sigmoid(gate)*normalized, stream)/K).

## Domain and shape evidence

The original Triton source accepts 2D matrices with unit inner stride, positive
K, D divisible by K, and contiguous H- or D-element norm weights. It masks tails;
H and R do not have to be powers of two or multiples of 512. Static checks alone
do not establish every runtime resource limit.

This Ascend C path supports BF16 on A3, positive H/R up to 8192, checked uint32
element spans, nonnegative row strides, finite positive epsilon, and M=0.
Quantized weights, FP16/FP32, and TP-sharded projection adapters are outside the
tested domain.

The decode acceptance cases are M=1,16,32,64, each with intermediate and final
mixers. M is explicitly selected by the user. Zhanlu parameter-consumer metadata
supports K4,H2560,D10240,R320,P12, intermediate down [336,10240], final down
[320,10240], and up [10240,320]. Actual A3 tests establish runtime validity for
the requested M values.

The original formal decode template relation generated a negative dimension.
That failed derivation remains recorded and supplies no valid M evidence.
Prefill cases are exploratory and excluded from decode acceptance. Epsilon
1e-6 is an explicit test setting; the deployed model configuration remains
unverified.

## Implementation and integration

Source, registration, CMake, and Python files follow the main repository layout.
The regular registered extension discovers the HC host/kernel files when direct
kernels are enabled. The validated development build uses the same root project:

```bash
cmake -S . -B build-hc \
  -DVLLM_ASCEND_BUILD_HC_ONLY=ON \
  -DPYTHON_EXECUTABLE="$(command -v python3)" \
  -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
  -DTORCH_NPU_PATH="$(python3 -c 'import torch_npu,os;print(os.path.dirname(torch_npu.__file__))')" \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
  -DSOC_VERSION=ascend910_9362 -DCMAKE_BUILD_TYPE=Release
cmake --build build-hc --target vllm_ascend_hc_C -j4
```

Load vllm_ascend_C or vllm_ascend_hc_C before calling
vllm_ascend.ops.qwen_hc_serial.combine_and_mix. The explicit
combine_and_mix_module adapter accepts an unquantized replicated GatedResidual.
The default model path is still Triton; automatic model activation and a full
model-serving run are not part of this validation.

The final path uses two F.linear calls and has no weight packing, hidden cache,
forward compilation, or prepared-weight mode. Slower transposed and singleton
matmul experiments remain available through historical commits and local
iteration records.

CombineNorm batches two HC branches for M16 and four for M32/64 when the H/K/UB
conditions allow it; other inputs use the per-branch path. BF16 combine is
preserved before RMS. A3 memory-vector MulAddDst shortens combine/affine work.
Gate batches HC branches and uses a two-level FP32 reduction for K4. M64 SiLU
batches two aligned rows. Generic tail, stride, shared-weight, and larger-K
fallbacks remain available. No C calls or V stages are reordered across C.

## Device correctness and Graph timing

The development target has been built on Ascend910_9362 with CANN9.2.0,
Torch2.10.0, and torch_npu2.10.0.post7. Use fresh build directories after source
changes. Inspect the actual binary identity before measuring.

```bash
python3 tests/manual/test_qwen4_exp_hc_serial_npu.py \
  --extension build-hc/vllm_ascend_hc_C.cpython-311-aarch64-linux-gnu.so \
  --mode validate --out boundary-results.json
python3 tests/manual/test_qwen4_exp_hc_serial_npu.py \
  --extension build-hc/vllm_ascend_hc_C.cpython-311-aarch64-linux-gnu.so \
  --cases tests/manual/qwen_hc_serial_cases.json \
  --mode validate --out decode-correctness.json
python3 tests/manual/test_qwen4_exp_hc_serial_npu.py \
  --extension build-hc/vllm_ascend_hc_C.cpython-311-aarch64-linux-gnu.so \
  --cases tests/manual/qwen_hc_serial_cases.json \
  --mode bench --out decode-performance.json
```

The independent CPU float64 reference materializes every specified BF16
boundary. Validation checks all three outputs in eager and Graph replay,
fresh outputs, injection view metadata, and input mutation. A tail/strided case
also changes input at stable addresses before replay. Boundary cases include
missing injection, shared norm, final mixers, saturation, zeros, and M=0.

BF16 accuracy gates were fixed before device tests: atol=rtol=2^-6, matched
ratio>=0.99, maximum absolute error<=1.0, and matching special values.

The original pinned Triton file supplies the full serial baseline; its SHA256
is checked before loading. Both paths use the same original projection weights.
Timing uses 25 alternating-order samples, eight complete calls per Graph, and
four replays per sample, normalized to one full call. Preserve all raw samples.
The goal is baseline latency / candidate latency >=1.5 for every case.

## Full-call C/(C+V)

Capture full Graph executions with msprof separately from event timing:

```bash
msprof --output=profiles --runtime-api=on --task-time=on --aicpu=on \
  --application="python3 tests/manual/profile_hc_batch.py \
  --extension build-hc/vllm_ascend_hc_C.cpython-311-aarch64-linux-gnu.so \
  --cases tests/manual/qwen_hc_serial_cases.json --out profile-manifest.json"
python3 tests/manual/summarize_hc_batch.py profiles \
  --manifest profile-manifest.json --out cv-share.json
```

The application keeps all sixteen sequential case/implementation Graphs alive
to prevent model-ID reuse. Each has five warm and sixteen retained replays.
The classifier maps actual model IDs, ordered windows and complete V-C-V-C-V
sequences to the manifest. It includes every execution task in each call,
including layout/copy tasks. Unknown or incomplete sequences invalidate the
percentage instead of silently dropping rows.

C_total_us is the mean of both Matmul task sums per complete call. V_total_us is
the mean of all other execution task sums in the same calls.
C/(C+V)=100*C_total_us/(C_total_us+V_total_us). This is a task-duration share,
not Cube hardware utilization or C/Graph latency. Profiler task sums and complete
Graph event time remain separate metrics.
