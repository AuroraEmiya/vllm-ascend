# Qwen3.8 Flash Next HC serial Ascend C path

The implementation provides three Ascend C Vector kernels and two independent
public torch.matmul calls on A3. It implements the Qwen4Exp-named HC path used
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

Load vllm_ascend_C or vllm_ascend_hc_C before using the selected development
entry in vllm_ascend.ops.qwen_hc_serial_matmul:

```python
from vllm_ascend.ops.qwen_hc_serial_matmul import PreparedHCWeights

# Explicit loading stage, before warmup and Graph capture.
projection = PreparedHCWeights(down_weight, up_weight)
combined, block_input, next_injection = projection(
    residual, block_output, previous_injection, norm_weight,
    eps=1e-6, hc_count=hc_count, use_combine=use_combine,
)
```

The object prepares ND and NZ snapshots of Down and an NZ snapshot of Up using
public Torch-NPU APIs. Down selects NZ for M divisible by16 and ND otherwise;
Up uses NZ. This policy follows separate measurements for the two matrix
geometries on the tested A3 software stack. There is no projection merge,
per-replay packing, hidden cache, or forward compilation. Rebuild the object
and recapture after replacing projection weights. Original weights remain
read-only and baseline implementations always use them directly.

Preparation time and physical storage are separate reported costs. Retaining
both Down layouts adds one Down-weight copy to the prepared storage. The
original model weights, if retained, consume additional storage.

This development adapter requires the process default internal-format policy
False and serialized eager/capture calls. It enables the format policy within
candidate preparation/forward and restores False before baseline calls. It
materializes ND projection outputs before the owned Vector kernels when
needed; all device conversion tasks belong to full-path profiling. Captured
Graph replay executes the prepared tasks without repeating Python policy
changes. Concurrent model-serving integration has not been validated.

The original qwen_hc_serial F.linear entry and the unprepared matmul entry remain
available as controls. The default model path is still Triton; this change does
not activate the prepared path automatically in a running model.

HC sources are split into `csrc/qwen_hc/op_host` (Torch registration,
resource queries and pure host tiling), `op_kernel` (three Vector stages), and
shared launch/tiling records. Both the HC-only and regular extension targets use
these files. Owned C++ uses camelCase/PascalCase; public registered names retain
the framework ABI.

`queryDeviceResources` runs under the input-device guard, queries Vector Core
count through `aclrtGetDeviceInfo`, and resolves UB capacity through the current
SoC's public platform API. Missing or zero resources fail explicitly. The cache
stores only immutable hardware information per device. The Torch-free platform
bridge uses the CANN library ABI; the rest of the host code retains Torch's ABI.
No core count or UB capacity is guessed. DMA/vector instruction widths, API
limits and measured work-grain policies have named constants.

The host computes task count, batch, tile width and exact live UB bytes before
launch. Block count is bounded by the queried Vector Core count. CombineNorm
chooses a divisor of K, up to four branches, from total groups and a target work
density of half the queried cores; H alignment and exact UB capacity can reduce
that batch. The final launch still uses min(task count, queried cores). This
preserves the BF16 combine boundary before RMS and uses the existing A3
memory-vector MulAddDst combine/affine path.

SiLU uses public `Silu<float>` and a row batch bounded by row waves, a measured
4 KiB input grain, current UB capacity and DMA block-count limits. Gate chooses
an aligned column tile using floor(queried cores / rows), a measured minimum of
eight FP32 vector repeats, and the actual UB budget. Its small-K path batches
HC branches and uses direct FP32 division plus a two-level reduction for K4.
Generic tail, stride, shared-weight and larger-K fallbacks remain. No C calls or
V stages are reordered across C.

The selected algorithm is the eighth candidate of the device-tiling budget.
Later 8 KiB SiLU grain and single-allocation SiLU experiments were measured and
rejected by the frozen full-Graph ranking; they are not combined into this
implementation. Pure host tiling tests exercise seven core counts and four UB
capacities, coverage/launch bounds, alignment and explicit capacity rejection:

```bash
g++ -std=c++17 -Wall -Wextra -Werror \
  tests/cpp/test_qwen_hc_tiling.cpp \
  csrc/qwen_hc/op_host/qwen_hc_tiling.cpp -o test-hc-tiling
./test-hc-tiling
```

## Device correctness and Graph timing

The development target has been built on Ascend910_9362 with CANN9.2.0,
Torch2.10.0, and torch_npu2.10.0.post7. Use fresh build directories after source
changes. Inspect the actual binary identity before measuring.

```bash
python3 tests/manual/test_qwen_hc_optimization_npu.py \
  --extension build-hc/vllm_ascend_hc_C.cpython-311-aarch64-linux-gnu.so \
  --mode validate --out boundary-results.json
python3 tests/manual/test_qwen_hc_optimization_npu.py \
  --extension build-hc/vllm_ascend_hc_C.cpython-311-aarch64-linux-gnu.so \
  --cases tests/manual/qwen_hc_decode_generalization_cases.json \
  --mode validate --out decode-correctness.json
python3 tests/manual/test_qwen_hc_optimization_npu.py \
  --extension build-hc/vllm_ascend_hc_C.cpython-311-aarch64-linux-gnu.so \
  --cases tests/manual/qwen_hc_decode_generalization_cases.json \
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
is checked before loading. Candidate snapshots preserve the original BF16
weight values; the original Triton and Triton-matmul control use original
weights and the default format policy. The test records preparation, physical
formats/storage, value equality, and rebuilding after a weight change.
Timing uses 25 alternating-order samples, eight complete calls per Graph, and
four replays per sample, normalized to one full call. Preserve all raw samples.
The goal is baseline latency / candidate latency >=1.5 for every case.

## Full-call C/(C+V)

Capture full Graph executions with msprof separately from event timing:

```bash
msprof --output=profiles --runtime-api=on --task-time=on --aicpu=on \
  --application="python3 tests/manual/test_qwen_hc_optimization_npu.py \
  --extension build-hc/vllm_ascend_hc_C.cpython-311-aarch64-linux-gnu.so \
  --cases batch-cases.json --mode profile --out profile-manifest.json"
python3 tests/manual/summarize_hc_batch.py profiles \
  --manifest profile-manifest.json --out cv-share.json
```

Split the 52-point manifest into batches of at most eight cases. Profile one
batch per process; the application keeps all three implementations' Graphs alive
to prevent model-ID reuse. Preserve each batch manifest and raw CSV. Each has five warm and sixteen retained replays.
The classifier maps actual model IDs, ordered windows and complete V-C-V-C-V
sequences to the manifest. It includes every execution task in each call,
including layout/copy tasks. Unknown or incomplete sequences invalidate the
percentage instead of silently dropping rows.

C_total_us is the mean of both Matmul task sums per complete call. V_total_us is
the mean of all other execution task sums in the same calls.
C/(C+V)=100*C_total_us/(C_total_us+V_total_us). This is a task-duration share,
not Cube hardware utilization or C/Graph latency. Profiler task sums and complete
Graph event time remain separate metrics.

## Optimization method and validation scope

The optimization process uses CANNBot as a method and knowledge library:

- Model Down and Up independently by M/N/K, dtype and input layout. Measure the
  complete path before changing tiling or public matrix entry points.
- Use the arch22/A3 route for UB capacity, branch/row batching and memory-vector
  operations. Preserve each branch reduction and every BF16 boundary.
- Rank candidates by measured stage costs, data reuse, transfer bytes and buffer
  lifetimes. Record rejected candidates and all auxiliary execution tasks.
- Keep Graph event latency separate from full msprof task sums. Verify the
  original complete baseline and an additional same-public-API control.

The frozen generalization manifest covers 26 explicit M values times two
phases at H2560/K4/R320/BF16. It tests one model geometry with synthetic
operands and does not estimate production traffic frequency or serving
throughput. Ten boundary cases additionally cover nonaligned dimensions,
strides, missing injection, shared norm, K2/6/8/9, empty rows and saturation;
the default boundary invocation covers six, and the additional four use
qwen_hc_tiling_boundaries.json. No cross-model performance claim is made.

Both the HC-only and regular `vllm_ascend_C` targets were built and imported on
the recorded A3 stack. The frozen HC-only artifact passed 52 decode points and
ten boundaries with 156 valid complete-path profiles. The regular extension
passed the eight acceptance points and ten boundaries with 24 valid profiles.
Its complete-Graph speedup against the original full Triton path was
1.192-1.232x; the required 1.5x per-point goal was not met. These are separate
artifact-specific collections, not one combined benchmark.

Focused repository checks, Markdown lint and explicit C++ clang-format checks
passed. The full `format.sh ci` attempt timed out while installing the actionlint
environment; its completion remains unverified. Model-serving activation and
concurrent eager/capture execution remain outside the validated integration.
