# Qwen3.8 Flash Next HC serial Ascend C path

This A3 BF16 implementation keeps both projections on public PyTorch linear
interfaces and implements the three vector stages in Ascend C. It targets the
Qwen4Exp-named HC path used by Qwen3.8 Flash Next.

## Functional boundary

Let K be the stream count, H the per-stream width, D=K*H, and R the low rank.
Inputs are residual [M,D], previous block output [M,H], optional previous
injection [M,K], contiguous norm weight with H or D elements, down weight and up
weight [D,R], plus epsilon, K and the final/intermediate mixer flag.

The complete result has three values:

- Updated residual [M,D], freshly allocated.
- Input to the next block [M,H], freshly allocated.
- Next injection [M,K], a view into the down projection storage, or None for a
  final mixer. Its row stride includes discarded alignment padding.

The previous injection is still consumed by a final mixer. Inputs are read only;
the implementation retains no hidden cache. Intermediate down weight has
R+K+P rows, with P=(-(R+K))%16. Final down weight has R rows.

Combine accumulation is rounded to BF16 **before** grouped RMSNorm. The norm
uses Gemma affine z + z*weight. SiLU consumes down[:,:R]/K. Gate mixing is
sum(sigmoid(gate)*normalized, stream)/K. Both matmul outputs and the SiLU output
are materialized in BF16. Dropping any rounding boundary changes the function.

## Source constraints and implemented domain

The pinned Triton source validates two-dimensional matrices, positive K,
D divisible by K, unit inner stride and contiguous H- or D-element norm weights.
It masks tails; H and R need not be powers of two or multiples of512.
Those static checks do not establish all device resource limits.

The Ascend C implementation supports BF16 on A3, positive H/R up to8192,
nonnegative row strides, checked uint32 element indexing, positive finite epsilon,
and M=0. FP16/FP32, quantized weights and TP-sharded projection adapters are outside
this implementation's domain.

Model structural metadata from Zhanlu gives K4,H2560,D10240,R320,
P12, intermediate down [336,10240], up [10240,320] and per-stream norm weights.
Two prefill shapes were evaluated by the formal Zhanlu graph generator. The
decode template's generated token relation produced a negative dimension and is
invalid. The separate decode case uses its unchanged original recorded M16;
it is not a successfully generated decode shape. A5 templates supply structural
metadata only. Performance evidence must come from A3.

The case manifest uses explicit test epsilon1e-6. The deployed model epsilon and
production request distribution remain unverified. These cases establish a
bounded operator workload.

## Integration and development build

Sources follow the main repository's csrc, kernel, CMake and Python directories.
The full registered build discovers the host and kernel files. Development may
use the optional HC-only target from the same root CMake project:

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

Load the registered extension before calling
vllm_ascend.ops.qwen_hc_serial.combine_and_mix.
The explicit combine_and_mix_module adapter accepts an unquantized,
replicated GatedResidual. Selection is explicit during development; the default
Triton model path remains available as the performance baseline.

## Independent device validation and measurement

```bash
python3 tests/manual/test_qwen4_exp_hc_serial_npu.py \
  --extension build-hc/vllm_ascend_hc_C.cpython-311-aarch64-linux-gnu.so \
  --mode validate --out boundary-results.json

python3 tests/manual/test_qwen4_exp_hc_serial_npu.py \
  --extension build-hc/vllm_ascend_hc_C.cpython-311-aarch64-linux-gnu.so \
  --cases tests/manual/qwen_hc_serial_cases.json \
  --mode validate --out model-results.json
```

Validation uses independent CPU float64 mathematics and specified BF16
materialization points. It checks all three outputs, alias/stride metadata, input
mutation, Graph replay, and a changed-input replay. BF16 gates are
atol=rtol=2^-6, matched ratio>=0.99 and max absolute error<=1.0, frozen before the
first NPU result. Boundary coverage includes tails, row strides, shared norm,
missing injection, final mixers, saturation, zero inputs and empty token batches.

Mode bench measures the complete five-stage path with NPUGraph events.
Both implementations use identical public linear calls and operands.
The baseline imports the unchanged original Triton file and verifies its SHA256.
Twenty-five interleaved samples retain raw measurements; latency is normalized by
full calls per graph and replays per sample. The target is original Triton graph
latency / Ascend C graph latency >=1.5 for every frozen case.

Mode profile provides complete graph calls for msprof. Report C/(C+V) from all
matmul task durations and all remaining execution task durations in the same
complete retained calls. Include copy/layout/other execution tasks; unclassified
or incomplete sequences invalidate the percentage. Graph event latency is a
separate metric.

## Current evidence boundary

The registered HC-only target compiled on Ascend910_9362 with CANN9.2.0,
Torch2.10.0 and torch_npu2.10.0.post7. Device correctness, graph speedup and task
share must be taken from retained validation records. The per-job report records
each candidate, failure, baseline identity, environment and remaining budget.
