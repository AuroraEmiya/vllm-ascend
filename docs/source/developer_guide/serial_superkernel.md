# Serial operators and SuperKernel

Both operators live in this single vLLM-Ascend checkout. Start with the Python
operator, then follow its matrix and vector calls. This layout follows the
existing repository's `ops/`, `compilation/`, ACLNN adapter, custom-op and test
directories.

## Source map

| Responsibility | Source |
| --- | --- |
| TextGatedResidual full serial pipeline | [text_gated_residual.py](../../../vllm_ascend/ops/text_gated_residual.py) |
| MLA Prolog full serial pipeline | [mla_prolog_serial.py](../../../vllm_ascend/ops/mla_prolog_serial.py) |
| Shared matrix primitive | [serial_matmul.py](../../../vllm_ascend/ops/serial_matmul.py) |
| Static compilation and Graph grouping | [serial_superkernel.py](../../../vllm_ascend/compilation/serial_superkernel.py) |
| Torch tensor, workspace and stream adapter | [torch_adpt.h](../../../csrc/aclnn_torch_adapter/serial_matmul_v3/torch_adpt.h) |
| Installed CANN L0 Matmul bridge | [opdev.cpp](../../../csrc/aclnn_torch_adapter/serial_matmul_v3/opdev.cpp) |
| Optional matrix build target and ABI boundary | [serial_matmul.cmake](../../../cmake/serial_matmul.cmake) |
| TGR custom-op host and device code | [x_tgr_vector](../../../csrc/moe/x_tgr_vector/) |
| MLA custom-op host and device code | [x_mla_vector](../../../csrc/attention/x_mla_vector/) |
| Shared extension registration | [torch_binding.cpp](../../../csrc/torch_binding.cpp) |

`ops/serial_superkernel.py` only re-exports the original imports for compatibility.
There is one compilation implementation and one matrix wrapper.
Matmul arithmetic is supplied by CANN. There is no handwritten Matmul
SuperKernel device source in this patch.

## Call paths

TGR: group Norm -> down/gate Matmul -> activation -> up Matmul -> residual/gate mix.
MLA: merged down Matmul -> Norm/RoPE/cache update -> query up Matmul -> split/RoPE
-> head Matmul -> output layout.

The Python operator expresses these dependencies. Its `compile_superkernel()`
constructs `CompiledSerialPipeline` from `compilation/serial_superkernel.py`.
The shared matrix wrapper invokes `_C_ascend.serial_matmul_v3_out`; the C++
adapter calls installed `l0op::MatMulV3Nd` or `BatchMatMulV3Nd`.
The vector calls invoke their separately registered custom operators.
The scope and static compilation make these children available for explicit
outer Graph grouping. Dependencies remain serial.

## Build and preparation

This experimental bridge is pinned to CANN 9.2 L0 signatures and opdev ABI.
Recorded hardware validation used A3 Ascend910_9362, Torch 2.10.0 and
Torch-NPU 2.10.0.post7. Revalidate other versions and hardware.

```bash
VLLM_ASCEND_ENABLE_SERIAL_V3=1 python setup.py build_ext --inplace
cd csrc
bash build.sh --pkg --ops=x_tgr_vector,x_mla_vector --soc=ascend910_93 --vendor_name=serial -j4
```

The matrix flag defaults off. Its opdev target uses `CANN_OPDEV_CXX11_ABI=0`
separately from the Torch extension ABI. The target lives under
`aclnn_torch_adapter/` because it adapts an installed operator; no local
Matmul device kernel is compiled. Install the generated OPP into this
checkout and source its vendor `bin/set_env.bash` before running device tests.
The two unchanged vector operators may also be built/installed separately
using `--ops=x_tgr_vector` and `--ops=x_mla_vector` with distinct vendor names.

## Capture and replay

```python
import torch
import torch_npu

from vllm_ascend.ops.mla_prolog_serial import MlaPrologSerial
from vllm_ascend.ops.text_gated_residual import TextGatedResidualSerial

# Construct one of the operators above with its documented weights first.
torch.npu.config.allow_internal_format = True
compiled = prepared_serial.compile_superkernel()
stream = torch.npu.Stream()
stream.wait_stream(torch.npu.current_stream())
with torch.npu.stream(stream):
    for _ in range(10):
        outputs = compiled(*inputs)
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        outputs = compiled(*inputs)
    compiled.optimize_graph(graph)
    graph.replay()
    stream.synchronize()
```

Prepare format-sensitive weights after enabling internal formats. Keep inputs,
weights and caches alive. Warmup, compilation, preparation, capture and grouping
are outside replay timing. Refresh same-address source weight changes using
TGR `refresh_weights_()` or MLA `refresh_weights()` outside replay and order
the update stream before replay. Address/shape/stride changes require recapture.
MLA cache indices must be distinct and in range; validate CPU metadata before
capture. Alias and mutation semantics are unchanged.

`optimize=False` uses identical static children without outer grouping.
`matrix_backend="torch"` retains the public matrix route, `"v3"` uses ND V3,
and `"v3_nz"` uses prepared NZ weights where supported. This matrix choice
does not change the formal MLA fused baseline `aclnnMlaProlog`.
The retained configuration remains `constant_codegen=1`, `preload_code=5`,
default `split_mode=4`, TGR pad640/batched activation, and MLA row layout with
sixteen-head split tiles. These are opt-in setup choices, not automatic routing.
Matrix inputs/outputs and HBM intermediates stay BF16; local vector FP32 and
public TGR FP32 inverse RMS retain their original semantics. FP64 is CPU-oracle
work outside device capture.

## Tests and benchmarks

| Purpose | Location |
| --- | --- |
| CPU metadata and preparation guards | `tests/ut/ops/test_text_gated_residual.py`, `test_mla_prolog_serial.py` |
| Compiler lifecycle and matrix routing tests | `tests/ut/compilation/test_serial_superkernel.py` |
| Single-call NPU Graph and same-address update guards | `tests/e2e/pull_request/one_card/ops/test_*_superkernel.py` |
| Independent reference, case manifests and performance harness | `benchmarks/kernels/text_gated_residual/`, `benchmarks/kernels/mla_prolog/` |

```bash
LOCAL_WORLD_SIZE=1 pytest -q tests/e2e/pull_request/one_card/ops/test_text_gated_residual_superkernel.py tests/e2e/pull_request/one_card/ops/test_mla_prolog_superkernel.py
```

The full benchmark uses ten complete calls per Graph and twenty alternating
Event pairs; the single-call tests are correctness guards. Inspect marked
complete device replay to establish grouping. One mixed task per ten-call Graph
does not mean one task per individual call. Mixed task duration does not expose
child C/V times; report these as null and keep static-control diagnostics separate.
Full grouping does not guarantee speedup or serving integration.

This refactor preserves the retained algorithm and historical performance
evidence. Its current source/binary identity and verification are recorded in
the task's separate refactor report; old timings do not certify a new binary.
