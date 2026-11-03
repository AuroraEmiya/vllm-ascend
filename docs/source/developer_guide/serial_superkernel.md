# Serial TextGatedResidualSerial SuperKernel

This opt-in A3 decode path compiles the existing serial arithmetic into static
child kernels and groups an outer NPUGraph with SuperKernel. Dependencies remain
serial. A grouped replay executes device child functions without issuing Python
or individual ACLNN launches for each stage.

## Requirements and build

The experimental matrix bridge is pinned to CANN 9.2.0 L0 C++ signatures and the
installed opdev ABI. Validation uses Ascend910_9362 (A3), PyTorch 2.10.0 and
Torch-NPU 2.10.0.post7. Other toolkit ABIs require revalidation.

Build the normal extension with `VLLM_ASCEND_ENABLE_SERIAL_V3=1`; its default is
off. `CANN_OPDEV_CXX11_ABI` defaults to0, separate from the Torch extension ABI.
Build and install the owned `x_tgr_vector` OPP into this checkout:

```bash
VLLM_ASCEND_ENABLE_SERIAL_V3=1 python setup.py build_ext --inplace
cd csrc
bash build.sh --pkg --ops=x_tgr_vector --soc=ascend910_93 --vendor_name=tgr_serial -j4
```

Install the generated OPP under the checkout's `vllm_ascend/_cann_ops_custom`
and source its vendor `bin/set_env.bash` before testing. Keep the retained
baseline OPP available separately when running the comparison harness.
The two wrappers share `serial_superkernel.py` and `csrc/serial_matmul/`.

## Capture and replay

Prepare an instance using its existing public constructor. Enable internal
formats before preparing NZ weights. Keep all inputs, weights and caches alive:

```python
import torch
import torch_npu

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

The first call compiles and installs static children. Warmup, capture, grouping
and weight preparation belong outside replay timing. Address/shape/stride changes
require recapture. Update source weights in place and call
`compiled.refresh_weights_()` for TGR or `compiled.refresh_weights()` for MLA
outside replay, ordering the update stream before replay. MLA cache indices
must be distinct, valid and checked on CPU before capture; cache tensors retain
their original alias and mutation semantics.

`optimize=False` retains the same static children without outer grouping.
`matrix_backend="torch"` uses the retained public matrix route; `"v3"` uses ND
V3 and `"v3_nz"` uses prepared NZ weights. V3 matrix choice is separate from the
MLA fused baseline interface: the formal comparison uses `aclnnMlaProlog`.

The retained full-group configuration uses `constant_codegen=1`,
`preload_code=5` and the framework's default `split_mode=4`. TGR pads constant
down/gate columns to640 and gathers only live336 columns for batched activation;
the additional persistent buffer is12.5MiB. MLA uses sixteen-head Split/RoPE
tiles and an owned whole-row DMA layout kernel. Controls remain explicit through
`pad_down`, `batch_activation`, `head_layout`, `split_head_tile` and
`optimize_options`. These options are static setup choices.

Matrix inputs/outputs and HBM matrix intermediates remain BF16. RMS/vector
arithmetic uses local FP32 where required. CPU FP64 exists only in the independent
test oracle; it is outside captured execution. The adapter fixes logical ND and
physical NZ descriptor metadata without copying or promoting matrix data.

## Validation

The hardware regression checks one complete call per Graph, then updates inputs
and weights at the same addresses. This detects stale first-replay results that
ten repeated calls could hide. MLA also checks touched and untouched cache regions.

```bash
LOCAL_WORLD_SIZE=1 pytest -q tests/e2e/pull_request/one_card/ops/test_text_gated_residual_superkernel.py
```

`benchmarks/kernels/text_gated_residual/superkernel_trial.py` runs the frozen shape manifest,
precision thresholds, original/static/SK controls and optional full msprof.
Use `--single-call-graph --no-timing` for the one-call guard. Performance uses
ten complete calls per Graph and20 alternating Event pairs; this is a separate
protocol from the one-call guard. Compiler workers are capped at4 only in the
benchmark/test process because installed9.2 otherwise reads host CPU count.

Compilation flags alone do not establish grouping. Inspect the marked complete
device replay: representative full-group cases produce one `sk_entry_mix12` per
Graph. Gate-free/auxiliary configurations must be checked individually. Mixed SK
task duration cannot be split into C/V child times from this trace; report null
for those fields and use the ungrouped static control for stage diagnosis.
Full grouping does not guarantee a speedup; the MLA multi-token path can remain
slower than its fused baseline. Per-case results and failures are kept in the
separate development evidence report.

## Public references

- [SuperKernel principles](https://asc.gitcode.com/guide/programming_guide/advanced_programming/super_kernel/principles.html)
- [Operator adaptation](https://asc.gitcode.com/guide/programming_guide/advanced_programming/super_kernel/operator_adaptation.html)
- [Generic ops-nn matmul](https://gitcode.com/cann/ops-nn/tree/v9.2.0/matmul)
