# Contact serial MLA Prolog

All native source is under this operator root. Each Vector stage has `op_host/`
(definition, shape inference, resource-derived tiling) and `op_kernel/` (Ascend C
and matching tiling layout). `torch_binding.cpp` registers three out operators;
`kernel_common.h` contains shared copy/sync helpers.

`vllm_ascend/ops/mla_prolog_contact.py` is the complete forward: down-Q matmul,
down-KV matmul, norm/cache Vector, up-Q matmul, split/RoPE Vector, per-head up-K
matmul, query-layout Vector. No SuperKernel scope/grouping is present. Native
address calculations write each layout; no Torch reshape/transpose/cat or
combined weight preparation is used. Input weights follow the public V3 format:
three dense NZ weights and ND per-head UK. FP32 arithmetic stays in Vector UB.
Every matrix operand and intermediate in GM is BF16.

The job manifest records the finite S1 decode domain. Quantization, combined
caches, query_norm and prefill are outside this version. Load the binding before
capture. Host tiling queries AIV cores and UB capacity and rejects unsupported
geometry/resources. Each Vector has two input and two output buffers.

Build with the repository entry:
`bash csrc/build.sh --pkg --ops=contact_norm,contact_split,contact_layout --soc=ascend910_93 --vendor_name=contact --ccache false -j4`.
Install the binary CANN run package under `vllm_ascend/_cann_ops_custom` inside
the task Docker. Source transfer uses GitHub commits and server fetch/checkout.
The ordinary extension supports `VLLM_ASCEND_ENABLE_CONTACT_SERIAL`; the
standalone loader builds the identical bridge for isolated operator validation.

The final execution option is static compilation of these same seven children:

```python
from vllm_ascend.ops.mla_prolog_contact import compile_contact_static

run = compile_contact_static()
run(*args)  # Lazy static compile/install and first execution; outside capture.
torch.npu.synchronize()
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    outputs = run(*args)
graph.replay()
torch.npu.synchronize()
```

The factory explicitly sets `super_kernel_optimize=False`; there is no scope or
binary grouping step. The four matrix calls remain literal `torch.matmul`.
Callers own all inputs, weights and caches. Update values at stable addresses
between replays; compile/capture again when shape, dtype or addresses change.
Compilation/installation and capture costs must be reported separately from
steady replay. Static caches must be isolated by the custom operator revision.

Final host tiling derives all live storage from the input and queried resources:

| Stage | Work units | UB storage |
|---|---|---|
| Norm/cache | B, one combined Q/KV row | Two packed BF16 input/output buffers; FP32 RMS/RoPE scratch |
| Split/RoPE | B × ceil(N/headTile), headTile=min(N,16) | Two packed BF16 input/output buffers; FP32 RoPE scratch |
| Query layout | ceil(B/rowTile) × ceil(N/headTile) | Two bound BF16 buffers, 4×headTile×rowTile×KV bytes total |

Layout chooses rowTile from 1/2/4, retaining enough units for the queried AIV
cores when possible and fitting UB. Its strided DMA handles the partial final
row group directly. No Vector arithmetic or extra layout operator is needed.
Norm and Split retain their next-unit prefetch. More norm cores, Vector inverse
broadcast, launch-local gamma caching and a separate NoPE queue were measured
and excluded because they did not establish stable gains in this decode domain.
