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
