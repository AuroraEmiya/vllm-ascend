# TextGatedResidual vector stages

This directory contains only the owned Ascend C vector operator. Its public
registration name and all host tiling/device arithmetic are unchanged.

- `op_host/`: operator definition, shape inference and host tiling.
- `op_kernel/`: device implementation and tiling data.
- `*_torch_adpt.h`: Torch-to-ACLNN adapter, registered in `csrc/torch_binding.cpp`.
- `vllm_ascend/ops/text_gated_residual.py`: complete serial operator and public setup API.

Matrix arithmetic uses the installed CANN implementation. Shared static
compilation/grouping lives in `vllm_ascend/compilation/serial_superkernel.py`.
See [the combined developer guide](../../../docs/source/developer_guide/serial_superkernel.md)
for the source map, build, lifecycle and tests. Historical optimization logs
are kept in the external task evidence, rather than mixed into this source map.
