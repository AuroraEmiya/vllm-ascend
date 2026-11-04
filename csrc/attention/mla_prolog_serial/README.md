# MLA complete serial operator

Python pipeline: `vllm_ascend/ops/mla_prolog_serial.py`.
Each direct Vector child owns `op_host/` and `op_kernel/`. `legacy/` retains the previous native control.
Matrix computation is shared CANN MatMulV3 through `csrc/aclnn_torch_adapter/serial_matmul_v3/`; no copied matrix kernel.
Static and SuperKernel paths use the same child operators and unchanged caller-owned Graph lifetime.
