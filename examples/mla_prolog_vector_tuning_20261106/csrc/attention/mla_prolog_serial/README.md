# MLA Prolog 串行子算子

完整调用在 `mla_superkernel/pipeline.py`，同一 stream 的依赖顺序是：

```text
down projection -> Norm/cache -> up projection -> Split/RoPE -> head matmul -> Layout
```

| 子算子 | 输入 | 输出/状态 | 关键 tiling 字段 |
|---|---|---|---|
| `x_mla_norm_cache` | 合并下投影、两套 gamma、sin/cos、index | Q RMSNorm、KV RMSNorm/RoPE、KV/KR cache 写入 | `branchesPerUnit`：1或2；`floatElements`：FP32工作区长度 |
| `x_mla_split_rope` | Q 上投影、sin/cos | NoPE `[N,T,128]`；Q RoPE `[T,N,64]` | `headTile`：head块长；`inputHeadWidth`：128或192个元素 |
| `x_mla_layout` | head matmul 输出 `[N*T,512]` | query `[T,N,512]` | `rowTile`：8或2；`workElements`：一个缓冲槽的元素数 |

`T=B*S`。每个目录的 `op_host/` 管自己的接口、shape 和 tiling，`op_kernel/` 只有最终计算与双缓冲流水。T=1 的 Layout 在 Python 中省去。

三次矩阵调用在 `csrc/aclnn_torch_adapter/serial_matmul_v3/`，使用外部 CANN L0 MatMulV3/BatchMatMulV3；SuperKernel 包装在 `mla_superkernel/compiler.py`，作用域包含同一完整流程。
