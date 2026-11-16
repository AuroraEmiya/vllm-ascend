# 最终三个 Vector 的 tiling 与流水

本版只保留最终实现，历史候选见 Git 基点 `55e5d7b9b7f35c8eea266091c3c13ed3c52ed959`。当前 host、kernel 与 Torch 接口不传算法 mode。

| 子算子 | 静态分块 | 共用计算 | 双缓冲与同步 |
|---|---|---|---|
| Norm/cache | T<512 每工作单元一支，其余同一 token 两支 | 同一 RMSNorm 和 KV RoPE | 两个 BF16 输出槽，`MTE3_V` 等待将复用的槽；输入/工作区用 `V_MTE2`、`MTE2_V`、`PIPE_V` |
| Split/RoPE | T<32 最多16个 head；其余根据 AIV 数选 tile，连续搬入完整 head | 同一 even/odd Gather、广播旋转、结果写回 | 两套输入/系数/输出槽，`MTE3_MTE2` 同时保护 NoPE 和 QR 输出 DMA |
| Layout | T<=32 每块8行，其余2行；head块受工作数与UB限制 | 同一个带 stride 的 DMA 寻址 | 两个 tile，`MTE2_MTE3` 等待输入，`MTE3_MTE2` 保护槽复用 |

`branchesPerUnit` 是实际工作分支数，`inputHeadWidth` 是128/192个 BF16 元素，`rowTile` 是实际行块长。`floatElements` 只表示 FP32 工作区，不再借作 Layout 行块字段。

host 查询实际 AIV 数与 UB 容量，计算 units、blockDim、工作区与 tail。kernel 返回前排空两槽事件，不使用 `PIPE_ALL`。Tensor view 只改变连续存储的元数据，真正布局变换在 Vector 内通过寻址完成。

SuperKernel 仍包围完整 Matmul/Vector 交替流程。本版需新 OPP、绑定、static child 和重新 capture；原实测数据与原源码的身份保留在旧交接包中。
