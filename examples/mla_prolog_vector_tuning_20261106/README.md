# MLA Prolog 最终串行实现

本目录是 2026-11-16 的可读性重构版。三个 Vector 只保留最终选取的算法，共用计算合并，调优候选和数字 mode 从生产路径移除。构建/实测状态见 [REFACTOR.md](REFACTOR.md)，历史性能不自动视为本版实测结果。

先读 [pipeline.py](mla_superkernel/pipeline.py) 的 `__call__`，其中六个编号步骤就是执行顺序：

| 步骤 | 入口 | 数据流 |
|---:|---|---|
| 1 | `_project(x, 0)` | `[T,H] @ [H,2112]`，合并 Q/KV 下投影 |
| 2 | `mla_norm_cache_out` | Q/KV RMSNorm、KV RoPE，原地更新 KV/KR cache |
| 3 | `_project(normalized_q, 1)` | `[T,1536] @ [1536,N*192]`，Q 上投影 |
| 4 | `mla_split_rope_out` | NoPE `[N,T,128]`，Q RoPE `[T,N,64]` |
| 5 | `serial_matmul_v3` | `[N,T,128] @ [N,128,512]`，按 head 投影 |
| 6 | `mla_layout_out` | Vector 寻址得到 `[T,N,512]`；T=1 时省去 |

`T=B*S`，`H` 是输入 hidden size，`N` 是本卡 head 数。返回 query `[B,S,N,512]`、query_rope `[B,S,N,64]` 和两个输入 cache 的别名。

```text
mla_superkernel/
  pipeline.py                         完整串行流程、权重准备和刷新
  compiler.py                         static child 编译、SuperKernel 分组
  serial_matmul.py                     Python 到 L0 V3 的矩阵桥
  reference.py                        CPU FP64 oracle
csrc/
  attention/mla_prolog_serial/
    x_mla_norm_cache/                 op_host/ 与 op_kernel/
    x_mla_split_rope/                 op_host/ 与 op_kernel/
    x_mla_layout/                     op_host/ 与 op_kernel/
    torch_adpt.h                      三个明确接口的 Torch 适配
  aclnn_torch_adapter/                CANN L0 V3 调用及 ABI 桥
  binding.cpp                        Torch schema
  benchmark/                         公开融合 V1 黑盒测试桥
scripts/                             构建、完整流程验证和静态元数据检查
reports/20261106/                    原版本历史报告
```

host 直接根据 shape 计算实际 tiling 参数：Norm 的 `branchesPerUnit` 表示一个工作单元处理的 Q/KV 分支数；Split 的 `inputHeadWidth` 是每个 head 搬入的元素数；Layout 的 `rowTile` 是每块行数。没有统一 mode 分派、dummy Tensor 或未入选算法。详见 [Vector 说明](csrc/attention/mla_prolog_serial/README.md) 与 [流水说明](TUNING_NOTES.md)。

默认矩阵后端为最终选取的 `v3_nz`；`torch_matmul_nz` 保留三次 `torch.matmul` 的接口研究，使用相同 Vector。`v3`、`torch_matmul` 为 ND/静态 shape 路由控制，性能与分组状态单独判断。

构建命令见 [BUILD.md](BUILD.md)。本版简化了三个 Vector 的 schema 与 tiling 布局，需要新 OPP、新 Torch 绑定、新 static child，并重新 capture Graph。不能加载旧绑定。S0 是普通 Graph，S1 编译 static child 但不做 outer grouping，S2 在 capture 后执行 `graph.super_kernel_optimize()`；这些控制都执行上表同一条依赖链。

输入沿用 BF16 非量化子域：非空连续 `[B,S,H]`，B<=65536、S>=1、B*S<=65536；H 为 1024/2048/3072/4096/5120/6144/7168/7680/8192；N 为 1/2/4/8/16/32/64/128；cache 为 PA_BSND，block size 为16或128；index 为合法去重 int64。矩阵输入/输出 BF16，Vector 归约/RoPE 内部 FP32，FP64 仅用于 CPU oracle。与融合 API 比较还要取双方共同支持范围。

旧源码固定在本地 Git `55e5d7b9b7f35c8eea266091c3c13ed3c52ed959`，公开版本为 `1c63c2b1f0a12ab32121998082565ecc1dec4c22`。2026-11-10 完整交接包与性能数据保留在工作台 `outputs/package-mla-handoff-20261110/`，历史候选不再参与当前构建。
