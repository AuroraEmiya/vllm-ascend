# MLA Prolog: three-Vector tuning and SuperKernel

来自我们自己的 R5 三矩阵 MLA Prolog，非 Contact。源码基点与逐文件来源见 `SOURCE_PROVENANCE.json`。
此目录可单独编译，不需要 vLLM 或整个 vllm-ascend 仓库。CANN 通用矩阵库仍是外部依赖，矩阵设备实现未复制进来。

追加十轮优化的结果、80组 B/S 表和逐轮数据见 [reports/20261106/REPORT.md](reports/20261106/REPORT.md)。完整 MLA 相对融合基线1.5×的目标未达到；最终组合改善了原串行流程。tiling、双缓冲和事件生命周期的设计见 [TUNING_NOTES.md](TUNING_NOTES.md)。

执行流程：合并下投影 Matmul → Vector RMSNorm + KV RoPE/cache → 上投影 Matmul → Vector split + Q RoPE → 分 head Matmul → Vector 输出寻址（B×S>1）。权重合并/NZ准备只在初始化和显式 refresh 时发生。所有矩阵输入、输出保持 BF16，Vector 内部用 FP32 完成归约和 RoPE。

源码入口：

- `mla_superkernel/pipeline.py`：完整串行流程与矩阵后端选择。
- `mla_superkernel/compiler.py`：scope、静态编译、outer Graph 分组。
- `csrc/attention/mla_prolog_serial/`：Vector 子算子的 `op_host/` 和 `op_kernel/`；`legacy/` 是原版本兼容入口。
- `csrc/aclnn_torch_adapter/serial_matmul_v3/`：已用的 CANN L0 V3 适配器；它调用外部 CANN 矩阵实现。
- `csrc/binding.cpp`、`mla_superkernel/build_extension.py`：独立 Torch 绑定及 ABI 桥。
- `scripts/run_study.py`：正确性、同地址更新、Graph Event 和设备 trace 的复现入口。错误自动保存到 `error.txt` 和 `result.json`。
- `scripts/bench_vectors.py`：每个 Vector 的同进程配对测试，`--modes adaptive` 使用组合策略。
- `scripts/bench_pipeline.py`：原版/优化版/安装的 `aclnnMlaProlog` 完整流程配对，支持 S0/S1/S2、同地址更新和 trace。
- `scripts/bench_shape_grid.py`：执行 `shape-plan.json` 的 80 组 B/S 采样，逐组保存结果与失败。
- `scripts/inspect_static_metadata.py`：提取本次静态编译包的 SuperKernel 能力元数据。

构建环境是 Linux aarch64、A3、CANN 9.2.0、PyTorch 2.10.0、Torch-NPU 2.10.0.post7，需 C++17 编译器、CMake、ninja 和 CANN opdev 头文件。L0 是版本绑定的非稳定接口；其他版本应重新验证。最终 `REPORT.md` 区分新编译、新上板和历史证据。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /path/to/this/directory
export MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
bash scripts/build.sh 2>&1 | tee build.log
source scripts/env.sh
python scripts/run_study.py --backend v3_nz --mode S2 --batch 1 --output outputs/v3-nz-S2-B1 --profile
python scripts/run_study.py --backend torch_matmul_nz --mode S2 --batch 1 --output outputs/torch-matmul-nz-S2-B1 --profile
python scripts/bench_pipeline.py --batch 2 --sequence 16 --mode S2 --vector-modes adaptive --output outputs/tuned-b2-s16 --profile --refresh-check
python scripts/bench_shape_grid.py --mode S0 --vector-modes adaptive --selection all --output outputs/bs-grid
```

`scripts/build.sh` 使用当前 CANN 安装内的 `msopgen` 和 CMake 支持生成构建目录，再放入这里的四个 MLA Vector 源码并编译 Torch 绑定。通用构建框架来自本机 CANN 安装，不放入交付包。每次从新的 build 目录开始；构建是否通过见最终 `REPORT.md` 中的新环境验证记录。

Torch 绑定的 schema 未变，可通过 `MLASK_EXISTING_BINDING=/absolute/path/to/our/native_binding.so` 复用 ABI 匹配的自有桥接二进制。三个 Vector 的 OPP 必须从本版 host/kernel 重新编译，不能把旧 R5 OPP 用于新的 mode20/21/30/32/12/14。`scripts/build.sh` 在复用绑定时仍会生成并编译本版 OPP；`scripts/env.sh` 选择本目录的新 OPP，结果记录实际加载的二进制 hash。

矩阵后端：

| 参数 | 两个二维投影 | 第三个分 head 投影 |
|---|---|---|
| `torch` | 原公开 `torch.matmul` ND/NZ 路由 | 原 `torch.bmm` / `npu_transpose_batchmatmul` |
| `torch_matmul` | 同原公开 ND/NZ 路由 | `torch.matmul`，继续使用我们的 R5 Vector 输出布局 |
| `torch_matmul_nz` | `torch.matmul` + 已准备 NZ 权重 | `torch.matmul`，继续使用我们的 R5 Vector 输出布局 |
| `v3` | 原 L0 MatMulV3，ND 转置权重 | 原 L0 BatchMatMulV3 |
| `v3_nz` | 原 L0 MatMulV3，NZ 权重 | 原 L0 BatchMatMulV3 |

对每个 backend、B=1/32 分别运行 `--mode S0`（eager）、`S1`（静态子 kernel，outer 分组关）、`S2`（同静态子 kernel，outer 分组开），每次使用新的 `--output`。不可跨源码/OPP 版本复用静态 cache。S1/S2 子 kernel 都以 SuperKernel-ready 方式编译，区别是 capture 完毕后是否调用 `graph.super_kernel_optimize()`。

一次 Graph 内仅一个完整 MLA 调用。Graph Event 每组计10次 replay、20组保留原始值；profiling 另记，不能用两者相减计算 C/V。`--profile` 保存框架 profiler 的原始设备任务，分组结论仍需结合静态 JSON 的 `supportSuperKernel` / `sub_operator_kernel_name` 与真实设备任务名称审核。编译/执行成功不能证明所有矩阵已合入 SuperKernel。

输入约束：BF16、1≤B≤65536、S≥1、B×S≤65536；Q rank=1536、KV rank=512、NoPE=128、RoPE=64、PA_BSND cache（block size 16/128）、INT64 去重且合法的 cache index。`N` 是本卡 head 数，支持 1/2/4/8/16/32/64/128。H 支持 1024/2048/3072/4096/5120/6144/7168/7680/8192。空输入和 packed-T 接口未实现。

公共形状依据见 [CANN aclnnMlaProlog API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnMlaProlog.md)。A2/A3 文档未给 S 固定上限；本实现选择上述有限非空子域。B/S 主网格按倍增步长取值，再加入轴向边界及非 2 次幂邻点。它们是公开接口范围内的合成采样，不是模型 trace 或形状推导工具的输出。

当前默认 `_vector_policy='tuned'`。原算法对照设为 `'baseline'`，显式 `_vector_modes=(norm,split,layout)` 可复现每轮候选。选择只依赖静态 B×S：Norm 在较大输入采用两套 gamma 常驻；Split 小输入使用粗 head tile，大输入使用连续 head 读取和内部寻址；Layout 小输入用 8 行分块，较大输入用 2 行分块。每个子算子的 host 按实际 AIV 数量、UB 容量、尾部行数和 head 数计算 tiling。代码保留未入选模式，最终报告列出它们的回退。

三个 Vector 的双缓冲以按槽的 `MTE3_MTE2` 或 `MTE3_V` 事件保护数据生命周期，不使用 `PIPE_ALL`。子 kernel 返回前排空槽事件，以保留静态编译及 SuperKernel 串行依赖。矩阵仍由外部 CANN/Torch 实现；是否完整分组要以本次设备 trace 和静态 metadata 为准。

刷新输入和权重时保留源 Tensor 地址，对已准备的矩阵调用 `pipeline.refresh_weights()`，该动作在 Graph replay 外执行；地址或 shape 改变时重新捕获。CPU FP64 仅用于 reference，不在 NPU 串行实现内提升 BF16 矩阵输入。
