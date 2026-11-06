# Standalone MLA Prolog / SuperKernel study

来自我们自己的 R5 三矩阵 MLA Prolog，非 Contact。源码基点与逐文件来源见 `SOURCE_PROVENANCE.json`。
此目录可单独编译，不需要 vLLM 或整个 vllm-ascend 仓库。CANN 通用矩阵库仍是外部依赖，矩阵设备实现未复制进来。

执行流程：合并下投影 Matmul → Vector RMSNorm + KV RoPE/cache → 上投影 Matmul → Vector split + Q RoPE → 分 head Matmul → Vector 输出寻址（B>1）。权重合并/NZ准备只在初始化和显式 refresh 时发生，继承我们的版本。

源码入口：

- `mla_superkernel/pipeline.py`：完整串行流程与矩阵后端选择。
- `mla_superkernel/compiler.py`：scope、静态编译、outer Graph 分组。
- `csrc/attention/mla_prolog_serial/`：Vector 子算子的 `op_host/` 和 `op_kernel/`；`legacy/` 是原版本兼容入口。
- `csrc/aclnn_torch_adapter/serial_matmul_v3/`：已用的 CANN L0 V3 适配器；它调用外部 CANN 矩阵实现。
- `csrc/binding.cpp`、`mla_superkernel/build_extension.py`：独立 Torch 绑定及 ABI 桥。
- `scripts/run_study.py`：正确性、同地址更新、Graph Event 和设备 trace 的复现入口。错误自动保存到 `error.txt` 和 `result.json`。

构建环境是 Linux aarch64、A3、CANN 9.2.0、PyTorch 2.10.0、Torch-NPU 2.10.0.post7，需 C++17 编译器、CMake、ninja 和 CANN opdev 头文件。L0 是版本绑定的非稳定接口；其他版本应重新验证。最终 `REPORT.md` 区分新编译、新上板和历史证据。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /path/to/this/directory
export MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
bash scripts/build.sh 2>&1 | tee build.log
source scripts/env.sh
python scripts/run_study.py --backend v3_nz --mode S2 --batch 1 --output outputs/v3-nz-S2-B1 --profile
python scripts/run_study.py --backend torch_matmul_nz --mode S2 --batch 1 --output outputs/torch-matmul-nz-S2-B1 --profile
```

`scripts/build.sh` 使用当前 CANN 安装内的 `msopgen` 和 CMake 支持生成构建目录，再放入这里的四个 MLA Vector 源码并编译 Torch 绑定。通用构建框架来自本机 CANN 安装，不放入交付包。每次从新的 build 目录开始；构建是否通过见最终 `REPORT.md` 中的新环境验证记录。

若已有我们原 R5 版本编译好的绑定和 OPP，验证 `torch.matmul` 接口时可直接设置 `MLASK_EXISTING_BINDING=/absolute/path/to/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so`、对应的 `ASCEND_CUSTOM_OPP_PATH` 和 `LD_LIBRARY_PATH`，再运行研究脚本；无需为这一步重新编译 Vector 或安装 vLLM。该路径必须是自己的 R5 产物，研究结果会记录文件 hash。

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

输入约束继承原实现：BF16、S=1、B≤128、Q rank=1536、KV rank=512、NoPE=128、RoPE=64、PA_BSND cache、INT64 去重且合法的 cache index。`N` 是本卡 head 数。能力研究固定 H=7168/N=32、B=1/32，属于合成覆盖子集。

刷新输入和权重时保留源 Tensor 地址，对已准备的矩阵调用 `pipeline.refresh_weights()`，该动作在 Graph replay 外执行；地址或 shape 改变时重新捕获。CPU FP64 仅用于 reference，不在 NPU 串行实现内提升 BF16 矩阵输入。
