# MLA Prolog 源码包编译与复现

验证 `torch.matmul` 接口可以直接复用我们 R5 版本已经编好的 Vector OPP 和 Torch 绑定。只有从源码重建交付包时，才需要下面的完整编译步骤。

本次已验证环境：Linux aarch64、Ascend910_9362 / A3、CANN 9.2.0、Python 3.12、PyTorch 2.10.0、Torch-NPU 2.10.0.post7、C++17、CMake、ninja。Torch 的 Python 依赖也须可导入，包括 sympy。通用矩阵实现来自外部 CANN 库；包里提供我们的 L0 V3 调用桥，不包含 CANN GEMM 设备源码。

解压到一个新的、可写目录，使用已经映射空闲 NPU 的开发容器：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /path/to/mla-superkernel
export MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
bash scripts/build.sh > build.log 2>&1
source scripts/env.sh
```

`build.sh` 会调用安装在 CANN 内的 `msopgen` 生成构建模板，放入本包四个 Vector 原型的 host/kernel，构建并安装到本目录的 `install/`，然后编译 `mla_superkernel_native.so`。成功标志是 `STANDALONE_MLA_BINDING_READY`；编译产物 hash 在 `outputs/build-binaries.sha256`。`prepare_build.py` 拒绝复用已有生成目录，重新从头构建时请使用新的源码目录。无需 vLLM 或 vllm-ascend 源码仓。

正常使用源码包新编译的绑定时，不设置 `MLASK_EXISTING_BINDING`。如果只是复用我们以前编好的 R5 产物做接口试验，可显式配置：

```bash
export PYTHONPATH=/path/to/this/package:${PYTHONPATH:-}
export MLASK_EXISTING_BINDING=/path/to/our/R5/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so
export ASCEND_CUSTOM_OPP_PATH=/path/to/our/R5/vendors/serial_transformer
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}
```

这个选项只加载指定文件，没有自动回退。二进制须与当前 Python、Torch-NPU 和 CANN 匹配。

三次都使用 `torch.matmul`、保持与 V3 对照相同的 NZ 权重与 Vector 路由：

```bash
python scripts/run_study.py --backend torch_matmul_nz --mode S0 --batch 1 --output outputs/torch-S0-B1
python scripts/run_study.py --backend torch_matmul_nz --mode S1 --batch 1 --output outputs/torch-S1-B1 --profile
python scripts/run_study.py --backend torch_matmul_nz --mode S2 --batch 1 --output outputs/torch-S2-B1 --profile
python scripts/run_study.py --backend v3_nz --mode S2 --batch 1 --output outputs/v3-S2-B1 --profile
```

将 `--batch 1` 改为 `32`，并更换输出目录，即可复现另一组 decode shape。`torch_matmul` 保留原有 ND/NZ shape 路由且第三次也用 `torch.matmul`；`torch` 则保留原实现第三次的 `bmm` / `npu_transpose_batchmatmul`，两者在报告中单列。

程序先在 capture 外完成准备与静态编译，再捕获一次完整 MLA 调用；S2 在 capture 完成后调用 `graph.super_kernel_optimize()`。S1 和 S2 的静态子 kernel 编译选项相同，区别是 outer Graph 是否分组。每次运行使用新的输出目录，保留不同版本、模式的静态 cache。

核心调用方式如下，实际 Vector 调用就在 `pipeline.__call__()` 中，与三次矩阵计算交替执行：

```python
compiled = pipeline.compile_superkernel(matrix_backend="torch_matmul", optimize=True)
compiled(token_x, rope_sin, rope_cos, index, kv_cache, kr_cache)
torch.npu.synchronize()  # 编译预热在 capture 外
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    outputs = compiled(token_x, rope_sin, rope_cos, index, kv_cache, kr_cache)
compiled.optimize_graph(graph)
graph.replay()
```

Scope 和编译设置只表达分组请求。检查 `kernel_details.csv` 及静态子 kernel JSON，才能确定哪些任务实际进入 SuperKernel。报告保存了三次 `torch.matmul` 正常执行、但三次矩阵任务仍独立的实例，也保存了 L0 V3 整条流程成为一个 SuperKernel 的实例。

本次 FP64 仅出现在 CPU reference；NPU 矩阵输入、输出均为 BF16。权重合并、ND/NZ 准备和显式 `refresh_weights()` 在 replay 外执行，计时不包含这部分准备成本。地址或 shape 改变时需要重新 capture。
