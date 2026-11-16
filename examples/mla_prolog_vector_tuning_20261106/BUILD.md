# 构建和验证

本版简化了三个 Vector 的接口。必须用新源码目录将 OPP、Torch 绑定和 static child 一起重建，再重新 capture Graph。设置旧 `MLASK_EXISTING_BINDING` 会被拒绝。

原测量配套为 Linux aarch64、A3 Ascend910_9362 / ascend910_93、CANN 9.2.0、Python3.12、Torch2.10.0、Torch-NPU2.10.0.post7，以及 C++17、CMake、ninja、sympy。重构验证状态见 [REFACTOR.md](REFACTOR.md)。矩阵库来自安装的 CANN，本包只含调用桥。

在自己的开发 Docker 内通过 Git clone/pull 获取代码，然后运行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /path/to/mla-prolog/source
unset MLASK_EXISTING_BINDING
export MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
bash scripts/build.sh > build.log 2>&1
source scripts/env.sh
```

`prepare_build.py` 使用安装版 msopgen 生成三个算子的项目，放入各自 host/kernel。`build.sh` 构建 OPP 并安装到 `install/vendors/mlask_transformer/`，随后构建 `build/binding/mla_superkernel_native.so`。成功标志为 `STANDALONE_MLA_BINDING_READY`，产物 hash 在 `outputs/build-binaries.sha256`。无需完整 vLLM/vllm-ascend 仓库。已有生成目录会被拒绝覆盖，请使用新的源码副本重建。

检查设备健康和占用，使用自己的空闲卡验证：

```bash
python scripts/bench_pipeline.py --batch 1 --mode S0 --backend v3_nz --output outputs/b1-s0 --refresh-check
python scripts/bench_pipeline.py --batch 32 --mode S2 --backend v3_nz --output outputs/b32-s2 --profile --refresh-check
python scripts/bench_pipeline.py --batch 32 --sequence 16 --mode S2 --backend v3_nz --output outputs/b32s16-s2
python scripts/run_study.py --batch 32 --mode S2 --backend torch_matmul_nz --output outputs/torch-b32-s2 --profile
```

完整流程验证含 CPU FP64 oracle、公开 `aclnnMlaProlog` 对比、cache 别名/未触碰部分、同地址刷新。Event 计时覆盖完整 replay，每组10次，默认20组；准备、编译、分组和 profiler 在计时外。S2 的实际分组以 kernel_details.csv 和静态元数据确认。

历史 `--vector-modes`、独立模式扫描、legacy/staged、transpose/tile8 选择已经移除。S0/S1/S2 只控制 Graph 编译/分组。
