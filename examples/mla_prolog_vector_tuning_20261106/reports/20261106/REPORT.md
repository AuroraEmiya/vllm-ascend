# MLA Prolog 三 Vector：追加十轮优化

已完成并留存 10/10 轮测量；10/10 轮邮件已发送。完整流程共留存 100 次 case 结果，其中 99 次通过且纳入性能对比。

**结论：追加十轮已完成，完整 MLA 相对 aclnnMlaProlog 的 1.5× 目标未达到。** 80组 B/S 的 S0 配对几何均值为原串行/候选1.0668×、融合/候选0.8084×；9组 S2 为1.0714×和0.8533×。因此本版改进了原串行流程，但在这些集合的同权几何均值上仍慢于融合基线。

本轮对象是我们自己的三矩阵、三 Vector 串行实现。性能融合基线固定为安装的 `aclnnMlaProlog`。A3 实测硬件为 Ascend910_9362，20 Cube/40 Vector，CANN9.2.0，Torch2.10.0/Torch-NPU2.10.0.post7。使用 80.48.33.118 的物理卡10，运行前后资源快照随各轮保存；所有容器归本任务所有。

## 形状与计时边界

[CANN 公共 API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnMlaProlog.md) 给出 B=0..65536，A2/A3 的 S 无固定上限。本实现选择非空 BF16 子域 B=1..65536、S≥1、B×S≤65536。主网格 B=[1,2,4,8,16,32,64,128,256]，S=[1,2,4,8,16,32,64,128]，主网格限制 B×S≤4096，补轴向大值和邻点后共80组。固定 H=7168、N=32、Q rank1536、KV rank512、NoPE128、RoPE64；额外 guard 单独列出。shape 是公开接口范围内的合成采样，不是模型 trace/形状推导工具输出。

每轮独立 Vector 计时：Graph 内50个相同 Vector 调用，20组随机交错配对 NPU Event 样本。完整 MLA 计时：Graph 内一次完整调用，每组10次 replay、20组随机交错 Event 样本。S0 是 eager 子算子图捕获，S1 是静态子 kernel 且 outer 分组关闭，S2 开启 outer SuperKernel 分组。编译、权重准备、refresh 和 profiler 时间不计入 Event 性能。不能把独立 Vector 几何均值当作完整 MLA 加速比。

所有矩阵的 NPU 操作数与输出保持 BF16，FP32 仅在 Vector 内做归约/RoPE；CPU FP64 仅用于独立 oracle。全输出与安装的 aclnnMlaProlog 分块比较；≤512 token 使用完整 CPU oracle，较大 case 独立 CPU oracle 抽查四个 token 并完整检查未写 cache。混合阈值 atol=rtol=2^-6、匹配率≥99%、最大绝对误差≤1；布局及未触碰 cache 精确一致。

## 十轮改动、收益与失败

|轮次|假设与修改|候选 mode 的配对几何均值|决策|邮件|
|---|---|---|---|---|
|1|Layout ping/pong 缓冲使相邻块 MTE2/MTE3 可重叠|11: 1.0273×|精度通过；512 行 1.196×，2048 行约 9% 退化，暂不全量采用，继续多行 DMA 分块|1a1b05daac1fb7c8|
|2|两行 token 组成一个 Layout DMA tile，沿 head 一次读取更长连续块并由 UB/GM stride 完成地址重排|11: 1.0257×, 12: 1.0673×|全部精度通过；候选在 512/2048 行分别约 1.246×/1.072×，小行数存在退化；保留作为较大 token 数候选。首次编译的 uint32 narrowing 错误及修复提交已保留|1a1b0678a1ad65c9|
|3|在双缓冲布局路径比较 2/4/8 行输入 DMA tile，降低短 DMA 数量并保留足够 AIV 任务。|12: 1.0853×, 13: 1.1504×, 14: 1.2281×|所有精度检查通过。较大 tile 并非持续更快：512 行下 4/8 行 tile 接近，2048 行下 2 行 tile 更快；保留多种策略供最终跨 shape 复测。|1a1b06fa4fc8d40c|
|4|以双缓冲隔离 NoPE、RoPE 和系数的跨 tile 生命周期，用按 slot 的 MTE3_MTE2 事件代替全阶段等待。|30: 1.2288×|全部精度通过；512/2048 行分别约 1.636/1.876 倍，保留大输入双缓冲候选，小输入继续用占用率策略验证。|1a1b071f99b0c663|
|5|按 token 数、head 数及实际 AIV 核数选择 head tile；足够并行时增大 tile，否则暴露 head 并行。|30: 1.2355×, 31: 1.1135×|精度全部通过。大输入进一步获益：512/2048 行约 2.007/2.609 倍；32 行有轻微回退，最终策略保留其测量结果。|1a1b074f482b28d7|
|6|一次连续读取完整 head，用 Vector 步长 Cast 选择 RoPE、用输出 DMA 的 UB 步长提取 NoPE。|30: 1.2357×, 31: 1.1215×, 32: 1.2081×|精度全部通过。128/512/2048 行分别约 1.521/2.427/3.234 倍；小输入仍受第5轮细 head tile 的回退影响，最终必须保留粗 tile 路径。|1a1b078509311443|
|7|只双缓冲 Norm/cache 的输出，等待即将重用的输出槽，保留共享 FP32 算术工作区。|20: 1.0267×|全部精度通过，但 32/64/128 行只有约 1-2% 收益，512/2048 行近乎持平或微退。输出 DMA 不是这个 Norm 路径的主要限制，继续测计算与工作分配策略。|1a1b07c2925c475b|
|8|将同一 token 的 Q/KV 两支放到同一工作单元，分别缓存两套 FP32 gamma，避免分支切换时反复加载。|20: 1.0251×, 21: 0.9278×|精度通过；512/2048 行约 1.084/1.202 倍，但 32-128 行回退。候选仅考虑较大 token 数，不能按平均值全局启用。|1a1b07f2afacc194|
|9|比较 gamma 常驻路径使用全部、1/2 和 1/4 AIV 核预算，验证任务粒度与并行度的权衡。|20: 1.0264×, 21: 0.9021×, 22: 0.8578×, 23: 0.6654×|精度全部通过，但半核/四分之一核在大输入显著回退，2048 行分别只有约 0.651/0.335 倍原版性能。拒绝减少核数的路径；组合版较大输入使用全部 AIV 核。|1a1b081dcc9d82ff|
|10|组合实测路由：Norm在512行起用双gamma常驻，Split小输入固定tile、大输入连续head读取，Layout小输入8行、大输入2行tile；扩展边界复测。|12: 1.0644×, 14: 1.5220×, 20: 1.0323×, 21: 1.1747×, 30: 1.0606×, 32: 1.6855×|保留按token数及平台资源计算的组合路由。54个独立Vector case、80组S0、9组S2、3组S1和5组额外guard均通过；完整MLA相对融合基线1.5倍目标未达到。V3 NZ保留整流程SuperKernel；torch.matmul执行与精度通过，但整流程分组失败证据已保存。|1a1b0a565cbb2c65|

全部 shape/模式的耗时与精度见 [iteration-performance.csv](iteration-performance.csv)。失败编译保留：第1轮编译安装通过但末尾绑定 hash 路径检查退出1，另有恢复审计；第2轮首个编译因 DMA 参数窄化失败，显式 uint32_t 修复后的 attempt02 重新构建并上板。两次均未计作新增优化轮。

## 完整 MLA 性能

|测试集合|通过 case|原串行/候选几何均值|aclnnMlaProlog/候选几何均值|达到1.5×融合基线|比原串行慢的 case|
|---|---:|---:|---:|---:|---:|
|bs-grid-s0|80|1.0668×|0.8084×|0/80|2|
|contract-guards-s0|5|1.0258×|0.7215×|0/5|1|
|sk-grid-s2|9|1.0714×|0.8533×|0/9|0|
|sk-layout-proof-b32|1|1.0731×|0.7297×|0/1|0|
|static-grid-s1|3|1.0478×|0.8672×|0/3|1|
|torch-matmul-control-b32|1|1.0571×|0.7625×|0/1|0|

逐个 B/S 的三个实现耗时、加速比、执行模式和状态见 [pipeline-performance.csv](pipeline-performance.csv)。完整原始数值、精度、refresh、已加载二进制 hash、准备时间和错误见各 case `result.json`；`pipeline-summary.json`（保留在本地outputs目录） 包含全部 case 的派生结果。基线缺失或失败不会补零，也不会纳入可比几何均值。

80组 B/S 的可读表格见 [B_S_RESULTS.md](B_S_RESULTS.md)，列出每组的融合、原串行、候选耗时及两个加速比；同权几何均值对每个 B/S case 等权，不代表模型请求分布。

S=1 覆盖通常的单 token decode；S>1 是多 token 接口范围采样，未声称全部来自单 token decode 流量。最大轴向 B=65536,S=1 的候选为24875.179μs，原串行28008.531μs、融合30553.357μs，即相对原串行1.1260×、相对融合1.2283×。本轮主网格未达到1.5×的 case 也全部留存。

## SuperKernel 与剩余瓶颈

默认完整流程继续通过 npugraph_ex 静态编译，并在 capture 后调用 `graph.super_kernel_optimize()`。每个 Vector 都按槽等待/归还事件，返回前排空；未使用 PIPE_ALL。是否把整个 MLA 合为一个设备任务由真实 trace 判定，编译成功或 supportSuperKernel 字段不单独视为分组证据。
- `sk-grid-s2` B=1, S=1: 1 个设备任务，SK 1，整流程分组=True；任务名 `['sk_1_mla_prolog_start_static_kernel_MatMulV3_d3882223ade648d1c2397006253f87c3509b9e86d85e66dad908311140ca87ee_1828_d0__kernel0_end_static_kernel_BatchMatMulV3_004d94b0c64c227c322dca8961c515c6ecca88ef4a2a0decdd1c01e98673bfac_1828_d0__kernel0']`。
- `sk-layout-proof-b32` B=32, S=1: 1 个设备任务，SK 1，整流程分组=True；任务名 `['sk_1_mla_prolog_start_static_kernel_MatMulV3_1590ba234049c109392dd524ff38cda0111f2a3002a3b56342b8d73b7b1f1336_29466_d0__kernel0_end_static_kernel_XMlaLayout_d6d5654c998c5bdedba8181c00024ff7b8a20efc9eb24826105aee9e9bb8a5e5_30589_d0__kernel0']`。
- `static-grid-s1` B=64, S=32: 12 个设备任务，SK 0，整流程分组=False；任务名 `['sk_scope_kernel_begin_dav_2201', 'sk_placeholder_kernel_dav_2201', 'sk_placeholder_kernel_dav_2201', 'aclnnSerialMatmulV3_MatMulV3Common_MatMulV3', 'XMlaNormCache', 'aclnnSerialMatmulV3_MatMulV3Common_MatMulV3', 'XMlaSplitRope', 'aclnnSerialMatmulV3_BatchMatMulV3Nd_BatchMatMulV3', 'XMlaLayout', 'sk_placeholder_kernel_dav_2201', 'sk_placeholder_kernel_dav_2201', 'sk_scope_kernel_end_dav_2201']`。
  同一 trace 的独立阶段总计：`{'matrix_us': 568.571, 'vector_us': 203.065, 'matrix_percent_of_C_plus_V': 73.6838353835228, 'vector_by_op_us': {'XMlaLayout': 76.542, 'XMlaSplitRope': 42.821, 'XMlaNormCache': 83.702}}`。这不是 Graph Event 延迟占比，SK 内部阶段未用减法推算。
- `torch-matmul-control-b32` B=32, S=1: 6 个设备任务，SK 3，整流程分组=False；任务名 `['aclnnMatmulWeightNz_MatMulCommon_MatMulV2', 'sk_5_mla_prolog_start_static_kernel_XMlaNormCache_693df6a3d73f21574bf062ec59a60f33e730740995302b5e2e292a1d48f7c775_57999_d0__kernel0_end_static_kernel_XMlaNormCache_693df6a3d73f21574bf062ec59a60f33e730740995302b5e2e292a1d48f7c775_57999_d0__kernel0', 'aclnnMatmulWeightNz_MatMulCommon_MatMulV2', 'sk_6_mla_prolog_start_static_kernel_XMlaSplitRope_9d68d9d64fff40e5607aed5daa6f179ef662063ef85e1f1bb8b77f2c0acbe84c_57999_d0__kernel0_end_static_kernel_XMlaSplitRope_9d68d9d64fff40e5607aed5daa6f179ef662063ef85e1f1bb8b77f2c0acbe84c_57999_d0__kernel0', 'aclnnMatmul_BatchMatMulNd_BatchMatMulV2', 'sk_7_mla_prolog_start_static_kernel_XMlaLayout_f1a4b1312a1616d98f517c7bdcbf210f7d3aebf247ac0082773d162423121d00_57999_d0__kernel0_end_static_kernel_XMlaLayout_f1a4b1312a1616d98f517c7bdcbf210f7d3aebf247ac0082773d162423121d00_57999_d0__kernel0']`。

B=32,S=1 的 V3 NZ S2 trace 只有一个设备任务，名称从 MatMulV3 开始、XMlaLayout 结束；其三次矩阵和三个 Vector 均在该任务内。B=1,S=1 的 trace 同样为一个任务，但单行布局无需执行，因此该 case 只证明三矩阵加两个 Vector。两组均通过同地址输入/权重/cache index 更新与 cache 别名检查。

公共 torch.matmul 对照（`torch_matmul_nz`，B=32,S=1）：状态PASS，候选74.80100095272064μs，设备任务6，整流程分组=False。此结果仅适用于本次 Torch/Torch-NPU/CANN 与接口路径；实际调用、静态编译和设备任务证据见该目录的原始文件。

本次 torch.matmul 对照执行与精度通过，但整流程分组未通过：trace 为3个独立矩阵任务（两次 MatMulV2、一次 BatchMatMulV2）和3个单 Vector SK 任务，交替串行执行。矩阵编译 JSON 未提供 supportSuperKernel/sub_operator_kernel_name，Vector 提供分组字段。这一失败是未达到整体分组要求，运行没有抛异常；不虚构异常信息。保留可整体分组的 L0 V3 NZ 为主路径。

可拆分阶段的 B=64,S=32 S1 trace：矩阵568.571μs（C+V的73.68%），三个 Vector 合计203.065μs。Norm83.702μs（10.85%）、Split42.821μs（5.55%）、Layout76.542μs（9.92%）。在这组可分阶段证据下，继续只改 Vector 的收益空间小于继续改善矩阵阶段的空间；Vector 内优先级为 Norm 与 Layout。该结论不替代 S2 内部阶段的直接测量。

最终组合在相同18组 token 行数上的独立 Vector 几何均值：norm=1.0624×, split=1.4444×, layout=1.2232×。保留的回退点为：{'norm': [{'rows': 256, 'mode': 20, 'speedup': 0.9778396874688592}], 'layout': [{'rows': 33, 'mode': 12, 'speedup': 0.920295139048413}, {'rows': 64, 'mode': 12, 'speedup': 0.9905800560849208}, {'rows': 2048, 'mode': 12, 'speedup': 0.9427334853652474}]}。没有为了改善均值过滤回退 shape。

## 与融合基线的源码量比较

本版生产源码 3384 行，固定融合基线源码 9785 行，绝对变化 -6401 行，百分比 -65.42%。计数来自实际测试的 Git commit。基线沿用已审计 9.0.0-beta.1 源码快照的30个 host/kernel C++文件，9785物理行；本版按相同生产文件清单统计39个 C++/头文件及 Python 编排/编译/加载文件，保留旧兼容代码。测试、oracle、CMake/构建脚本、文档、生成代码、外部 CANN/Torch 矩阵实现不计入；基线含其多架构/分支，本版是 BF16 子域，因此这是源码体量对比。逐文件绝对计数见每轮 source-size.json。

## 源码与复现

最终公开提交：[vllm-ascend 308781e61482263d26645f7511634de119338753](https://github.com/AuroraEmiya/vllm-ascend/commit/308781e61482263d26645f7511634de119338753)，目录 `examples/mla_prolog_vector_tuning_20261106/`。本地独立源码：`D:\Workflows\OP_DEV\CANNbot\projects\dev-mla-vector-tuning-20261106\source`。

Vector 生产实现统一在 `csrc/attention/mla_prolog_serial/`，每个 `x_mla_*` 下各有 `op_host/` 与 `op_kernel/`。Python 编排入口 `mla_superkernel/pipeline.py`；SuperKernel 接口 `compiler.py`；外部矩阵 L0 适配在 `csrc/aclnn_torch_adapter/serial_matmul_v3/`。矩阵设备实现由安装的 CANN 提供，本轮没有复制或修改矩阵源码。

从仓库拉取该目录后按 source/README.md 与 BUILD.md 构建即可；本轮每个候选均由本地 Git commit 经 GitHub 再由服务器 fetch。复现不依赖整个 vLLM 仓库。运行和构建容器由本任务隔离管理，资源结束状态另见 resource-release.json。

## 共享资源观察与释放

只读监控自S0中途开始，原始归属不明标志1次，与记录中case运行窗口重叠0次。该标志位于最后一个S1 case结束3.617秒后；相同PID此前属于本任务cgroup，下一采样为空，符合退出阶段cgroup退化现象。原始标志没有删除或改写，核对见 resource-watch/anomaly-audit.json。早期轮次保留各自运行前后快照；监控观察不作为资源锁。

仅停止已核对完整容器ID且无活动测试进程的两个任务容器。释放快照时间2026-11-06T11:30:47.441670+00:00；16芯片健康=True；物理卡10进程=[]；共享docker.sh保持原hash。其他容器及进程未作修改。
