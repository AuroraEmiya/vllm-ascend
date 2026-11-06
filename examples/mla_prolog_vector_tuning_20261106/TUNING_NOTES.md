# 三 Vector 的 tiling 与流水设计

代码来自我们自己的串行版本。本轮只改 Vector host/kernel 以及静态形状策略，矩阵主对照固定 L0 V3 NZ。基线接口是安装的 aclnnMlaProlog，未用融合 MLA 的 tiling/kernel 作为实现来源。

|子算子|输入与工作单元|实际搬运/计算|缓冲生命周期|
|---|---|---|---|
|Norm/cache|下投影 [B×S,2112]；Q1536、KV512、KR64|FP32 RMSNorm、RoPE；BF16 输出；合法独立 cache slot 写入|输出两个 BF16 槽；MTE3_V 仅等即将重用的槽；输入/FP32 工作区另用 V_MTE2、MTE2_V、PIPE_V 约束|
|Split/RoPE|上投影 [B×S,N×192]；row/head tile|NoPE128 写成 [N,B×S,128]，RoPE64 在 Vector 中用步长 Cast 和 even/odd Gather 计算|两个输入/系数/输出槽；MTE3_MTE2 保护每个槽的 NoPE 和 RoPE 输出 DMA|
|Layout|第三矩阵输出 [N,B×S,512]；row tile × head tile|DataCopyPad 的 GM/UB 步长直接得到 [B×S,N,512]，无单独转置 kernel|两个 tile 槽；仅等重用槽，MTE2_MTE3 保护当前输入后再输出|

每个 kernel 返回前消费两个空闲槽事件，避免遗留事件跨静态子 kernel。源码不使用 PIPE_ALL。双缓冲不保证收益：例如 Norm 的输出 DMA overlap 在大输入几乎无收益，而 Split 的双缓冲及连续读取效果明显。

host 从平台对象读取可用 AIV 核数和 UB 容量。所有模式计算工作单元数、有效 tail、实际 head tile、UB 峰值，再设置 blockDim。不是按某个模型名称路由。

- Norm mode20 的工作单元是 Q/KV 两支，mode21 将一 token 两支共同调度并保留两套 FP32 gamma。mode22/23 测半核/四分之一核预算，较大输入回退，未入选。
- Split mode30 保持 headTile≤16；mode31 根据 row/head 数和 AIV 数扩展并行度；mode32 使用同一 head tiling，再合并输入 DMA。小 token 数的细 tile 使系数重复加载次数增加，实测回退，因此默认小输入选择 mode30。
- Layout mode11/12/13/14 分别是 1/2/4/8 行 tile，DB 的 UB 峰值为 4×headTile×rowTile×512 字节。先以独立任务数保持并行，再按 UB 上限收缩 headTile。GM stride 单位为字节，UB stride 单位为32字节块，tail 保留预定行步长。

最终路由由 `MlaPrologSerial.vector_modes` 给出：Norm 在 rows≥512 用21、其余20；Split 在 rows≥32 用32、其余30；Layout 在 rows≤32 用14、其余12。每个 host 继续根据实际平台资源计算具体 tiling。

独立 Vector 测量不能代表完整 MLA。源代码相同的 S0、S1、S2 是完整流程控制；SuperKernel 是否覆盖所有子算子，按真实 kernel_details.csv 和本次编译元数据确认。S2 内部 C/V 时间不从 Event 与 profiler 延迟相减推算。

主网格和轴向边界在 shape-plan.json。每个 B/S case 都有三个实现的 Event 原始样本、CPU FP64 精度检查和已加载二进制 SHA256。较大 case 的 CPU oracle 为指定 token 抽样，全输出仍与 aclnnMlaProlog 比较，未触碰 cache 全量精确检查。失败编译、回退 shape、分组失败和资源冲突均保留，不能补零或静默过滤。

权重合并和 NZ 准备只发生在初始化与显式 refresh。BF16 矩阵输入不会先转换为 FP32。`view` 只描述连续数据的 B/S 元数据，实际布局转换由 Vector 内部寻址完成。
