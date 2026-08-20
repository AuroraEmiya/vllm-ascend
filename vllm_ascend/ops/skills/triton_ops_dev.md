# 从 vLLM-Ascend Triton 算子提炼的 NPU 开发技巧

## 1. 适用范围与结论等级

本文件把 `vllm_ascend/ops/triton/` 的实现模式转化为后续 NPU Triton 算子开发规则。它不是硬件规格书，也不把某个现有 tile 或 launch 参数当作通用最优值。

- **Observed**：可由主分支源码直接确认。
- **Recommended**：对后续 NPU 算子具有可执行价值，但仍需在目标硬件、CANN 与模型 shape 上验证。
- **Unknown**：必须由 benchmark、profile、IR 或硬件文档解决。

本项目的首要约束是：目标不是 CUDA/GPU，而是 NPU Triton-Ascend。所有性能结论都应区分 VectorCore 路径、AICore/Cube 路径和 GM/UB 数据移动路径。

---

## 2. 总体方法：先确定计算域，再选择实现

在设计 kernel 前，对每个张量、每个 `pid`、每条 load/store 写出以下合同：

```text
逻辑对象：token / row / head / page / block / expert / request / chunk / tile
输入布局：shape、stride、dtype、物理轴含义
输出布局：shape、stride、是否允许原地写
有效域：哪些 lane、token、head、block 在当前 program 中有效
并发关系：是否有多个 program 写同一位置；若没有，如何保证
数值合同：padding、空域、NaN、Inf、sentinel、tie-break、因果边界
硬件路径：VectorCore / Cube(AICore) / 混合；预期的 GM↔片上数据流
```

这一步优先于 tile 调参。文件名、变量名和现有 CUDA 写法都不能替代当前 kernel 的地址与有效域推导。

---

## 3. 技巧一：把 Ascend 能力探测集中化，不要散落硬编码

### 源码模式

`triton_utils.py`：

- 初始化时读取 `num_aicore` 和 `num_vectorcore`；
- 从 `triton.language.extra.cann.extension` 解析 `insert_slice`、`extract_slice`、`get_element`；
- 若扩展不存在，回退到 `triton.language`；
- 不能解析时显式报错，而非静默换成错误实现。

### 可执行规则

1. 在 package 初始化或 model runner 初始化阶段执行一次 device property 初始化。
2. kernel wrapper 只能通过统一接口获取 core 数，不在各文件写固定数字。
3. 需要 CANN extension primitive 的路径必须 fail-fast；不能用“看起来相近”的 Python 或 Triton 表达式无验证替换。
4. 以 `get_vectorcore_num()` 做分片时，先确认当前算子属于向量型工作负载；不要把它用于估计 Cube 吞吐。

### 反例

```python
# 错误：把设备 SKU、AICore 数或 VectorCore 数硬编码进 kernel wrapper。
grid = (80,)
```

```python
# 错误：extension 不存在时静默跳过 slice/get_element 逻辑。
try:
    op = extension_op
except Exception:
    op = lambda x: x
```

### Unknown

`num_aicore` 与 `num_vectorcore` 对某个 Triton launch 的具体调度比例不是公开的等式。它们是启动分片的候选信息，不是无需 profiling 的最优 grid。

---

## 4. 技巧二：动态工作量按真实范围分片，而不是按静态最大长度分片

### 源码模式

`activation/swiglu_quant.py` 使用 group-list 推出真实 `total_rows`，然后：

```text
block_size = ceil(total_rows / NUM_CORES)
program pid 处理 [pid * block_size, min((pid + 1) * block_size, total_rows))
```

### 推荐做法

- 对 MoE、ragged batch、变长 decode、分页 KV、chunked SSM，优先在设备端或 host 端得到**真实工作范围**。
- 一个 program 负责连续区间，保证写地址连续，减少地址表访问次数。
- `pid` 越界时直接 return；尾块用 mask。
- 真实工作量很小时，避免为“理论最大 core 数”强行构造大量无效 program。

### 性能风险

连续 range 均分按元素数而非计算量均分。以下场景可能失衡：

- expert 的 token 数极不均匀；
- 某些 row 的分支、量化、page table 或 cache miss 远重于其他 row；
- chunk 末尾有不同状态更新量。

因此应记录每个 program 的实际有效工作量，至少比较：连续切分、按 expert 切分、按 page/chunk 切分。

---

## 5. 技巧三：把 NPU 对齐视为 ABI 合同，不是“可选微优化”

### 源码观察

SwiGLU quant 代码明确说明 group-list 在 NPU 上需要 32-byte UB 对齐，并根据元素类型把 expert 数补到：

| dtype | 对齐后的元素倍数 | 原因 |
|---|---:|---|
| `int64` | 8 | `8 × 8 B = 32 B` |
| `int32` | 16 | `16 × 4 B = 32 B` |

### 可执行规则

1. 对所有会整体向量读取的 metadata 写明最小 alignment、padding 元素和值。
2. padding 元素必须在逻辑 mask 内无效，且其值不能污染 reduction、地址或索引。
3. 调用端负责构造对齐 buffer，kernel 负责 mask 实际长度；不要把对齐假设只写在 kernel 注释里。
4. 对 page table、block table、expert offsets、token offsets 等 integer metadata 单独压测。它们常是小数据，但会触发大量随机访问或动态控制。

### 不可推广的部分

“32 bytes”是已有 group-list 路径的明确合同，不可直接推导为所有 GM、L2、L1、L0、UB 访问的统一最优值。

---

## 6. 技巧四：指针类型、地址向量和复制循环必须按后端稳定写法组织

### 源码模式

`batch_memcpy.py`：

1. `pid` 读取一组源/目的 pointer 和 size；
2. 在循环外把**标量 pointer** 转成 `uint8*`；
3. 循环内对该 pointer 增加向量 offset；
4. load/store 使用尾块 mask。

### 推荐规则

```text
先得到标量基址
→ 转换 pointer type
→ 生成 arange offset
→ 基址 + offset 得到地址向量
→ masked load/store
```

避免：

```text
先创建一组整数地址
→ 把整数地址向量整体 cast 成 pointer 向量
```

后者在标准 Triton 语义中可能看似等价，但已有 vLLM-Ascend 修复记录表明 Triton-Ascend 的 pointer offset/axis 分析可能无法稳定处理这种形式。

### 验证项

- size 为 0、1、`BLOCK_SIZE-1`、`BLOCK_SIZE`、`BLOCK_SIZE+1`；
- source/destination 对齐与非对齐；
- 多个 pointer 对应不同 size；
- 地址数量、字节偏移与 index 位宽上界；
- warmup、graph replay、buffer reuse。

---

## 7. 技巧五：将“形状专用”和“运行时长度”严格分开

### 源码模式

- `bincount.py` 采用 `do_not_specialize=["batch_size", "seq_len"]`。
- `layernorm_gated.py` 的 `HAS_BIAS`、`HAS_Z`、`NORM_BEFORE_GATE`、`IS_RMS_NORM` 作为 `tl.constexpr` 控制计算图形状。
- `swiglu_quant.py` 将列数、expert 数、group-list 类型、scale 开关写为 `tl.constexpr`。

### 判断表

| 参数类别 | 优先处理方式 | 原因 |
|---|---|---|
| 改变地址公式/矩阵形状/代码控制图的离散开关 | `tl.constexpr` | 生成可优化的明确变体 |
| 高频变化的 batch size、实际 seq_len、token 数 | 普通 runtime 参数 + `do_not_specialize` | 避免编译 cache 爆炸 |
| 仅影响 grid/buffer 大小的 host 数值 | wrapper 中计算 | 让 kernel 内部保持稳定 |
| 可能改变性能但不改变数学语义的调度项 | 独立 benchmark 搜索 | 不与算法修改耦合 |

### 反例

- 每个请求长度都作为 `tl.constexpr`：会产生大量编译实例。
- 所有参数都 runtime 化：可能使 `tl.dot` tile、布局和循环无法静态优化。

正确策略是按**是否改变静态 IR 结构**区分，而不是按参数名字区分。

---

## 8. 技巧六：优先融合 producer→consumer 的 layout 转换，但不融合不相干的阶段

### 源码实例

- `linearnorm/*`：QKV split、RMSNorm、RoPE/MRoPE 与 TP 变体。
- `activation/swiglu_quant.py`：SwiGLU 后直接 per-row 量化和写 scale。
- `layernorm_gated.py`：归一化、bias、gate 融合。
- `fla/fused_qkvzba_split_reshape.py`：多段投影结果的 split/reshape 融合。

### 推荐准则

适合融合：

```text
同一元素的 producer→consumer
同一行归约后的直接 consumer
只增加寄存器级中间结果、不要求跨 program 通信的布局重排
```

不应强行融合：

```text
需要 global barrier 或跨 program reduce 的阶段
需要写出并在后续多次复用的大中间结果
会让 live tensor 数量显著增大、导致 UB/寄存器压力不可控的阶段
```

### 验证方式

1. 先保留 unfused reference；
2. 对相同 input 的 fused/unfused 输出逐元素比对；
3. 分别测 compute、GM traffic、kernel count、占用或资源报错；
4. 不用端到端加速直接归因于某个 fused expression。

---

## 9. 技巧七：归约、softmax 与状态更新必须定义空域语义

这是高风险通用规则，尤其影响 attention、top-k、chunk state、sampling 和 ragged batch。

### 必须先回答

```text
当前 row/chunk/block 是否可能没有任何有效 lane？
mask 为 false 时 load 的 other 值是否会进入 max/sum/exp/log/div？
是否存在 all -inf 的 max？
是否会发生 -inf - (-inf)、0 * NaN、0 * Inf？
全部 partial 都为空时输出是什么？
```

### 安全模板

```text
识别 has_valid
→ 在 max / exp / divide 之前构造有限的 safe operand
→ 空块不更新 online-state，或写确定的 zero accumulator + finite sentinel
→ merge 前先保证 accumulator 本身有限，再给权重
```

### 不能做

```text
dangerous = exp(score - max_score)   # score/max_score 已可能非法
out = dangerous * mask               # 0 * NaN 仍是 NaN
```

### 适用源码区域

- `batch_invariant/softmax.py`
- FLA/KDA 的 cumsum/state/chunk output
- `reject_sample.py` 的概率路径
- 后续实现的 sparse attention / split-K merge

空域的数学合同应先于性能优化固定，并形成独立回归测试。

---

## 10. 技巧八：ragged 序列先生成 metadata，再执行主计算

### 源码模式

`gdn_chunk_meta.py` 分别生成：

```text
cu_seqlens → chunk_counts → chunk_offsets → final_chunk_indices
```

### 推荐架构

```text
request metadata
  → 纯 metadata kernel / host precompute
  → 固定布局的 workspace
  → 主计算 kernel
  → 必要的 state/output merge
```

### 好处

- 主 kernel 的 program mapping 可更简单；
- 计算与 metadata 错误可拆开测试；
- block/chunk/page 地址能单独断言；
- 减少每次主循环重复做 cdiv、prefix、分支判断。

### 注意

metadata 并非免费。针对短序列/小 batch，多个 metadata launch、workspace 分配和读取可能抵消收益。要同时测完整 API 和主 kernel。

---

## 11. 技巧九：把递归长序列算法拆成阶段，但明确状态所有权

### 源码结构

`fla/*` 和 `kda/*` 都采用类似的阶段化组织：

```text
chunk / cumsum
→ state delta
→ state update 或 triangular solve
→ chunk output
→ output update/merge
```

### 推荐规则

1. 为每个中间 tensor 写出 shape、layout、producer、consumer、生命周期和是否可复用。
2. 任何 state buffer 都要写清“哪个 chunk / program 负责写、哪个阶段负责读”。
3. 跨 chunk 的递归依赖不应伪装成无依赖 elementwise kernel。
4. 可并行的 chunk 维与必须串行的 state 维要在设计阶段分离。
5. 多阶段输出如果需要 merge，先定义其数值归一化与空 partial 合同。

### 性能账本

对每阶段记录：

```text
FLOPs / tl.dot 次数
GM read/write 字节数
临时 workspace 字节数
向量/矩阵 tile
静态循环次数
是否跨阶段重新读取同一数据
是否有串行依赖
```

这比只看源码中的 Big-O 更可用于 NPU 调优。

---

## 12. 技巧十：Cube / `tl.dot` 方案要独立于 VectorCore 方案验证

### 为什么

FLA/KDA、matmul 和某些 attention 子阶段会触发矩阵乘路径；RoPE、norm、sampling、metadata 和 memcpy 则主要是向量/访存路径。两类路径的瓶颈、资源和最优 tile 往往不同。

### 推荐实验顺序

1. 固定数学等价的 reference；
2. 只替换 `tl.dot` 周边的 layout 或 tile；
3. 保持 dtype、输入、mask、输出 layout 不变；
4. 记录 compile 是否通过、资源报错、数值误差、单 kernel 时延；
5. 再测试 end-to-end。

### 禁止推断

- 不能从 VectorCore 数量推出 Cube 最佳 grid；
- 不能从一个 `tl.dot` case 通过推出所有 `head_dim`、dtype、stride 和 page layout 都安全；
- 不能把 GPU warp 或 CUDA shared-memory 的直觉直接套到 NPU 的 L0/L1/UB/MTE 管线。

---

## 13. 推荐的 NPU 算子开发流程

### 阶段 A：冻结合同

```text
目标 API、输入/输出、dtype、layout、stride
有效范围与 padding
是否 exact、容差与 reference
NPU 型号、CANN、torch-npu、triton-ascend、vLLM 版本
测量边界：完整 API / 单 kernel / metadata / merge
```

### 阶段 B：建立最小可验证版本

- 一个逻辑输出单元对应一个清晰的 program mapping；
- 所有 load/store 均有地址解释和 tail mask；
- 先实现 reference 可覆盖的基础路径；
- 明确空域、NaN/Inf、边界和 sentinel。

### 阶段 C：确定数据流和存储策略

```text
GM → [MTE?] → UB/L1/L0 → Vector/Cube → 输出
```

此图必须在你有硬件资料、IR 或 profile 后填写；没有证据时保留为 Unknown。

### 阶段 D：逐项优化

优先顺序：

1. 移除不必要的 GM round trip；
2. 融合同元素 producer→consumer；
3. 让 metadata 对齐、连续、可复用；
4. 降低无效 lane 与重复 reduction；
5. 调整 VectorCore/Cube 分片和 tile；
6. 再调整 backend-specific launch 配置。

一次只变更一个主因素，并保留旧版 benchmark 与 correctness case。

---

## 14. 正确性与调试检查表

### 基础边界

- 最小合法 shape；
- tile/chunk/page 边界；
- 边界前一格、边界、边界后一格；
- 非整除尾块；
- batch=1 与多 batch；
- 多 head / GQA / TP 对应关系；
- 动态长度不同的请求；
- 0-length 或 padded rows；
- 非连续 stride、非零 page、可辨识的 page 内容；
- BF16/FP16/FP8（若接口支持）。

### 数值

- finite 计数在误差统计前检查；
- reference mean/max error；
- 最大误差索引与局部元数据；
- all-empty、partially-empty、first-empty-later-valid；
- NaN/Inf 输入是否属于合同内；
- stable tie-break、sentinel 与无效 index 的输出值。

### 执行模式

- 当前 eager；
- fresh serving；
- graph capture/replay；
- warmup 后重复；
- buffer reuse / 多 stream（若适用）。

历史 dump 只能解释过去现象，不能代替 fresh serving 验收。

---

## 15. Benchmark 与 profile 规则

每个结果必须注明：

```text
设备与软件版本
输入 shape、dtype、layout、stride
被测窗口：完整 API / 单 kernel / metadata / merge
warmup 和重复次数
是否包含输入构造、临时 buffer 分配、同步、首次编译
median、mean、p10/p90
正确性验证是否在计时窗口外完成
```

建议比较三层：

| 层级 | 目的 |
|---|---|
| 单 kernel | 判断 tile、访存、资源、codegen 的直接影响 |
| 算子子流水 | 判断 metadata/workspace/merge 是否吞噬收益 |
| 完整 serving 路径 | 判断 launch、graph、buffer 生命周期和实际收益 |

端到端变快不能单独证明某个 helper 或某个 `tl.dot` 是唯一收益来源。

---

## 16. 当前项目的默认约束

后续在本项目中设计 NPU Triton 算子时，默认采用下列约束：

1. 先通过统一工具读取 NPU 的 AICore/VectorCore 能力；不写死 core 数。
2. 先确认目标是 Vector、Cube 还是混合路径；性能假设必须对应这一分类。
3. 对 metadata 约定 dtype、对齐、padding 值与有效长度；让调用端与 kernel 共享同一 ABI。
4. 对 pointer 算法使用“标量基址转换后再做向量 offset”的写法。
5. 对动态长度优先 runtime 参数与稳定 grid；只将离散计算图开关设为 `tl.constexpr`。
6. 融合只针对相邻 producer→consumer；跨 program reduce/state 先保留阶段边界。
7. 所有 reduction、softmax、split-K、chunk state 都先定义 empty-domain 行为。
8. 性能结论必须来自目标 NPU 的 benchmark/profile，不从 GPU 或其他 NPU 的直觉外推。

---

## 17. 源码依据

- 主目录：<https://github.com/vllm-project/vllm-ascend/tree/main/vllm_ascend/ops/triton>
- Ascend Triton helper：<https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/triton_utils.py>
- SwiGLU quant：<https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/activation/swiglu_quant.py>
- 批量 memcpy：<https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/batch_memcpy.py>
- GDN chunk metadata：<https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/gdn_chunk_meta.py>
- FLA：<https://github.com/vllm-project/vllm-ascend/tree/main/vllm_ascend/ops/triton/fla>
- KDA：<https://github.com/vllm-project/vllm-ascend/tree/main/vllm_ascend/ops/triton/kda>
