# MiniMax-M3 Index Score 提取、测试与 A3 优化分析

## 1. 任务边界

### Baseline

当前 baseline 是本次上传的 `Pasted text.txt`。

本次只保留：

```text
prefill index score
decode index score
```

本次不包含：

```text
top-k 选择
prefill score finalize
invalid top-k index mask
sparse attention
主 K/V cache 处理
vLLM custom-op 注册
模型或 serving 调度
CUDA/ROCm/PDL 分支
```

### 目标设备

目标是 Ascend A3。项目中的 `triton_ascend_decode_score_practice_skill.md` 和
`decode_score_optimization_report_20260721.md` 来自另一套 A5/32-AIC 环境，本文只把其中的方法作为候选实验，不把具体 program 数、chunk 数、瓶颈或收益写成 A3 事实。

结论等级：

- **Observed**：可由当前 baseline 源码直接确认。
- **Inferred**：由源码结构推导出的性能假设，需要 A3 benchmark/profile 验证。
- **Unknown**：当前缺少 A3 编译、运行或 profile 证据。

---

## 2. 产物

```text
index_score_a3_baseline.py
    从 baseline 提取的 standalone score 模块

bench_index_score_a3.py
    单/多实现、单/多 case、prefill/decode 通用测试脚本

index_score_extraction_a3_analysis.md
    计算流程、提取边界和 A3 优化通道分析
```

当前环境没有 Triton、torch_npu 或可用 NPU，因此只完成了：

```text
Python 语法检查
case 注册与选择检查
CPU case 构造检查
独立 PyTorch reference 生成检查
```

尚未完成：

```text
Ascend Triton 编译
NPU correctness
A3 性能测试
A3 msprof/simulator 分析
```

---

# 3. Triton 算子提取方法

## 3.1 先确定真正的调用链

### Prefill

当前调用链：

```text
minimax_m3_index_score
    → normalize index_kv_cache
    → allocate score
    → head_dim == 1 ?
        ├─ _prefill_scalar_key_extrema_kernel
        └─ _prefill_scalar_index_score_kernel
      :
        └─ _prefill_index_score_kernel
    → return score

后续 top-k 路径（不属于本次提取）：
    minimax_m3_index_topk
    → force init/local scores
    → fill invalid tail
    → torch.topk
    → mask invalid indices
```

### Decode

当前调用链：

```text
minimax_m3_index_decode
    → normalize index_kv_cache
    → allocate score
    → allocate init_mask/local_mask
    → _prepare_decode_score_masks_kernel
    → _decode_index_score_kernel
    → _fill_decode_score_tail_kernel
    → torch.topk
    → mask invalid indices
```

本次提取到 `_fill_decode_score_tail_kernel` 为止。`torch.topk` 和 index mask 不属于 score 计算。

## 3.2 依赖分类

### 必需计算依赖

Standalone 文件只保留：

```python
import torch
import triton
import triton.language as tl
```

### 本地化 helper

原实现：

```python
from vllm.utils.math_utils import round_up
```

提取后：

```python
def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment
```

### 删除的框架胶水

```text
vllm.platforms.current_platform
vllm.triton_utils
custom-op registration
sparse attention wrapper
top-k wrapper
ROCm num_stages workaround
CUDA PDL launch
```

当前 NPU baseline 的 PDL 路径为 false，因此 standalone NPU 文件删除 PDL 参数和 CUDA intrinsic，不改变当前 NPU 数学结果。

## 3.3 保留的 ABI

### 公共输入

```text
idx_q:
    [total_query_tokens, index_head_count, head_dim]

index_k_cache:
    [num_physical_pages, 128, head_dim]

block_table:
    [request_count, max_logical_blocks]

block_table[request_id, logical_block_id] = physical_page_id
```

一个 logical sparse block 固定对应 128 个 index-K token。

### Score 输出

```text
score:
    [index_head_count, total_query_tokens, score_block_stride]
    dtype = float32

score_block_stride:
    round_up(ceil(max_seq_len / 128), 16)
```

### Output reuse

两个 standalone API 都支持：

```python
out=preallocated_score
workspace=preallocated_workspace
```

这样 benchmark 可以排除 tensor allocation，同时仍然计入实际 Triton kernel 和 wrapper launch。

---

# 4. 数学语义

对 query 向量 `q` 和 logical block `b`：

```text
page = block_table[request, b]
K_b = index_k_cache[page, :, :]
```

基本 score：

\[
S(q,b)=\max_{0\le j<128}(q\cdot K_{b,j})
\]

只有同时满足 sequence 和 causal 可见性的 token 才能参与 max。

`sm_scale` 在当前 score-only baseline 中不参与计算。正的全局 scale 不改变 block 排序，这是当前代码的明确行为；如果未来 score 被其他路径按数值而非排序消费，需要重新审计该合同。

---

# 5. Prefill score 实现分析

## 5.1 Launch 映射

Observed：

```text
grid[0] = ceil(max_query_len / 96)
grid[1] = batch_size × index_head_count
```

一个 program 负责：

```text
一个 request
一个 index head
最多 96 个连续 query token
该 query tile 能看到的全部 logical K blocks
```

## 5.2 数据流

```text
load request metadata
    ↓
load Q tile [BLOCK_SIZE_Q, head_dim]
    ↓
for each visible logical block:
    load page_id
    load K page [head_dim, 128]
    tl.dot(Q, K)
    row-wise max over 128 positions
    store [BLOCK_SIZE_Q] scores
```

## 5.3 Full-visible 与 boundary

Observed：kernel 已经把 block 分成两类。

### Full-visible historical block

条件：

```text
该 block 对 tile 内最早 query 已完全可见
并且该 block 是完整 sequence block
```

执行：

```text
unmasked K load
tl.dot
row max
store
```

### Boundary block

可能同时与 causal 边界或 sequence 尾部相交。

执行：

```text
masked K load
tl.dot
causal/sequence tl.where(-inf)
row max
store
```

当 prefix 不对齐时，一个 query tile 可能涉及两个 boundary blocks。

## 5.4 `head_dim == 1` 特化

Observed：

完整可见 page 先计算：

```text
page_min
page_max
```

随后根据 query 标量正负选择：

```text
q >= 0 → q × page_max
q < 0  → q × page_min
```

这避免对每个 query tile 重复扫描 128 个 K token。boundary page 仍走逐 token 精确路径。

## 5.5 Prefill 当前成本结构

Observed：

- Q tile 在 program 内复用全部 visible blocks。
- K page 每个 query tile、每个 index head 都会重新加载。
- 一个 program 顺序遍历它负责的所有 blocks。
- block 维度没有进入 grid。
- index-K cache 没有 head 轴，同一 K page 被多个 index heads 共用，但当前 grid 按 head 分 program。

Inferred：

- 长 context、较小 batch/head/query-grid 时，单 program 长 block loop 可能限制并行度。
- 增加 block-chunk grid 可提高 program 数，但会重复加载 Q tile。
- 合并多个 index heads 可复用 K page，但会增加 Q tile、dot 输出和片上资源压力。

Unknown：

- A3 上当前 prefill 是 MTE2-bound、Cube-bound、Vector max-bound，还是 program 并行度不足。
- `BLOCK_SIZE_Q=96` 在 A3 上是否接近最优。
- block-table/page 的物理连续性和 L2 命中情况。

---

# 6. Decode score 实现分析

## 6.1 阶段一：init/local mask

Observed：

```text
_prepare_decode_score_masks_kernel
```

为每个 flattened query token 和 logical block 写两个 bool workspace：

```text
init_mask[query, block]
local_mask[query, block]
```

优先级：

```text
normal score
→ init: 1e30
→ local: 1e29
```

代码中 local `tl.where` 位于最外层，因此 init/local 重叠时 local 的 `1e29` 覆盖 init 的 `1e30`。

## 6.2 阶段二：split-K score

Observed：

```text
grid = (request_count, num_kv_chunks)
```

`num_kv_chunks` 来自 Triton autotune：

```text
1, 2, 4, 8, 16, 32, 64, 128, 256
num_stages = 1 or 2
```

prune 规则把总 program budget 限制在基于 512 的范围内。这个数是 baseline 策略，不是 A3 硬件事实。

一个 program：

```text
负责一个 request 的一个 block chunk
一次加载所有 index heads × decode query lanes 的 Q tile
顺序遍历 chunk 内 blocks
每个 K page 只加载一次并服务全部 HQ rows
```

QK tile：

```text
Q: [index_heads × BLOCK_SIZE_Q, head_dim]
K: [head_dim, 128]
QK: [index_heads × BLOCK_SIZE_Q, 128]
```

随后对 128 个 K token做 row max。

## 6.3 Causal/sequence 处理

Observed：

对每个 block 都构造：

```text
pos = block_id × 128 + token_offset
pos_mask = pos < kv_len(query_lane)
```

即使是完全可见历史 block，也执行 `tl.where(..., -inf)`。

## 6.4 Forced score

Observed：

当前顺序始终是：

```text
load K
QK dot
max
load init/local mask
用 sentinel 覆盖 score
```

因此最终被 init/local 常量覆盖的 block 仍支付 K load 和 dot 成本。

## 6.5 阶段三：tail fill

Observed：

```text
_fill_decode_score_tail_kernel
```

把每个 query 的：

```text
[row_num_blocks, max_block_count)
```

写为 `-inf`。

对齐产生的：

```text
[max_block_count, score_block_stride)
```

不属于后续 `torch.topk` 的读取范围，baseline 不初始化该 padding。

## 6.6 Decode 当前成本结构

Observed：

- score 主 kernel 已复用 K page 到全部 index heads/query lanes。
- mask 预处理产生两个 bool workspace 和额外 GM 写入。
- score 主 kernel读取两个 bool workspace。
- tail 是独立 kernel。
- chunk 使用 ceil 切分，chunks 大于实际 blocks 时会产生空 program。
- autotune key 不包含 runtime `seq_len`，有利于固定 shape/graph，但 ragged workload 的最佳配置可能不同。

Inferred：

- mask workspace 和 tail kernel 可能形成可消除的 launch/GM 开销。
- 长历史区间可分成 full-visible 热路径和一个 boundary block，减少每 block mask 运算。
- forced block 跳过 K load/dot 可能减少搬运，但只有在 forced block 占比或 K load 成本足够高时才可见。
- program/chunk 数影响并行覆盖与每 program 连续 K 流长度，A3 最优点需要重新扫。

Unknown：

- A3 上主瓶颈是否与 A5 一样是 MTE2。
- A3 的 AIC 数、program 调度关系、`num_stages` lowering 和 UB/L1/L0 压力。
- mask/tail 融合后节省是否会被主 K-cache 成本完全覆盖。

---

# 7. 测试脚本设计

## 7.1 支持矩阵

| 能力 | 参数 |
|---|---|
| 单实现文件 | `--files baseline.py` |
| 多实现文件 | `--files baseline.py opt1.py opt2.py` |
| Prefill | `--mode prefill` |
| Decode | `--mode decode` |
| 两者 | `--mode both` |
| 单 case | `--case CASE_NAME` |
| 多 case | 重复 `--case` 或 `--cases a,b,c` |
| 全部 case | `--all-cases` |
| 正确性 | `--validate` / `--no-validate` |
| 性能 | `--benchmark` / `--no-benchmark` |
| 平均计时 | `--warmup --iters --repeats` |
| JSON | `--json-out result.json` |

## 7.2 已注册 case

### Prefill

```text
prefill_aligned_small
prefill_unaligned_ragged
prefill_permuted_pages
prefill_scalar_d1
prefill_long_context
```

### Decode

```text
decode_zero_short
decode_aligned_q1
decode_unaligned_ragged_q1
decode_q4_ragged
decode_permuted_pages
decode_init_local_overlap
decode_long_context
```

## 7.3 Correctness reference

Reference 与 Triton 实现独立，使用 CPU FP32：

```text
按 block_table gather page
→ FP32 QK
→ causal/sequence 截断
→ page 内 max
→ decode init/local sentinel
→ decode logical tail -inf
```

验证内容：

```text
finite score atol/rtol
-inf 位置精确一致
定义域不能出现 NaN
不能出现意外 +inf
报告 max_abs/max_rel 和比较元素数
```

Prefill baseline 只定义每个 query 的 causally visible block 域；其余 score/padding 不作为 standalone score 的语义输出。

Decode 定义 `[..., :max_block_count]`；alignment padding 不参与验证。

## 7.4 Timing 边界

```text
构造输入一次
预分配 out/workspace
warmup
synchronize
for repeat:
    synchronize
    start
    for iteration:
        public score API(out=..., workspace=...)
    synchronize
    elapsed / iterations
```

输出：

```text
mean
median
min
max
std
```

计时包含 score 路径内的 Triton kernel launches，不包含：

```text
输入随机生成
out/workspace allocation
reference
JSON 写入
```

---

# 8. 命令示例

## 8.1 查看 case

```bash
python bench_index_score_a3.py --list-cases
```

## 8.2 单文件、单 decode case

```bash
python bench_index_score_a3.py \
  --files index_score_a3_baseline.py \
  --mode decode \
  --case decode_aligned_q1 \
  --warmup 20 \
  --iters 100 \
  --repeats 10
```

## 8.3 多文件、同一个 decode case

```bash
python bench_index_score_a3.py \
  --files index_score_a3_baseline.py score_opt_v1.py score_opt_v2.py \
  --mode decode \
  --case decode_unaligned_ragged_q1 \
  --warmup 20 \
  --iters 100 \
  --repeats 10 \
  --json-out decode_compare.json
```

## 8.4 单文件、多个 prefill case

```bash
python bench_index_score_a3.py \
  --files index_score_a3_baseline.py \
  --mode prefill \
  --case prefill_aligned_small \
  --case prefill_unaligned_ragged \
  --case prefill_permuted_pages
```

等价写法：

```bash
python bench_index_score_a3.py \
  --files index_score_a3_baseline.py \
  --mode prefill \
  --cases prefill_aligned_small,prefill_unaligned_ragged,prefill_permuted_pages
```

## 8.5 多文件、多个 decode/prefill case

```bash
python bench_index_score_a3.py \
  --files index_score_a3_baseline.py score_opt_v1.py \
  --mode both \
  --cases prefill_aligned_small,prefill_long_context,decode_aligned_q1,decode_q4_ragged
```

## 8.6 全 case

长 context reference 成本较高。性能 sweep 建议关闭 reference：

```bash
python bench_index_score_a3.py \
  --files index_score_a3_baseline.py score_opt_v1.py \
  --mode both \
  --all-cases \
  --no-validate \
  --warmup 20 \
  --iters 100 \
  --repeats 10
```

## 8.7 msprof 稳态单 case

```bash
msprof op \
  --output=./prof_decode_aligned_q1 \
  python bench_index_score_a3.py \
    --files index_score_a3_baseline.py \
    --mode decode \
    --case decode_aligned_q1 \
    --no-validate \
    --warmup 20 \
    --iters 100 \
    --repeats 10
```

该命令在同一 profile 进程内执行约 1020 次公共 score API。Decode baseline 每次包含三个 kernel，查看 op summary 时应分别识别 mask、score 和 tail，而不是把随机输入算子混入目标边界。

---

# 9. A3 性能提升通道

以下是候选实验顺序，不是已确认收益。

## 9.1 先建立 A3 事实基线

必须记录：

```text
精确 A3 型号
npu-smi 信息
CANN/driver/firmware
PyTorch/torch_npu
Triton-Ascend 版本
shape/dtype/layout
是否 graph
```

对每个主要 case 获取：

```text
公共 API 平均时间
每个 kernel 时间与 Count
Core Type / Task Type
Block Num
MTE2/MTE1/MAC/Scalar active time
冷态与稳态
```

只有 profile 证明 MTE2 active 接近关键路径，才能把 A5 的 MTE2-bound 经验迁移为当前 A3 结论。

## 9.2 Decode 优先级

### D0：固定可解释基线

保持当前三阶段和数学语义，只关闭复杂 autotune，生成显式 chunk 版本用于 sweep：

```text
chunks = 1, 2, 4, 8, 16, 32, 64
num_stages = 1, 2
```

同时记录总 program 数：

```text
request_count × chunks
```

不要先假设它应等于 AIC 数。

### D1：精确 block 分片

当前 ceil chunk 可能产生空 program。候选：

```text
quotient/remainder 精确均分
每个 program 至多相差一个 block
chunks <= num_blocks
```

对比：

```text
空 program 数
每 program block 数
kernel time
```

### D2：full-visible / boundary 分离

常见 decode_query_len 很小，绝大多数历史 blocks 对所有 query lanes 完全可见。

候选：

```text
Loop A: full-visible blocks
    无 pos vector
    无 causal tl.where

Loop B: 最后一个或少数 boundary blocks
    保留 mask
```

### D3：on-the-fly forced 判定

移除 init/local bool workspace：

```text
在 score kernel 内由 block_id/query_position 计算 is_init/is_local
```

需要比较：

```text
减少一个 mask kernel
减少两个 bool GM 写入和读取
增加主 kernel scalar/integer 运算
```

### D4：forced block 跳过 K load/dot

候选：

```text
如果当前 block 对全部有效 HQ lanes 都是 forced：
    直接 store sentinel
否则：
    计算 QK，并对部分 lane 覆盖 sentinel
```

`decode_query_len > 1` 时不同 lane 的 local 区间可能不同，不能用 request 级简单判断替代 lane 级语义。

### D5：融合 logical tail

让 score owner 覆盖：

```text
[chunk_start, chunk_end) within max_block_count
```

对有效 block 计算，对无效 block 写 `-inf`，删除 tail kernel。

该版本应和“只融合 tail、不改 grid”的版本分开，避免无法归因。

### D6：2-way block interleave

候选循环：

```text
load/compute block 0
load/compute block 1
store block 0
store block 1
```

或后端能够识别的多 buffer 组织。是否形成 MTE2/Cube 重叠必须由 A3 IR/profile 证明。

### D7：K page locality

对比：

```text
contiguous block_table
permuted block_table
局部连续但 request 间分散
```

若性能差异显著，优先研究 page layout/L2 locality，而不是继续增加 program 数。

## 9.3 Prefill 优先级

### P0：BLOCK_SIZE_Q sweep

候选：

```text
32, 64, 96, 128
```

观察：

```text
program 数
Q reuse
K page 重读
片上资源/编译失败
kernel time
```

96 是当前 baseline 常量，不是 A3 最优事实。

### P1：增加 block-chunk grid 维度

候选映射：

```text
grid = (query_tile, batch_head, block_chunk)
```

收益候选：

```text
增加并行 program
缩短每 program 的顺序 block loop
```

代价：

```text
每个 block chunk 重载同一 Q tile
更多 block-table/address 开销
```

重点 sweep 是“Q 重用 vs program 并行度”，不能只看 program 数。

### P2：跨 index head 复用 K page

因为 index-K cache 无 head 轴，可以尝试把多个 heads 合入 Q 行维：

```text
Q: [heads × query_lanes, D]
K: [D, 128]
```

该方式已在 decode 使用，但 prefill 的 `96 × heads` 行数可能显著增加 dot 输出和片上资源。需要分别测试 heads=1/2/4 和不同 BLOCK_SIZE_Q。

### P3：多 query tile/program 内循环

与 block-chunk 相反，候选让一个 program 处理多个较小 query tiles，以复用 metadata/page traversal。适用于 program 数过多或短 context，但可能增加状态和循环长度。

### P4：head_dim==1 extrema 生命周期

当前每次调用对所有 physical pages 重新计算 min/max。

候选问题：

```text
本轮实际引用 pages 占总 pages 比例？
index-K cache 在调用间是否更新？
是否有可靠版本号/失效机制？
```

只有 cache 生命周期明确，才能考虑跨调用缓存 extrema；不能为了性能使用 stale extrema。

## 9.4 跨路径候选

### C1：page ID 批量加载

当前 block loop 每次 scalar load 一个 page ID。可测试小 block tile 的 page ID vector load，但 K pages 本身仍是独立地址，是否改善依赖后端地址生成和 block-table locality。

### C2：dtype 与 accumulation

Decode 显式：

```python
tl.dot(..., out_dtype=tl.float32)
```

Prefill 没有显式 out dtype。任何 dtype/accumulation 修改都必须先证明：

```text
finite score误差
block ranking/top-k 一致性
fp8/bf16/fp16 覆盖
```

### C3：score 与 priority/finalize 融合

Prefill 的 init/local 和 invalid tail 当前在后续 finalize kernel。若把它们移入 score：

```text
可以跳过 forced block QK
可能删除 finalize 部分工作
```

但这已经改变测量边界。必须比较：

```text
原 score + finalize
vs
融合 score
```

不能只拿融合 kernel 与原 score kernel 单独比较。

---

# 10. 推荐实验版本拆分

为保证每次性能变化可归因，建议按单变量版本生成：

```text
index_score_a3_baseline.py

decode_opt_d1_exact_chunks.py
decode_opt_d2_full_boundary.py
decode_opt_d3_inline_priority.py
decode_opt_d4_skip_forced.py
decode_opt_d5_fused_tail.py
decode_opt_d6_interleave2.py

prefill_opt_p0_block_q.py
prefill_opt_p1_block_chunks.py
prefill_opt_p2_fused_heads.py
```

每个文件保持相同公共 API：

```python
prefill_score(..., out=None, workspace=None)
decode_score(..., out=None, workspace=None)
```

先分别验证单项，再组合表现稳定且原因明确的改动。

---

# 11. A3 验收标准

## Correctness

```text
所有注册 case 通过
bf16/fp16
aligned/unaligned
ragged
decode_query_len 1/>1
contiguous/permuted pages
init/local overlap
head_dim==1 prefill
out/workspace reuse
```

## Performance

```text
同一输入
同一 out/workspace 复用
同一 warmup/iters/repeats
多次重复统计
候选运行顺序轮换或分进程复核
```

## Profile

```text
目标 kernel Count 足够大
冷态与稳态分离
不把 overlap ratio 相加
不以 cube_utilization 单独判断瓶颈
不把 program 直接等同于 core
```

最终结论必须写清适用：

```text
A3 精确型号
软件版本
shape/dtype/layout
选择的 case
测量边界
```
