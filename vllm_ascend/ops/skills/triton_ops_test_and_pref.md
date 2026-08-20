---
name: triton-case-first-benchmark-and-validation
version: 2.0
language: zh-CN
scope: all Triton operators, Ascend NPU first
status: project baseline
---

# Triton 算子 Case-First：精度 / 性能 / Simulator 统一 Case Skill

## 0. 目标

任何 Triton 算子在进入功能修改、精度修复或性能优化前，优先建立一个可复用的 case 文件。

case 文件不是临时测试脚本，而是该算子的“实验入口”。底层基础设施可以统一，但每次运行必须先选择唯一主任务：精度、性能或 Simulator。case 文件必须承担：

- 自定义 shape；
- 动态加载任意 kernel 文件与入口；
- 单 case / 多 case；
- 单 kernel / 多 kernel；
- correctness / regression / performance / simulator static case；
- 稳定、可复现的 A/B 或 A/B/C 对比；
- 为 msprof、sanitizer、编译定位提供单 case 入口；
- 以结构化结果记录当前环境、源码版本、shape 与性能统计。

核心原则：**先把 case 做成稳定实验平台，再修改 kernel。**

---

# 1. 两条最高优先级硬约束

## 1.1 Case 文件必须支持自定义 shape 和自定义 kernel 文件

禁止把测试写成只能修改 Python 源码才能换 shape 或换 kernel。

最低要求：

```bash
python test_op.py \
  --kernel-file op_v1.py \
  --kernel-entry op \
  --batch 4 \
  --seq-len 1024 \
  --hidden-size 4096
```

推荐进一步统一成：

```bash
python test_op.py \
  --kernel baseline=op_v1.py:op \
  --kernel candidate=op_v2.py:op \
  --case-json '{"batch":4,"seq_len":1024,"hidden_size":4096}'
```

### Kernel 动态加载要求

- 使用文件路径加载，不要求目标 kernel 文件位于 Python package 内；
- 每个 kernel 使用唯一 module name，避免 Python module cache 导致“改了文件却仍跑旧代码”；
- 输出记录 kernel 文件绝对路径；
- 输出记录文件 hash，推荐 SHA256；
- 入口函数名必须可配置；
- 一个文件包含多个 public API 时，可以指定其中任意一个；
- loader 失败时直接报错，不静默 fallback 到另一个实现。

推荐 kernel 描述：

```text
label:path:entry
```

例如：

```text
origin:/workspace/op.py:run_op
O1:/workspace/op_o1.py:run_op
O2:/workspace/op_o2.py:run_op
```

---

## 1.2 Case 文件必须同时支持单 case / 多 case、单 kernel / 多 kernel

测试框架至少覆盖以下四种模式：

```text
1 case  × 1 kernel
N cases × 1 kernel
1 case  × N kernels
N cases × N kernels
```

其中最重要的是 `N cases × N kernels`，因为它是性能优化和回归判断的标准模式。

### 单 case

用于：

- 编译报错定位；
- UB overflow；
- sanitizer / memcheck；
- msprof；
- 最小精度复现；
- 某一个 shape 的稳定 benchmark。

示例：

```bash
python test_op.py \
  --kernel candidate=op_o2.py:run_op \
  --case decode_b4_q1_s131072
```

### 多 case

用于：

- shape sweep；
- 边界 sweep；
- 功能回归；
- 性能趋势；
- dispatch 阈值定位。

示例：

```bash
python test_op.py \
  --kernel candidate=op_o2.py:run_op \
  --cases short,medium,long,boundary_127,boundary_128,boundary_129
```

### 多 kernel

禁止 benchmark 脚本只写死 `baseline` / `candidate` 两个变量。

底层数据结构应支持：

```python
kernels = [
    KernelSpec("origin", "op.py", "run_op"),
    KernelSpec("O1", "op_o1.py", "run_op"),
    KernelSpec("O2", "op_o2.py", "run_op"),
]
```

这样可以直接比较：

```text
origin vs O1 vs O2 vs safeguard
```

A/B 只是 N-kernel 框架的一个特例。

---

# 2. 每次运行必须先选择唯一主任务

Case 文件可以同时具备完整能力，但**一次运行只允许有一个主关注点**。不要在同一份输出中把精度、性能、Simulator 三类结果混成一个结论。

统一任务参数：

```bash
--task correctness|performance|simulator
```

任务决定：

```text
关注的指标
输入构造方式
是否需要 reference
是否进入 timing loop
输出摘要格式
推荐使用的 shape 资产
```

## 2.1 Correctness：精度问题只聚焦结果对比

当问题是“结果不对、NaN、边界错误、与上游不一致、某个 patch 是否保持精度”时：

```text
主目标：证明输出语义是否正确。
主指标：exact / allclose / finite / mismatch count / max error / worst index。
默认不做性能结论。
```

推荐执行流：

```text
resolve case
→ build deterministic inputs
→ reference / reference kernel
→ target kernel(s)
→ synchronize
→ shape/dtype/finite 检查
→ exact/allclose
→ mismatch / worst-point 报告
```

硬规则：

- correctness 模式默认不打印 p50、speedup 等性能结论；
- 若为了排查超时需要记录 wall time，只标成诊断信息，不作为性能验收；
- reference、CPU copy、assert、error statistics 都不进入任何 timing window；
- 多 kernel correctness 对比时，必须使用同一份逻辑输入；
- 输出优先展示第一个 mismatch、最大误差位置、实际值、期望值与相关 metadata；
- 精度问题必须优先使用业务 shape + boundary shape，不能只跑随机大 shape。

推荐命令：

```bash
python test_op.py \
  --task correctness \
  --kernel origin=op.py:run_op \
  --kernel patch=op_patch.py:run_op \
  --shape-set business \
  --validate exact
```

## 2.2 Performance：性能问题只聚焦时间与趋势

当问题是“哪个实现更快、性能是否回退、dispatch 阈值在哪里、随着 shape 增长如何变化”时：

```text
主目标：获得稳定、可比较的时间数据。
主指标：p50 + p10/p90 + paired speedup / delta。
默认不在每个 repeat 内做精度统计。
```

推荐执行流：

```text
resolve case(s)
→ build fixed inputs
→ 可选 correctness preflight 一次
→ compile/warmup
→ 多 kernel balanced/random order
→ device timing
→ repeats
→ p10/p50/p90 / mean / stability
→ shape trend summary
```

硬规则：

- performance 模式可以在正式计时前做一次轻量 correctness preflight，但不得放进 timing loop；
- reference 不参与 repeat；
- 不输出大段 mismatch/error 统计污染性能报告；
- 多 kernel 必须共享相同输入、相同 warmup/iters/repeats 和 synchronization boundary；
- shape 梯度任务优先使用单轴 sweep，观察趋势、拐点、dispatch、资源边界；
- 最终业务验收必须再跑 business shape，而不能只凭 sweep 曲线；
- 不同机器的绝对时延不能直接用于版本优劣结论。

推荐命令：

```bash
python test_op.py \
  --task performance \
  --kernel origin=op.py:run_op \
  --kernel O1=op_o1.py:run_op \
  --shape-set gradient \
  --sweep-axis seq_len \
  --sweep-values 128,256,512,1024,2048 \
  --warmup 30 --iters 100 --repeats 12 \
  --timing device \
  --order balanced-random
```

## 2.3 Simulator：聚焦“可静态复现的最小 kernel 调用”

Simulator 任务不是 correctness sweep，也不是 performance sweep。它的目标是生成一个**固定、简单、无歧义的 kernel 调用样例**，供后续 simulator、流水建模、编译器或硬件链路复现。

```text
主目标：静态、直接、可重复地调用目标 kernel。
主输入：最小合法 shape + 一个静态业务代表 shape。
主输出：完整 invocation contract + 可持久化输入/metadata。
```

Simulator case 必须满足：

- 直接调用目标 public kernel/wrapper，不绕完整 serving 框架；
- shape 固定，不做 sweep；
- seed 固定，输入生成确定性；
- 所有 metadata 明确打印 shape/dtype/stride/value range；
- 不在调用循环内重新 random；
- 尽量不包含无关 Python 调度、模型对象、custom-op registration；
- 支持将输入、metadata、必要输出 dump 到固定文件；
- 支持单次调用和少量固定 repeat；
- 若算子支持 output/workspace 复用，Simulator case 应显式展示 buffer；
- 静态业务 shape 必须来自真实验收 shape，而不是随意虚构的大 shape。

推荐至少维护两个 Simulator case：

```text
sim_smoke
  → 最小合法、编译快、能证明 kernel 能直接调用。

sim_business_static
  → 固定的代表性业务 shape，用于 simulator 模拟真实流水。
```

推荐命令：

```bash
python test_op.py \
  --task simulator \
  --kernel target=op.py:run_op \
  --case sim_business_static \
  --seed 0 \
  --dump-inputs /tmp/op_sim_case.pt \
  --dump-contract /tmp/op_sim_case.json
```

Simulator 输出 contract 至少记录：

```text
kernel path / entry / sha256
shape
input/output dtype
stride/layout
scalar parameters
constexpr-like launch parameters（从 wrapper 可观察的部分）
seed
input dump path
output/workspace shape
一次直接调用的 Python 入口
```

---

# 3. Shape 必须按三类长期资产管理

以后所有 Triton case 优先把 shape 分成三类；这三类的目的不同，不能互相替代。

建议 Case 增加：

```python
@dataclass(frozen=True)
class Case:
    name: str
    category: str  # gradient | business | simulator
    shape: dict[str, int]
    dtype: str
    seed: int
    params: dict[str, object]
    source: str | None = None
```

## 3.1 Gradient Shape：性能梯度任务

Gradient shape 用于观察一个工作量维度变化时，性能、编译资源、dispatch 或瓶颈如何变化。

典型问题：

```text
seq_len 增大时 latency 怎么变？
batch 从 1 到 64 是否出现拐点？
blocks 在 160→176 是否发生 dispatch 切换？
topk 从 8→64 是否让 reduction 进入另一资源区间？
某个 tile 边界前后是否出现 UB overflow？
```

### 规则：默认一次只扫一个轴

例如：

```text
固定：batch=4, heads=8, head_dim=128, topk=16
只变化：blocks=16,32,64,128,256,512,1024
```

这样结果可以直接形成：

```text
x = blocks
y = p50 latency
```

禁止默认做巨大的 Cartesian product：

```text
batch × seq_len × heads × topk × dtype
```

除非任务明确是 coverage matrix。

Gradient shape 必须记录：

```text
sweep_axis
sweep_value
其他 frozen 参数
分支/阈值说明（若已知）
```

建议命名：

```text
grad_seq_128
grad_seq_256
grad_seq_512

grad_blocks_160
grad_blocks_176
```

主要用于：

```text
performance
threshold finding
compile/resource boundary
profile representative point selection
```

## 3.2 Business Shape：业务验收 shape

Business shape 是真实模型、真实服务或明确需求中关心的 shape，是最终精度和性能验收依据。

它不是“看起来像业务”的 shape，而应有来源：

```text
模型配置
线上 trace
CI case
需求文档
用户明确给出的 shape
已确认的典型/最大 workload
```

每个 business case 推荐记录：

```text
name
shape
source
priority: P0/P1/P2
phase: prefill/decode/train/other
expected dtype/layout
acceptance rule
```

例如：

```python
Case(
    name="biz_decode_b4_q4_blk1024",
    category="business",
    shape={...},
    source="MiniMax-M3 decode acceptance",
    ...,
)
```

Business shape 用于两个最终 gate：

```text
Correctness gate：业务 shape 精度通过。
Performance gate：业务 shape 达到目标或至少无不可接受回退。
```

**Gradient shape 的最优趋势不能替代 Business shape 的验收。**

## 3.3 Simulator Shape：简单调用 + 静态业务输入

Simulator shape 为后续 simulator 模拟流水提供稳定输入。它包含两类：

```text
A. simple invocation shape
   最小合法 shape，目标是简单直接地把 kernel 跑起来。

B. static business shape
   从 business shape 中选择一个或少量代表 shape 固定下来，作为真实流水模拟输入。
```

Simulator shape 与普通业务 benchmark 的区别：

```text
不 sweep
不随机改变 metadata
不追求覆盖所有边界
不依赖模型 runner
输入可 dump、可重放
kernel 调用路径尽量短
```

静态输入推荐记录/导出：

```text
Tensor shape
Tensor dtype
Tensor stride
固定 seed
必要时实际 tensor payload
scalar metadata
block/page/index table
输出/workspace layout
kernel entry
```

若输入很大，可区分：

```text
contract-only：只保存构造规则 + seed + shape + metadata
payload：完整 torch.save / binary dump
```

Simulator case 也可作为 msprof、编译 IR、sanitizer 的稳定单 case 输入，但其首要目的仍是“固定流水复现”，不是性能 sweep。

## 3.4 三类 Shape 与三类 Task 的默认配对

推荐默认关系：

| Task | 第一选择 | 第二选择 | 不建议作为唯一依据 |
|---|---|---|---|
| correctness | business | boundary / adversarial | gradient performance sweep |
| performance | gradient | business | simulator smoke |
| simulator | simulator static | business-static | gradient sweep |

一个成熟算子通常同时拥有：

```text
一组 gradient cases
一组 business cases
一组 simulator cases
```

但一次运行只选择符合当前任务的一组。

---

# 4. Case 必须是“数据定义”，不能散落在执行逻辑里

推荐：

```python
@dataclass(frozen=True)
class Case:
    name: str
    shape: dict[str, int]
    dtype: str
    seed: int
    params: dict[str, object]
```

或者为算子定义强类型 dataclass。

要求：

- shape 参数只在 Case 中定义；
- 输入构造函数只消费 Case；
- reference 只消费 Case + inputs；
- kernel 调用 adapter 只消费 module + Case + inputs；
- benchmark 不自行推导一套隐藏 shape；
- 输出必须打印完整 resolved case，而不是只打印 case 名。

禁止：

```python
# 执行函数内部偷偷写死
BLOCK_SIZE = 128
batch = 4
seq_len = 4096
```

如果 128 是算子 ABI 常量，应明确叫 `page_size` / `block_size` 并进入 Case 或 operator contract，而不是成为 benchmark 隐藏常数。

---

# 5. Shape 来源必须至少支持三层

## 5.1 CLI 临时覆盖

用于快速实验：

```bash
--batch 4 --q-len 16 --hidden-size 5120
```

## 5.2 内置 Named Cases

用于项目长期回归：

```python
CASES = {
    "small": Case(...),
    "typical": Case(...),
    "large": Case(...),
    "boundary": Case(...),
}
```

## 5.3 外部 JSON / JSONL / YAML 配置

用于大量 shape sweep 和 CI：

```bash
python test_op.py --cases-file cases.json
```

推荐 JSON 作为最低公共格式，避免 benchmark 依赖额外 YAML package。

### Shape override 规则

推荐顺序：

```text
内置默认值
    < named case
    < cases-file
    < CLI 显式覆盖
```

最终必须输出 resolved case，确保用户知道实际跑的 shape。

---

# 6. 输入生成必须与 kernel 实现解耦

结构应为：

```text
Case
  ↓
build_inputs(case)
  ↓
同一份 immutable logical input
  ├─ kernel A
  ├─ kernel B
  ├─ kernel C
  └─ reference
```

## 6.1 多 kernel 必须消费同一逻辑输入

A/B/C 不能分别随机生成输入，否则性能和精度结果不可比较。

推荐：

```python
base_inputs = build_inputs(case, seed)
inputs_a = clone_if_mutating(base_inputs)
inputs_b = clone_if_mutating(base_inputs)
```

## 6.2 明确 kernel 是否原地修改输入

每个 tensor 标记：

```text
read-only
in-place
output
workspace
metadata
```

若某 kernel 会原地修改输入，多 kernel 对比时必须在计时窗口外恢复输入，或为每个 kernel 准备等价副本。

## 6.3 不只使用随机输入

至少允许：

```text
random
zeros
ones
monotonic / arange
adversarial
boundary-specific
real captured metadata
```

layout / page table / index 类算子禁止只用全零全一数据，因为会隐藏地址错误。

---

# 7. Correctness 必须是一等模式，不与 benchmark 混在一起

推荐统一参数：

```bash
--validate none
--validate exact
--validate allclose
--validate reference
--validate-only
```

## 7.1 Reference 原则

优先级：

```text
独立 PyTorch/CPU 数学 reference
> 官方已有 reference
> 已确认稳定的旧实现
> 另一个 Triton kernel
```

另一个 Triton kernel 不应成为唯一 reference，因为可能共享相同错误。

## 7.2 Validate 在 timing 外进行

流程：

```text
build inputs
→ compile/warmup
→ validate
→ synchronize
→ benchmark
```

不要把 reference、`torch.testing.assert_close`、CPU copy、debug print 放进性能窗口。

## 7.3 输出检查顺序

优先：

```text
shape/dtype/device
→ finite / NaN / Inf
→ 语义不变量
→ exact / allclose
→ max/mean error
→ worst index
```

对于应完整写出的 output，可提供 debug 模式用 NaN 初始化输出，以暴露漏写。

---

# 8. Case 集必须包含“典型 + 边界 + 反例”

每个算子第一版 case 至少包含：

```text
P0 典型业务 shape
P0 最小合法 shape
P0 最大/重点业务 shape
P0 tile/block/chunk 边界
P0 边界 - 1
P0 边界 + 1
P1 非整除 tail
P1 batch=1 与多 batch
P1 动态长度/ragged（若接口支持）
P1 dtype 变化（若接口支持）
P1 特殊 sentinel / empty domain（若存在）
P2 adversarial 数值或布局
```

如果优化针对某个 dispatch 分支，必须有 case 明确落在：

```text
branch 前
branch 边界
branch 后
```

不能只跑一个“典型 shape”然后宣布优化完成。

---

# 9. 性能模式必须严格定义 timing boundary

每个 benchmark 必须说明计时范围：

```text
kernel-only
operator API
multi-kernel pipeline
end-to-end wrapper
```

推荐默认测完整 public operator API；需要定位时再增加 `kernel-only` 模式。

## 9.1 计时窗口内禁止

```text
随机输入构造
reference
正确性检查
print
文件 IO
编译
首次 import
无关 CPU→NPU copy
无关 tensor allocation（除非 API 本身必须分配且这就是被测合同）
```

## 9.2 标准参数

至少支持：

```bash
--warmup 30
--iters 100
--repeats 10
--timing device|wall
```

建议输出：

```text
p10
p50
p90
mean
min
max
MAD 或 std
```

主比较指标优先使用 p50，并同时观察稳定性。

---

# 10. 多 kernel 对比必须采用公平调度

## 10.1 Pair / multi-kernel order

A/B 对比至少支持：

```text
AB
BA
balanced-random
```

推荐默认 `balanced-random`，避免：

- 后运行者缓存更热；
- 温度/频率随时间变化；
- 固定顺序系统偏差。

N-kernel 时每 repeat 应随机或轮换 kernel 顺序，并记录实际 order。

## 10.2 同环境原则

同一比较组必须保持：

```text
同 device
同进程策略
同输入
同 dtype
同 graph 状态
同 synchronization boundary
同 warmup/iters
```

不同机器的绝对时间不能直接做版本性能结论；只能比较同一环境中的同 run 或严格同配置 run。

---

# 11. 必须支持 process isolation

最低支持：

```bash
--isolate-cases
```

或：

```bash
--isolate-shapes
```

每个 case/shape 用独立 child process 运行。

用途：

- 避免不同 shape 的 Triton compile/cache 状态互相污染；
- 某个 shape 编译崩溃不影响其他 shape；
- UB overflow、device fault、OOM 能定位到具体 case；
- sanitizer/msprof 更容易使用；
- 清晰记录每个 case 的 kernel hash 和环境。

对于稳定的快速 smoke test 可以同进程；正式 sweep 建议支持 isolation。

---

# 12. 必须支持“只跑一个 case 一次”的 Debug/Profile 模式

一个好 case 文件不应再额外写 `profile_xxx.py`。

至少支持：

```bash
--case NAME
--kernel LABEL
--warmup 20
--iters 1000
--repeats 1
```

从而直接嵌入：

```bash
msprof ... -- python test_op.py ...
```

或：

```bash
mssanitizer --tool=memcheck -- python test_op.py ...
```

Profile 模式应能够：

- 输入只构造一次；
- 先 warmup；
- 目标 kernel 稳态重复足够多次；
- 不在循环中重新 random；
- 可复用 output/workspace。

---

# 13. 输出必须结构化并可追溯

终端给人看，JSON 给机器看。

每个 case 至少输出：

```json
{
  "case": {...},
  "kernel": {
    "label": "O2",
    "path": "/abs/path/op_o2.py",
    "entry": "run_op",
    "sha256": "..."
  },
  "runtime": {
    "python": "...",
    "torch": "...",
    "torch_npu": "...",
    "device": "..."
  },
  "validation": {...},
  "timing": {...}
}
```

多 kernel 时输出：

```text
每个 kernel 的独立统计
+ 相对指定 reference kernel 的 delta / speedup
+ paired / same-repeat 比较结果
```

支持：

```bash
--json-out result.json
```

输出文件必须可在之后独立解释，不能依赖“我记得当时跑了什么参数”。

---

# 14. Kernel Adapter 层是必须的

不同文件的 public API 可能不同，但 benchmark 主循环不应该包含大量：

```python
if module_name == ...
```

应建立统一 adapter：

```python
def call_kernel(module, case, inputs):
    ...
```

复杂算子可注册：

```python
ADAPTERS = {
    "prefill": call_prefill,
    "decode": call_decode,
}
```

Adapter 负责：

- 从 Case 解析 host 参数；
- 调目标 public API；
- 对齐不同版本允许存在的 API 兼容差异；
- 返回统一的 outputs 对象。

但 adapter 不能修改 kernel 数学语义来“让两边对得上”。

---

# 15. 多阶段算子必须同时支持子阶段与完整流水

例如：

```text
score → topk → merge → finalize
```

case 文件应能选择：

```bash
--stage score
--stage topk
--stage full
```

这样：

- full 证明端到端收益；
- stage 证明性能瓶颈和收益来源；
- stage correctness 可快速定位问题。

但输出报告必须明确当前测的是哪一个 boundary，不能用单 kernel 数字代替完整 operator 数字。

---

# 16. Compile Failure 也是正式结果

case runner 应把失败分类为：

```text
PASS
NUMERIC_MISMATCH
NONFINITE_OUTPUT
COMPILE_FAILURE
UB_OVERFLOW
DEVICE_RUNTIME_FAULT
OOM
TIMEOUT
UNKNOWN_FAILURE
```

多 case sweep 中，一个 case 编译失败时：

- isolation 模式下保存该 case 日志；
- 给出 case name、shape、kernel hash；
- 继续其他 case（除非指定 fail-fast）；
- summary 中明确失败，而不是没有数据。

这对 Triton-Ascend 特别重要，因为 tile/shape 可能在编译阶段触发资源边界。

---

# 17. 推荐 CLI 规范

所有新 Triton case 尽量统一：

```text
Kernel:
  --kernel LABEL=FILE:ENTRY       可重复
  --reference-kernel LABEL

Case:
  --case NAME
  --cases a,b,c
  --cases-file FILE
  --case-json JSON
  --<shape-dim> VALUE

Mode:
  --task correctness|performance|simulator
  --stage NAME
  --validate none|exact|allclose|reference
  --validate-only

Shape Set:
  --shape-set gradient|business|simulator
  --sweep-axis NAME
  --sweep-values v1,v2,v3

Benchmark:
  --warmup N
  --iters N
  --repeats N
  --timing device|wall
  --order balanced-random|fixed|random

Execution:
  --device npu:0
  --seed N
  --isolate-cases
  --fail-fast

Simulator:
  --dump-inputs FILE
  --dump-contract FILE

Output:
  --json-out FILE
  --verbose
```

参数名可根据算子补充，但这些公共能力尽量不重新发明。

---

# 18. 推荐内部代码结构

```text
imports
↓
Case / KernelSpec / Result dataclass
↓
CLI parser
↓
load_kernel_from_file()
↓
resolve_cases()
↓
build_inputs(case)
↓
build_reference(case, inputs)
↓
call_kernel(module, case, inputs)
↓
validate_output()
↓
benchmark_one_kernel()
↓
benchmark_kernel_group()
↓
run_one_case()
↓
run_case_sweep()
↓
summary / JSON
↓
main()
```

不要把所有逻辑堆进 `main()`。

---

# 19. 第一版 Case 文件的最低验收标准

一个新算子的 case 文件只有同时满足下面条件，才认为“磨刀完成，可以开始优化 kernel”：

```text
[ ] kernel 文件可通过 CLI 任意替换
[ ] kernel entry 可配置
[ ] 支持 --task correctness|performance|simulator
[ ] correctness 输出聚焦精度，不混入性能结论
[ ] performance 输出聚焦时间/趋势，reference 不进 timing loop
[ ] simulator 支持静态直接调用和 input/contract dump
[ ] shape 可通过 CLI 修改
[ ] shape 明确分类为 gradient / business / simulator
[ ] 支持 named cases
[ ] 支持 cases-file
[ ] 支持 1×1 / N×1 / 1×N / N×N
[ ] 多 kernel 共享同一逻辑输入
[ ] 有独立 correctness/reference 路径
[ ] validation 不进入 benchmark 计时窗口
[ ] 有 warmup / iters / repeats
[ ] 有 p50 与稳定性统计
[ ] 支持单 case debug/profile
[ ] 支持 case isolation
[ ] 输出 kernel path + SHA256
[ ] 输出 resolved shape/config
[ ] 支持 JSON 结果
[ ] 边界 shape 至少覆盖 boundary-1 / boundary / boundary+1
[ ] 编译失败、UB overflow、device fault 可定位到具体 case
```

不满足这些能力时，不建议为某个算子继续复制一份新的临时 benchmark。

---

# 20. 禁止事项

```text
不要把 kernel 文件名写死在 import 语句里。
不要在 correctness 任务里同时输出一堆 speedup 并据此做性能结论。
不要在 performance repeat 内执行 reference、assert 或误差统计。
不要用 gradient sweep 代替真实 business shape 验收。
不要让 simulator case 每次随机生成不同 shape/metadata。
不要每优化一个版本就复制一份新 benchmark。
不要只支持 baseline/candidate 两个固定变量。
不要让 A/B 分别随机生成输入。
不要把 reference 放进计时循环。
不要把首次编译算进性能。
不要只测一个典型 shape。
不要隐藏最终 resolved shape。
不要用另一个 Triton 实现作为唯一 correctness reference。
不要让一次失败直接中断整个多 case sweep 而丢失其他结果。
不要跨机器直接比较绝对 latency 后宣布版本提升/退化。
不要把完整 API、单 kernel、helper 的时间混成一个指标。
不要为了 profile 再写一份与正常 benchmark 输入逻辑不同的脚本。
```

---

# 21. 推荐开发顺序

拿到新 Triton 算子后：

```text
Step 1  冻结 API、shape、dtype、layout、语义
Step 2  写 Case dataclass 和输入生成器
Step 3  写动态 kernel loader + adapter
Step 4  写 independent reference / invariants
Step 5  建立 gradient / business / simulator 三类 shape 资产
Step 6  打通 business + boundary 的单 case correctness
Step 7  打通多 case correctness sweep
Step 8  建 gradient 单轴性能 sweep
Step 9  加 benchmark 计时与统计
Step 10 加多 kernel balanced comparison
Step 11 建 sim_smoke + sim_business_static 直接调用
Step 12 加 isolation / JSON / input-contract dump / profile 单 case
Step 13 保存初始 baseline
Step 14 才开始修改和优化 kernel
```

原则：**case 是先于优化代码存在的基础设施，而不是优化结束后的补充测试。**

---

# 22. 与项目其他 Skill 的关系

本 Skill 负责“如何优先编写可复用 cases”。

它与已有 Skill 的边界：

```text
triton_dev_skill
  → 如何理解算子、建立变量合同、发现性能瓶颈

triton_debug_skill
  → 如何设计有判别力的正确性/数值/边界回归

npu_triton_project_skill
  → Ascend NPU 项目硬约束、benchmark/profile 证据纪律

triton_case_first_skill（本文件）
  → 如何把上述原则落实成每个算子统一、可复用的 case runner
```

后续新算子的 benchmark/test 默认优先遵循本 Skill，而不是从历史某个算子的 case 脚本直接复制后继续特化。