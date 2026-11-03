# D19 C2 A3 full60：本地源码与采集候选

范围是当前迭代的完整串行 S，原始 60 case 各 10 组；F 默认为 0。
本候选尚未联网、发布、编译或上板。主报告、原包及历史 Event 没有改动。
远端唯一执行者为 hc_expansion；root 审阅并发布固定 Git commit，statistics_audit
独立检查实际输入、身份和任务映射。先 pilot 原编号 **0、14、52、56**。

## 源码、输入和 LOC

`source-manifest.json` 是逐文件原件/候选 SHA 和来源清单，`SOC-ADAPTATION.diff`
给出唯一运行源码变更：model.prepare/Prepared 增加必填 keyword-only `soc`，
传给原 C++ prepare 的严格 SoC 比较。两个 CMake 原字节不改，构建显式传
`-DSOC_VERSION=Ascend910_9362`。`c_only` 和 `__call__` AST 与原件相同：
每次完整 **cat 权重 → 新 BF16 C → 原 AscendC V**。没有把 cat 挪到 prepare。

原 V 的 CMake 同时编译旧 `cv60n_gemm.cpp`，因此作为未执行的构建依赖保留。
本 runner 从不调用旧 `cv60n_run`/旧 C prepare/run_out，不导入旧完整 C wrapper。
不复制原 DSO。旧 C build 依赖不算当前 S 的功能实现 LOC。

`IMPLEMENTATION_REVIEW.json` 在本次候选上重新扫描：device **557**、专用 host
**177**，总 **734 LOC**；与包中对应 C/V 字节一致，AscendC 变化 **0 LOC**。
LOC 是去空白和词法注释后的物理源码行；排 Python、纯框架注册、通用 MatMul。
新编译 binary 身份尚未建立，不能以源码相同冒充历史 DSO 身份相同。

`frozen/case-config.json` 是唯一计划 SHA
`25775fe0785dd902117b3d1958d70e465cb9f93f7df160d7e186cd9d83110731`
的配置/hash-only 投影，不含旧性能结果。`frozen/inputs_for.py` 仅摘出原函数，
AST 原样执行；`frozen_v` 原生成器保留被丢弃的 kv/scores 随机数推进、state/gamma
策略及 CPU 分块 FP32→BF16 转换。目标环境先在 CPU 逐例复现 5 个历史不可变输入
hash（X/Wkv/Wgate/metadata/gamma），不符则在设备分配前停止。S0 新增 digest，
历史 S0 未记录，不声称历史 state hash 相符。数值模式以 NOTICE 的逐例编号为准。

## 发布边界和本地检查

只允许 `PUBLISH_ALLOWLIST.txt` 中的路径，逐字节按 `publish-manifest.json` 校验。
不得整目录上传；materialize/review helper、本地日志、缓存、新输出均不在清单。
原始 all.json/cases.csv、旧时延/速度、环境、任何 DSO/object/build/PROF、凭据都不发布。

本地可以运行：

```powershell
python projects/perf-serial-graph/c2-cv-retest-20261102/test_candidate.py
```

本机没有 torch，故只验证源码、AST、配置、词法 LOC 和合成任务映射。
mock 不证明真实 RNG hash、ABI、Graph、kernel topology 或 NPU 性能。

初版自检20项通过仍漏了两个独立审查阻塞：prepare 未转传 soc、pilot 审批只信
顶层身份。初版45文件和183项独立审核（181通过/2阻塞）保全在本地
`review-history/v1/`，不发布该审核日志。修订版补真实 prepare→构造函数调用回归，
并从当前工作目录的 fresh build/source/bundle/runner 重建 pilot 身份，逐一核对底层
raw/topology/execution SHA 后重新运行全部任务映射。28项本地测试通过，其中 pilot
文件链含1个完整正例和26个攻击子例；独立修订复核仍待完成，上板仍未验证。

## 新目录构建与外层资源合同

固定 Git commit 的 allowlist 内容部署到全新任务目录，例如：

```bash
R=/home/s30083169/perf-serial-graph/hc-a3-20261102/work/c2-a3-cv-20261102
E=/home/s30083169/perf-serial-graph/hc-a3-20261102/evidence/c2-a3-cv-20261102
python3 "$R/build.py" --task-root "$R" --soc Ascend910_9362 \
  --cann-root "$ASCEND_HOME_PATH" --jobs 4
```

必须有实际 CANN9.2、compiler、torch/torch_npu/ABI 和 9362/arch2201 的环境原证据。
build.py 只做 fresh CMake CPU 编译；已有 build/build-evidence 直接拒绝，不清理旧目录。
它保留日志、命令、DSO/object/JSON/编译配置 SHA；runner 再检查实际 /proc/self/maps
中的 task DSO 路径/hash、无同名外来 DSO、runtime framework/ABI 与构建一致。
严格 C++ prepare 通过才构成实际 SoC 字符串相符证据。

外层 HC 在启动前核 health/HBM/磁盘、chip↔physical↔logical 映射、容器内存限制、
其他计算 PID 与实际 hostPID/ns/cgroup/starttime，不采用 PID 数推断归属。
设置原 guard 所需 PERF_HOST/PERF_NPU_ID/PERF_CHIP_ID/PERF_PHYSICAL_DEVICE 和
ASCEND_RT_VISIBLE_DEVICES；这不是改卡动作授权。runner 额外要求实际 framework
device_count=1/current_device=0。每组前后资源观察不放在 warmup→target 中间。

runner 初始化设备后写 `ownership-request.json` 并暂停，外层在 180 秒内根据真实
host /proc 证据写入本 attempt 的 receipt。其结构为：

```text
nonce: request.nonce
outer_executor_verified: true
process: request.process（逐字段一致）
selected: request.selected（逐字段一致）
host_pid: 实际宿主 PID
host_NSpid: 从宿主 /proc/<pid>/status 实际读出的整数数组，末项是容器 PID
evidence_files: [{path: 当前可读证据绝对路径, sha256: 实际SHA}]
```

这份 receipt 是外层执行者的明确归属核验，runner 不伪造外层检查。guard 的 fresh
快照仍不是锁。超时/变化即停；不占用、终止或改变他人进程。

单 case/单进程结束释放所有缓冲和 Graph pool。实际 HBM/CPU budget 由执行者依据
当时芯片和 cgroup 核实后显式给出。清单下界加 warmup 只是一道必要检查，不是峰值
保证；case41 尤其需实测 OOM 边界。失败保留原样，不裁 T/P/请求/metadata。

## warmup：真实 ops probe 后才允许 fallback

使用当前 runner 的真实 256MiB FP16 ArgMax probe；外层仍消费该 probe 进程的
ownership-request。下列命令只是交接模板，预算/epoch/可用芯片由唯一执行者填写：

```bash
python3 "$R/prepare_warmup.py" --task-root "$R" --epoch "$EPOCH" \
  --probe-command "python3 $R/runner.py --task-root $R --out-dir $E/probe --ownership-receipt $E/probe/ownership-receipt.json --epoch $EPOCH --soc Ascend910_9362 --probe-warmup-op {warmup_op}" \
  --probe-observation "$E/probe/runner-observation.json" --out-dir "$E/warmup"
```

prepare_warmup 复用未修改官方 collector 的探测/选择 helper。只有确已进入实际 ops
调用后该调用失败，才构建随包官方 compiled warmup，target arch 为 ascend910_93；
资源、归属、输入准备失败或 op 已通过但后置 gate 失败均直接停，不冒充缺 ops。
两模式的精确 256MiB tensor 都只分配一次，不在每组重复分配。

## 外部 msprof application 与 10 组

```bash
python3 "$R/collect.py" --task-root "$R" --case-id 0 --stage pilot --epoch "$EPOCH" \
  --out-dir "$E/case00-attempt01" \
  --ownership-receipt "$E/case00-attempt01/runner/ownership-receipt.json" \
  --warmup-selection "$E/warmup/warmup-selection.json" \
  --hbm-budget-bytes "$HBM_BUDGET" --cpu-budget-bytes "$CPU_BUDGET"
```

同样运行 14/52/56，每个独立 session。collector 保留 official 参数
`--aic-mode=task-based --aic-metrics=PipeUtilization`、application 长度<2048、
全 session task 数≤540。没有 API 内围计时、没有更改技能采集方式，没有自动重试
或删除失败 PROF。timeout 后外层仅清理本任务实际归属进程，复核资源后另起 attempt。

每例先复现输入，执行两次 eager、Graph capture、poison replay，并检查 eager/Graph
输出及 state bit hash、dtype/shape/finite。随后恰好 10 组：
**S0 reset + output/packedW/pair poison → sync → warmup → sync → 完整 S replay → sync**。
每组之前的资源观察不在 target 中。最终再检查输出/state anchor 和不可变输入。
这些 gate 只证明采集自洽，既有严格精度失败仍保留 **DIAGNOSTIC**。

## 实际任务先观察，独立审阅后才有指标

官方 v5 原 collector 按 OpName 全局过滤；本实现的 Graph 准备会出现与 target
同名的 setup。故保留官方原字节，显式采用 setup-aware extension，不能把 prefix
同名任务偷偷丢弃。`execution-observation.json` 明确未映射；不是假造 v5 已接受。

`ordered-task-observation.json` 保存所有 CSV 字段/原记录/物理行/任务类型/stream/
start/end/duration。独立 reviewer 从 pilot 实际 CSV 确认 C、pack、V 名称与类型，
用源码 launch/类型声明和真实 task 行证据填写 sequence-spec：

```text
{reset:[spec...], warmup:[spec...], target:[spec...]}
spec = {op_name,op_type,task_type,category,compute_kind,stage（仅target）,evidence:[{path,sha256,reason}]}
category: setup / warmup / C / V / other
compute_kind: cube / vector / noncompute
stage: pack / C / V / other
```

不在此预填实际名字、类型或 pack task 数。C/V 必须各一次源码对应设备 launch；
pack 的全部实际计算均在 target 内。未知/MIX/AICPU compute 不准强归 C/V；未支持
TaskType 保留失败，先审协议再修改有版本差异的 mapper，不猜值。

```bash
python3 "$R/draft_topology.py" --csv "$ACTUAL_OP_SUMMARY" \
  --execution-manifest "$E/case00-attempt01/execution-observation.json" \
  --sequence-spec "$REVIEWED_SEQUENCE_SPEC" --output "$E/case00-map-draft.json"
```

draft 只按实际明确 sequence 分边界，prefix/suffix 仍待审（compute_kind=null）。
reviewer 须核全 prefix/每组reset/warmup/完整target/suffix，填写真实证据 hash、
reviewer/review_evidence，改状态 REVIEWED。map 绑定本 CSV SHA、source/build/
runner/bundle/epoch/warmup/route；不能把一次 pilot 的 CSV hash 冒充另一次 session。

```bash
python3 "$R/task_mapping.py" --csv "$ACTUAL_OP_SUMMARY" --topology "$REVIEWED_CASE_MAP" \
  --execution-manifest "$E/case00-attempt01/execution-observation.json" \
  --output "$E/case00-task-mapping.json"
```

必须 **expected==parsed==mapped**，全行一次无过滤；额外、遗漏、同名 setup、类型
改变、stream 改变均拒绝。输出包含可追溯 v5 字段及显式 extension/离线审阅来源，
不声称原 name-only collector 通过。每个 C/V 用完整 Task Duration，不乘 Block Num。
copy/memset/runtime 另列 other，未知 compute 无指标。逐组算 C/(C+V) 后取中位数；
不是 median(C)/(median(C)+median(V))。C+V+other 核完整 target，总和不当 Graph wall。
有 other>0 标记 compute-only 分母需 root 审阅，不静默称整个调用占比。

四个 pilot 完整有效且独立审阅后，可作为正式 60 中的四例，不择优重采。外层写
pilot approval（status=APPROVED_FOR_REMAINING_S、reviewer、epoch、build_identity_sha256、
pilot_results=[{path,sha256}] 四项）。remaining 门禁不相信这些顶层声明：逐条读取
已绑定 SHA 的 CSV/topology/execution，核实际 UID、5输入hash、新S0、shape/dtype、
10组和严格 true gates、当前 loaded DSO、warmup、9362和四种实际路由，再对原 CSV
重算并核对全部 task/group/metric。缺失、旧身份、篡改或伪报结果均停止。
其余原 case 使用 `--stage remaining
--pilot-approval <文件>`。每个 case 都按实际 CSV 产生自己的 reviewed map；不猜其 topology。

汇总时 `summarize.py --task-root "$R" --mapping-index <JSON列表> --output <新文件>`。
index 项为 `{case_id,path,sha256}`，可记录失败；缺失留 null，要求 60 原 UID、每例 10
完整组、同 source/build/epoch。永不与历史 F Event 相除/相减；旧 Event/加速比原样。

## 交付检查点

- 本地：源码 hash/AST/LOC、60配置、28项调用/证据链/负例 mock 通过。初版20项
  自检及183项独立审核发现的两个阻塞均保留；更早的测试55–59编号假设失败也保留。
  冻结配置和PLAN未改。
- 目标环境待做：真实5hash与S0、fresh9362编译/load/ABI、资源归属/峰值、Graph/state、
  实际 task 名称/类型/数量/全行映射；pilot 未观察前无 C/V 结论。
- 原始 PROF、CSV、每组/identity/失败证据留 task evidence，不进 GitHub source bundle。
- profile 成功仍是 DIAGNOSTIC，不升级严格精度验收；不自动加测 F、模型或新 Shape。
