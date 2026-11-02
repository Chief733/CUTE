# FastAnalyse 阶段结果

基准输入：`benchmarks/20260918_three_way/offload/segments`，5 个窗口、8 个 Rank，共 40 个 rank-segment；每段覆盖 4 个训练 step。训练窗口实测到达间隔约 28.3 秒。

## 已完成

- C++17 直接扫描 `ts_track`、`stars_soc`、`task_track`、`api_event`、hash/type 字典和 compact HCCL op-info。
- 输出 step、device task 类型、kernel/node 名称、耗时分位数、top-K、HCCL 元数据及解析覆盖率，不调用 MindSpore/CANN `analyse()`。
- Python 完成跨 Rank step/kernel/HCCL 对比和异常标记。
- 接入 sidecar：最多 8 个 rank 解析进程与归档并发；跨 Rank 汇总使用独立轻量进程。
- tmpfs 使用双门闩回收：持久化归档完成，且 rank 快速解析已经成功或明确失败。解析失败不会丢失磁盘原档，也不会永久占住 tmpfs。
- 支持协调进程重启后接管仍存活的子进程，或恢复已经原子落盘但尚未登记的结果。
- 合法采集若没有 `ts_track`，明确标记 `step_trace_available=false`，step 列表为空，但继续输出 device task、kernel 和 HCCL；非空但残缺的记录仍失败闭合。

## Sidecar 从原始版本到当前版本的改动

这里的“原始版本”特指 2026-09-24 交接总结中已经存在、并用于 `20260918_three_way/offload` 的 sidecar，不是没有旁路采集的 MindSpore 原生基线。原始版已经具备以下能力：训练回调按窗口把 CANN 原始数据写入 `/dev/shm`，`stop/finalize` 后原子发布 `ready.json`；独立协调进程用一个归档线程逐 rank 做 SHA-256、文件和目录 `fsync`、原子改名提交；归档完成后再由单个独立进程在 `analysis/` 副本上串行调用 MindSpore `Profiler.analyse()`。训练不等待归档和解析，容量不足时停止新增采集但让训练继续。这些基础能力被保留，而不是本轮新增。

本轮改动将 sidecar 从“异步保存原始数据、事后慢解析”扩展为“异步保存原始数据、同时在线快速解析和跨 Rank 汇总”：

| 方面 | 原始版本 | 当前版本 |
| --- | --- | --- |
| 解析实现 | Python 启动 MindSpore/CANN 官方 `analyse()`，生成完整 timeline/CSV/SQLite | 新增 C++17 `fast_parse`，直接扫描原始定长二进制，只生成在线诊断所需的紧凑 JSON；官方 `analyse()` 保留为可选离线兼容路径 |
| 解析触发点 | 必须先等 `archive.json` 持久化提交，再解析磁盘归档副本 | `ready.json` 发布后立即从已封口、只读的 tmpfs 段启动快速解析；归档和快速解析并行。重启恢复时也可从已归档输入补跑 |
| 并行度 | 全局最多一个官方解析进程，40 个 rank-segment 串行排队 | 最多 `--fast-workers` 个快速解析进程，默认 8 个，可并行处理同一窗口的 8 个 Rank；跨 Rank 汇总由另一个轻量进程完成。官方解析若开启，仍保持独立单任务 |
| 进程分工 | 训练进程负责采集；协调进程负责串行归档和调度官方解析 | 训练进程仍只负责采集；一个常驻 Python 协调进程管理一个归档线程、最多 8 个单 CPU C++ rank 解析进程、一个轻量窗口聚合进程，以及可选的一个官方解析任务 |
| 原始数据生命周期 | 磁盘原子提交后即可删除 tmpfs 段 | 增加“双门闩”：只有磁盘归档提交成功，并且对应快速解析已经成功或明确失败，才释放 tmpfs 段；任一消费者仍在使用时均不删除 |
| 故障语义 | 归档失败保留 RAM 原件；官方解析失败保留磁盘原档 | 延续原语义，并补充快速解析失败闭合：先确保归档成功，再记录 `failed` 并释放 RAM，磁盘原档仍可重试；解析失败既不会丢原始数据，也不会永久泄漏 tmpfs |
| 运行模式 | 默认“归档 + 官方分析”，另有 `archive-only` | 新增 `--fast-analyse`、`--no-official-analyse`、`--fast-workers`、超时、top-K、异常比率、期望 Rank 集等参数；支持“仅归档”“归档 + 官方分析”“归档 + 快速分析”“快速和官方双写” |
| 输出 | `segments/.../archive.json`、官方 `analysis/` 和 `parse_result.json` | 新增每 Rank 的 `fast/segment_N/rank_R.json`、进程终态 `rank_R.result.json`、8 Rank 的 `window_summary.json`/`window_result.json`，原始归档目录保持不变 |
| 在线统计 | 依赖官方 CSV/timeline；训练退出时曾有 39/40 段待解析 | 直接输出 step、device task 类型、具名 kernel 统计、top 慢任务、HCCL 元数据、结构 hash、覆盖率，并跨 Rank 比较 kernel/HCCL 数量和耗时、标记异常 |
| 生命周期与恢复 | 有后台进程和基本状态文件 | 增加 PID namespace ID、`/proc` PID 和进程启动时钟三元身份，防止 PID 复用；文件锁拒绝重复协调器；重启可接管仍存活的 rank/窗口子进程，也能恢复已原子发布但未登记的结果；子进程带超时、降优先级、CPU 亲和性和干净环境 |
| 状态与验收 | 主要观察归档、官方解析积压和采集覆盖 | `status.json` 新增 fast rank/window 的 pending、active、complete/failed 和真实子进程状态；`inspect_run.py` 同时校验 Rank/step 连续性、快速结果 schema、窗口完整性及所有归档文件 SHA-256 |

主要代码落点如下：

- 新增 `fastanalyse/src/fast_parse.cpp`：负责原始二进制解码、关联 host/device task 并生成 `fast_summary_v1`。
- 新增 `fastanalyse/tools/aggregate_files.py`：只读取紧凑的 Rank JSON，产生 `fast_window_summary_v1` 跨卡对比。
- 新增 `profiler_sidecar/fast_worker.py`：管理有界并发、超时、结果原子发布、进程接管和窗口聚合。
- 修改 `profiler_sidecar/worker.py`：把快速解析接到 `ready.json` 路径，与归档并行，并实现归档完成与快速解析终态的双门闩回收。
- 修改 `profiler_sidecar/launch.py`：加入快速/官方解析开关和参数检查，记录快速路径配置；旧启动方式仍默认运行官方分析。
- 修改 `profiler_sidecar/common.py`：加强容器 PID namespace 下的进程身份和存活判断，避免 PID 复用或错误接管。
- 修改 `profiler_sidecar/inspect_run.py`：增加快速 Rank/窗口结果检查，并可将完整 SHA-256 与覆盖验证保存为 `verification.json`。
- 新增 `scripts/run_llama7b_fastanalyse.sh`：推荐在线入口，等价于启用 `--fast-analyse --no-official-analyse --fast-workers 8`。
- 增加快速解析单元测试、40 段官方 SQLite 对照、sidecar 生命周期测试、2 卡在线 smoke/archive-only A/B，以及真实 Llama2-7B 8 卡验收。

性能结果也发生了量级变化。原始官方解析处理 40 个 rank-segment 累计约 1.71 小时，并且训练退出时曾只有 1/40 完成；当前快速解析离线并行回放 40 段只需 2.128 秒。真实 8 卡在线运行中，4-step 窗口最短每 28.300 秒到达一次，完整 8-rank 快速解析及聚合最慢 1.833 秒，稳态吞吐余量为 15.44 倍，窗口之间没有积压。

以下边界没有被本轮改动消除：训练进程仍同步执行原生 `Profiler.stop()`/`finalize()`，仍承担采集钩子、事件编码和 tmpfs 写入成本；`ready.json` 只表示内存段已封口，不表示已经持久化；快速摘要不是完整 timeline/CSV/SQLite 的替代品；当前 HCCL duration 是 host launch API 区间，不是设备通信完成时长；checkpoint 与 profiler 归档共用共享文件系统时仍可能出现持久化长尾。

## 正确性

本地历史产物 `validation_40x8_final.json` 的 40/40 个输入全部通过。该 JSON 包含历史 Trace 的绝对路径，不提交到源码仓库，可用文末命令重新生成：

- step ID、边界和数量与官方 `step_trace` 一致；
- device task 类型与数量一致，时间差只包含官方纳秒取整误差；
- 112,816 个示例 host kernel 映射与官方 `TaskInfo` 一致；
- kernel 分组、数量和耗时统计一致；
- 每段 2,168 个 HCCL op、11 个分组及 op/data type/algorithm/group/count 元数据一致；
- compute kernel 名称覆盖率为 98.08%–98.11%，其余记录仍按稳定的 device task 类型保留，没有静默丢弃。

## 吞吐

最终 40 段、8 worker 回放数值来自本地历史产物 `benchmark_40x8_final.json`。该文件不提交到源码仓库，可用文末命令重新生成：

| 指标 | 结果 |
| --- | ---: |
| 40 段总墙钟时间 | 2.128 s |
| 处理速率 | 18.80 rank-segment/s |
| 被扫描输入吞吐 | 1064.4 MiB/s |
| 单 rank 端到端 p99 | 0.440 s |
| 8-rank 窗口 p99 | 0.443 s |
| 占 28.3 s 窗口预算 | 1.57% |

通过完整 sidecar 协调器回放一个真实 8-rank 窗口（包含 0.5 秒轮询及跨 Rank 汇总）耗时 1.202 秒，8/8 rank 和 window 均为 `complete`。

归档本身在 40 段上的累计用时为 80.43 秒，低于 5 个训练窗口的 141.5 秒到达预算。四个正常窗口逐 rank 串行归档合计 8.55–9.27 秒；一个 Lustre 长尾窗口为 44.46 秒，因此平均吞吐可持续，但仍需要现有 tmpfs 缓冲吸收存储长尾。

## Llama2-7B 真实 8 卡在线验收

2026-10-20 在 8×910B2 上完成 20-step、seq_len=4096 的真实训练。step 3–20 在线采集，每 4 step 封一个窗口，关闭官方 `analyse()`，启用 8 个快速解析 worker。运行目录为 `profiler_sidecar/runs/llama7b_fast_8card_20261020`，机器可读结果为其中的 `speed_report.json` 和 `verification.json`。

- 训练稳态均值 7.116 秒/step；完整 4-step 窗口实际到达间隔为 28.300–28.726 秒，中位数 28.419 秒。
- 4 个稳态窗口、32 个 rank-segment 的 C++ 内部解析 p99 为 0.431 秒，最慢 0.437 秒。
- 从最后一个 rank 的原始数据 ready 到完整 8-rank 汇总可读，4 个稳态窗口为 1.729、1.773、1.833、1.755 秒，最慢占最短窗口预算的 6.48%。保守吞吐余量为 15.44 倍；若只看 C++ 解析，余量为 64.78 倍。
- 每个稳态汇总都在下一窗口到达前完成，最小剩余时间 26.466 秒；串行持久化归档最慢 14.238 秒，也小于 28.300 秒到达间隔。
- 40/40 rank 解析和 5/5 跨 Rank 汇总完成；step 3–20 在所有 rank 连续覆盖；2,184,045,834 bytes 原始文件逐个 SHA-256 校验通过，无采集中断。结束时 fast rank/window pending 和 active 均为 0，tmpfs `raw_bytes=0`。

因此，目标工作负载的**稳态在线解析吞吐已经实测快于训练数据产生速度**，完整 8-rank 可分析结果有至少 15.44 倍余量。

末尾还有一个需要保留的系统边界：step 19–20 是 2-step 尾窗，C++ 解析最慢仍只有 0.199 秒，但框架同时向共享文件系统保存约 72 GiB checkpoint，导致串行归档耗时 34.424 秒、跨卡汇总结果从 ready 到发布耗时 35.468 秒。该运行没有下一窗口且最终完全排空，所以不影响上述稳态吞吐结论；但它暴露的是持久化元数据/归档与 checkpoint 的 I/O 争用，而不是解析计算慢。在解决输出控制面与 checkpoint 存储隔离前，不应宣称 checkpoint 写盘期间也始终保持 1.833 秒延迟上界。

## `sink_size=2` 的逐实际 Step 耗时修复与验收

2026-10-21 保持 `sink_size=2`，完成了延迟采集条件下逐实际 step trace 的修复。根因不是 `fast_parse` 漏解码，而是旧 sidecar 在 `init_start_profile: false` 时直到热身结束才创建 Profiler；此时训练图已经编译，图内没有 `60000/60001` step marker。现在首个 Profiler 在图编译前以 `start_profile=False` 初始化，到第二个 sink 开始前才执行 `start()`，所以不采集编译和 step 1–2，同时保留图内逐-step 插桩。编译缓存摘要新增 instrumentation 版本，防止复用旧版未插桩图。

干净端到端运行位于 `profiler_sidecar/runs/llama7b_steptrace_preinit_8card_20261021c`：8 卡 Llama2-7B、seq_len 4096、step 1–2 热身、采集 step 3–6、4-step 窗口、无 checkpoint。训练返回 0，sidecar 为 `complete`，8/8 rank 快速解析和 1/1 跨 Rank 汇总均为 `complete`；476,769,578 bytes 原始数据逐文件 SHA-256 校验通过，错误列表为空。

每个 Rank 都解析出 8 条 step trace 记录、4 对完整 start/end；设备原始 `index_id` 因延迟启动从 1–4 计数，解析器通过 `ready.json` 的声明区间映射为权威 `global_step` 3–6。跨 Rank 设备耗时如下：

| global step | Rank 中位数 | Rank 最小值 | Rank 最大值 |
| ---: | ---: | ---: | ---: |
| 3 | 7019.694 ms | 7006.681 ms | 7033.436 ms |
| 4 | 7125.028 ms | 7114.125 ms | 7179.999 ms |
| 5 | 7145.558 ms | 7071.909 ms | 7178.174 ms |
| 6 | 7085.631 ms | 7059.179 ms | 7149.999 ms |

完整每 Rank、每 step 数值位于 `fast/segment_000000/window_summary.json` 的 `step_comparisons[].per_rank_ns`，单 Rank 原始边界和耗时位于 `rank_R.json` 的 `steps[]`。4-step 数据窗口产生耗时为 28.432–28.516 秒；最慢 C++ Rank 解析 0.396 秒，从最后一个 Rank `ready.json` 到 8-Rank 汇总可读为 1.682 秒。按完整在线结果计算，解析速度相对训练窗口仍有 16.90 倍余量。

快速路径同时增加了 `ts_track_available`、`step_trace_available`、`step_trace_complete` 和 `expected_step_pairs`，以及 `--require-complete-step-trace` 强校验。在线推荐脚本默认启用强校验：任何 Rank 缺少实际 step、存在未配对 marker 或全局 step 不连续，都会明确失败，不能再以空 `steps` 的成功结果通过。

## 8 卡 20-step 多窗口逐 Step 验收

2026-10-21 使用 `sink_size=2` 完成新的 8 卡 Llama2-7B、seq_len 4096、20-step 在线实验。运行目录为 `profiler_sidecar/runs/llama7b_steptrace_multiwindow_8card_20step_20261021b`；step 1–2 用于热身，连续采集 step 3–20，每 4 step 封口，形成 `3–6`、`7–10`、`11–14`、`15–18` 和尾段 `19–20` 共 5 个真实窗口。本次显式关闭训练 checkpoint，避免共享存储 I/O 干扰解析延迟。

- 训练返回码为 0；40/40 rank-window 和 5/5 跨 Rank 汇总全部完成，失败、采集中断和最终积压均为 0。
- 每个 Rank 都连续覆盖 step 3–20：18 个 global step × 8 Rank，共 144 个逐 Rank/step 设备耗时；所有 rank-segment 的 `step_trace_complete=true`，没有缺失或未配对 marker。
- 2,184,173,685 bytes 原始数据逐文件 SHA-256 校验通过，`verification.json` 的 `errors` 为空。
- 稳态训练均值为 7.027 秒/step；完整 4-step 窗口的最短到达间隔为 28.042 秒。
- 32 个完整窗口 rank 解析的 p99 为 0.432 秒，最慢为 0.437 秒；从最后一个 Rank 发布原始数据到完整 8-Rank 汇总可读，4 个完整窗口分别为 1.599、1.328、1.289、1.794 秒。
- 以最短窗口间隔和最慢完整汇总计算，在线吞吐余量为 15.63 倍；若只看 C++ rank 解析为 64.17 倍。每个窗口都在下一个窗口到达前完成，完整窗口最小剩余时间为 26.555 秒。

机器可读的吞吐、窗口和 18 个 step 汇总位于 `speed_report.json`；所有 144 个原始数值位于各窗口 `window_summary.json` 的 `step_comparisons[].per_rank_ns`。这次结果同时证明：快速解析在多窗口在线运行中持续快于训练数据产生速度，并且输出确实包含每个被采集实际 step、每个 Rank 的耗时。

## 真实在线 smoke 与 A/B

使用 2 张 910B2、12 个真实 MatMul + AllReduce step、每 4 step 一个窗口完成在线验证：

- 训练正常完成，2 Rank × 3 窗口全部归档；6/6 rank 快速解析、3/3 跨 Rank 汇总均为 `complete`；
- SHA-256、Rank 覆盖和 step 1–12 连续性检查无错误，结束时 `raw_bytes=0`，峰值仅 200,838 bytes，无采集中断；
- 该手工 Cell smoke 不产生 `ts_track`；降级路径仍在每段解析出 62 个 device task、4 个 kernel 分组和 1 个 HCCL 分组；
- C++ 内部解析为 2.68–2.89 ms；含 sidecar 0.5 秒轮询的 `ready.json` 到结果 p99 为 1.082 秒；
- 与相同采集、相同归档但不开快速解析的 archive-only 对照相比，两 Rank 训练循环平均总时长为 3.0226 s 对 3.0160 s，单次 A/B 差异 +0.22%；训练启动器总耗时反而低 0.21 s。样本较小，只能说明没有观察到明显增量，不能把 0.22% 当作稳定性能结论。

通过记录位于 `profiler_sidecar/runs/fast_online_smoke_20261020c`，对照位于 `profiler_sidecar/runs/fast_online_control_20261020`。

## 8 卡 Llama2-7B 40-step 训练开销 A/B（2026-10-21）

使用 `sink_size=2`、每次训练 40 个实际 step、step 1–2 热身、step 3–40 连续采集、每 6 step 切窗、关闭 checkpoint。N 为无 profiling，A 为采集和归档，B 在 A 基础上启用 8 个快速解析 worker 并关闭官方 `analyse()`。首轮 N/A/B 用于试跑和缓存预热，不进入正式统计。随后按 B→A、B→A、A→B、A→B 运行四对全 Rank 命中编译缓存的 A/B，并穿插四次预热后的 N。

主指标使用 Rank 的 `MFLossMonitor.step_end` 单调时钟事件，计算 step 2 到 step 40 的间隔/38，再取 8 Rank 最大值；因此包含训练期间窗口切换，却不混入首次编译或训练退出后的后台排空。四次正式运行的秒/step 中位数为 N **6.9810**、A **7.0241**、B **7.0485**。A/N 为 **+0.62%**，B/N 为 **+0.97%**，这两个是跨运行组中位数的描述性比较。四对 B/A 差值依次为 **+0.77%、−0.04%、+0.44%、+0.18%**，均值 **+0.34%**；基于四对的近似 95% Student-t 区间为 **−0.22%～+0.89%**，不能判定快速解析造成稳定减速。

15 次训练均正常结束；10 次 A/B 的 560 个 Rank 窗口段归档完整；5 次 B 的 280 个 Rank 快速解析和 35 个窗口汇总全部成功。8 Rank 的计时事件与 step 3–40 覆盖完整，无采集中断，最终 tmpfs 原始积压为 0。完整设置、逐运行数值、缓存命中和结论边界见 `profiler_sidecar/runs/overhead_8card_40step_6window_20261021b/OVERHEAD_REPORT.md`，机器可读汇总为同目录的 `summary.json`。本次没有做归档文件的第二次全量 SHA-256 复算。

## 前三类诊断离线回放（2026-10-21）

新增 `tools/diagnose_windows.py`，对一次运行的 fast 窗口摘要依次运行设备 step 时长异常、同角色 rank 耗时偏离和滚动趋势/变点原型。它对每个 rank 的 `global_step`、声明区间、`step_trace_complete`、step 数和跨 Rank 摘要做交叉核对；数据不全时输出质量问题并降级比较。角色组通过命令行显式传入。输出 schema 为 `fast_diagnosis_replay_v1`，报告独立存放，不改动原始归档或 fast 摘要。

四次预热后 B 运行 `03_B`、`07_B`、`10_B`、`13_B` 分别回放：每次 7 个窗口、38 个实际 step、304 个 rank-step 全部有效，合计 **28 个窗口、1216 个 rank-step**。默认实验阈值（相对 10%、绝对 250 ms、稳健 z=5）下，四次运行的三类发现均为 0、数据质量问题均为 0。这只说明这些正常样本上没有触发候选，不构成生产误报率或漏报率估计。每次机器可读结果位于 `diagnosis_runs/overhead_8card_40step_6window_20261021b/{03_B,07_B,10_B,13_B}.json`。

`tools/evaluate_diagnosis.py` 在 `03_B` 的真实摘要副本上做了六组可复现回放，当前验收结果位于 `diagnosis_runs/overhead_8card_40step_6window_20261021b/injection_eval_v2/evaluation.json`：

| 情景 | 预期与实测 |
| --- | --- |
| 未修改基线 | 304/304 rank-step，有效；0 发现、0 质量问题 |
| rank 7 的 step 28 增加 2 秒 | 自身历史尖峰和同一步同伴偏离各 1 条 |
| 从 step 28 起所有 rank 增加 1.4 秒 | 检出持续回退；同伴偏离为 0 |
| 从 step 15 起所有 rank 每步累加 140 ms | 检出渐进漂移；同伴偏离为 0 |
| segment 3 缺 rank 7 | 报不完整窗口、缺 rank、同伴组不完整，未把缺失当正常值 |
| 缺整个 segment 3 | 报 segment 与 step 断档，后续历史基线重置 |

六组情景全部通过预期检查；阶跃回退只输出 `regression`，渐进变慢只输出 `drift`。每组的最小输入副本与完整诊断 JSON 均保留在 `injection_eval_v2/` 下。另有 `tests/test_diagnose_windows.py` 覆盖单点尖峰、持续回退、渐进漂移、缺 rank、跨摘要不一致与角色组校验。当前阈值仍需正常长训和更真实的故障样本校准；真实同步训练可能抹平单 rank 的整步差异，因此同伴耗时偏离只能作为症状证据。该节只描述离线回放；主机 sink 与进度心跳的在线接入见下一节。

## 前三类诊断在线化与 8 卡验收（2026-10-21）

训练 callback 现在按 `sink_size=2` 非阻塞投递主机批次时长和完成 step；独立线程写 tmpfs JSONL，并每 5 秒发送心跳、丢弃计数与观察到的队列深度。首个含编译的 sink 批次从历史基线中排除。sidecar 增量消费遥测，在每个 6-step 快速窗口汇总完成后自动运行设备逐 step 诊断，原子发布 sink、同组 Rank、窗口、进度及数据质量结果；消费游标和有限历史可在进程重启后恢复。主机批次耗时与设备逐 step 耗时分别建基线，不混算。

正式验收使用 8×910B2、Llama2-7B、40 个实际 step、step 1–2 热身、step 3–40 连续采集、7 个窗口、关闭 checkpoint。修正预热处理后的在线运行 `online_diagnosis_8card_40step_6window_20261021c` 与最终带队列观测的 `...20261021e` 均完整完成：每次 160 个 Rank sink 结果、20 个同组批次结果、7 个设备窗口、304 个有效 rank-step；两次均为 0 诊断发现、0 质量问题、0 遥测丢弃。最终轮 8 卡观察到的遥测队列最大深度为 1；正常样本的同批次 Rank 正向相对偏差最大约 0.48%（修正轮），低于当前 10% 实验阈值。这些短样本上的 0 发现不能估计生产误报率。

最终轮的主机事件到 sink 诊断结果 p50/p95/最大为 **447/604/716 ms**；快速窗口结果完成到诊断 p95 为 **71 ms**；最后一个 Rank 发布 `ready.json` 到设备诊断 p95 为 **2.37 s**，这包含快速解析和窗口汇总。设备异常仍需等待所在 6-step 窗口封口。实测延迟来自正常训练的结果发布，并不是实际故障发生到告警的端到端分布。完整数值、覆盖和遥测位于 `profiler_sidecar/runs/online_diagnosis_8card_40step_6window_20261021e/ONLINE_REPORT.json`；修正轮报告位于对应 `...20261021c/ONLINE_REPORT.json`。

在修正轮的真实快速摘要副本上，流式注入 rank 7 的 step 28 增加 2 秒，同时增加对应主机 sink 批次 2 秒；主机自身尖峰、同批次同组 Rank 偏离、设备逐 step 自身尖峰与同一步 Rank 偏离均被检出，0 质量问题。结果位于 `profiler_sidecar/runs/online_diagnosis_stream_eval_20261021c/evaluation.json`。其约 0.99 s 主机、1.14 s 窗口 CPU 回放处理时间受同时运行的训练负载影响，不包含等待真实 sink 或窗口的时间。缺首批 sink、缺 Rank/同伴超时、进度落后、队列满时不阻塞训练且计数丢弃，以及窗口失败和重启恢复均有 CPU 单测；`profiler_sidecar` 18 项、`fastanalyse` 15 项测试通过。注入覆盖的是指定异常，不能外推为通用漏报率。

相邻同配置纯快速解析对照 `online_diagnosis_control_8card_40step_6window_20261021d` 的稳态为 **7.0543 s/step**；两次在线轮分别为 **7.0072** 与 **7.0335 s/step**，描述性差值为 **−0.67%**、**−0.30%**。三次训练 YAML 除输出路径外相同，主指标均取每 Rank step 2→40 的单调时间差/38，再取 8 Rank 最大值。本组短运行没有观察到在线诊断导致的训练减速；单个对照和运行间波动不足以给出稳定开销上界。较早的 `...20261021b` 为接线验收，运行时尚未启用正确的预热基线排除和队列深度观测，不计入上述正常样本与开销比较。

2026-10-26 为释放磁盘，已删除开销量化 10 次 A/B 运行以及在线诊断 `...21b`、`...21c`、对照 `...21d` 的原始 `segments`，保留报告、配置、计时、快速摘要和诊断结果。最终 `...21e` 的完整原始 Trace 仍保留，可用于重新解析或生成 Timeline。详细清单见 `profiler_sidecar/runs/CLEANUP_20261026.md`。

## 当前边界

- 当前 HCCL 耗时字段是 host launch API 区间，不是 device 上通信完成延迟；定位通信慢 Rank 时可先用 step、相关 task 和 launch 异常筛查，设备通信细分仍待解析 FFTS/HCCL 子任务记录。
- 二进制 schema 固定并校验于 CANN 8.0.RC3；升级 CANN 后必须重新做官方 SQLite 对照，未知或残缺记录会失败闭合。
- Llama2-7B 真实 8 卡多窗口运行已证明稳态解析吞吐不积压；40-step、6-step 窗口的四对预热后 A/B 给出快速解析增量约 +0.34% 的点估计，但小样本区间跨零，尚不能判定稳定减速。完整方案相对无 profiling 的约 +0.97% 是不同运行组中位数比较，仍受基线漂移影响。
- checkpoint 与快速结果、归档共用共享文件系统时会放大 durable `fsync` 延迟；需要将在线结果控制面迁到 tmpfs/本地盘并异步持久化，或为 checkpoint 与 profiler 归档使用不同存储路径。

## 复现

```bash
cd /root/work/cyf/pretrain/fastanalyse
make test
python3 tools/validate_replay.py --jobs 8 \
  --output validation_40x8_final.json \
  /root/work/cyf/pretrain/benchmarks/20260918_three_way/offload/segments
python3 tools/benchmark_replay.py --jobs 8 --window-interval-seconds 28.3 \
  --output benchmark_40x8_final.json \
  /root/work/cyf/pretrain/benchmarks/20260918_three_way/offload/segments
```

上述 `benchmark_*.json` 和 `validation_*.json` 是历史 Trace 的机器可读回放产物，已由 `.gitignore` 排除。
