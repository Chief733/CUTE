# FastAnalyse

`fast_parse` 是只读 Ascend 原始 profiling segment 的离线原型。目前直接解析：

- `ts_track`：step start/end；
- `stars_soc`：ACSQ device-task start/end、耗时、类型及 stream/task ID；
- `task_track`、`api_event` 和 hash/type 字典：将 Level 0 device task 关联到 node/kernel 名称；
- compact HCCL op-info：collective 名称、数据类型、算法、通信组、元素数量及 host launch 耗时；
- `info.json`：HWTS 频率。

它不会调用 `Profiler.analyse()`，也不会生成 timeline、CSV 或 SQLite。摘要同时保留 device task 类型和名称来源，避免把 framework node 名与硬件 task 类型混为一谈。

## 构建与运行

```bash
cd /root/work/cyf/pretrain/fastanalyse
make
./fast_parse /path/to/segment_000001/rank_0 > fast_summary_v1.json
```

也可原子写入指定文件：

```bash
./fast_parse --output /tmp/fast_summary_v1.json /path/to/rank_0
```

输入可以是 rank segment 目录、其中的 `profiler` 目录，或单个 `PROF_*` 目录。解析器要求仅发现一个 `PROF_*` 和一个 `device_*`，避免误读多个采集段。

## 测试

```bash
make test
```

与已经保留的官方 SQLite 对照单段，或并行校验整个回放集：

```bash
python3 tools/validate_against_sqlite.py /path/to/segment_000001/rank_0
python3 tools/validate_replay.py --jobs 8 /path/to/segments
python3 tools/benchmark_replay.py --jobs 8 /path/to/segments
python3 tools/parse_window.py --output-dir /tmp/window_1 /path/to/segment_000001
```

实施阶段及验收门槛见 [PLAN.md](PLAN.md)，40 段正确性、吞吐和 sidecar 接入结果见 [RESULTS.md](RESULTS.md)。

在线模式已接入 `profiler_sidecar`。推荐关闭耗时较长的官方分析，只保留原始归档和快速摘要：

```bash
bash /root/work/cyf/pretrain/scripts/run_llama7b_fastanalyse.sh
```

每个窗口的输出位于运行目录的 `fast/segment_N/`：`rank_R.json` 是 rank 摘要，`rank_R.result.json` 是调度结果，`window_summary.json` 是跨 Rank 对比。`rank_R.json` 的 `steps[].global_step`/`duration_ns` 给出训练实际 step 及设备耗时；`index_id` 保留设备原始编号。跨 Rank 集中结果在 `window_summary.json` 的 `step_comparisons[].per_rank_ns`。当前 HCCL duration 明确表示 host launch API 耗时，不应解释为 device 通信完成耗时。

不是所有合法采集都会产生 `ts_track`。此时摘要以 `coverage.ts_track_available=false`、`step_trace_available=false`、`step_trace_complete=false` 明确降级，`steps` 为空，但 kernel/device task/HCCL 仍照常解析。有 `ts_track` 文件但只有 task-flip、没有 step marker 时，`ts_track_available=true` 而后两个字段仍为 false。sidecar 的 `--require-complete-step-trace` 会要求 `step_pairs` 等于声明区间并校验连续 `global_step`，用于在线逐-step 场景。

## 前三类诊断的离线回放

`tools/diagnose_windows.py` 对**一次运行**的 `profile/fast` 目录按窗口和 `global_step` 顺序回放。它只读取现有摘要，不修改采集数据，也不触发 NPU。输入指标是封段后的设备逐实际 step 时长 `device_step_duration_ns`；主机 sink 批次墙钟时间和进度心跳尚未接入。它实现三类候选诊断：

- `iteration_latency`：每 rank 相对自身历史的单 step 耗时尖峰，使用滚动中位数、MAD、绝对增量和相对增幅；
- `peer_rank_latency`：同一实际 step 内，同角色 rank 相对其他组员中位数的耗时偏离；组别必须显式提供；
- `rolling_trend`：用短长窗口对比和连续确认输出 `regression` 或 `drift`。

当前 8 卡纯 DP 作业的回放示例：

```bash
python3 tools/diagnose_windows.py /path/to/one_run/profile/fast \
  --output /path/to/replay/diagnosis.json \
  --peer-groups 0,1,2,3,4,5,6,7
```

多个同角色组用分号分隔，例如 `--peer-groups '0,1,2;3,4,5'`。不提供该参数时只运行自身历史和趋势检测；不能把未知并行角色的 rank 自动视为等价。各次运行分别调用命令，不能拼接成同一条时间序列。阈值可用 `--relative-threshold`、`--absolute-ms` 和 `--robust-z-threshold` 调整；默认值分别为 10%、250 ms 和 5，**均为尚未通过长训校准的实验阈值**。

输出是独立的 `fast_diagnosis_replay_v1` JSON：`findings` 包含方法、step、rank、观测值、历史/同伴基线、增量、证据和置信度；`quality_issues` 显式记录缺 rank、缺段、step 断档、trace 不完整和跨摘要不一致；`coverage` 记录有效与预期 rank-step 数。窗口边界的 step 不进入历史基线，若它自身异常仍输出低置信度候选。缺 rank 时跳过不完整组的同伴比较，并重置该 rank 的历史；缺段后重置全部历史。该原型只报告症状，不将耗时最长的 rank 断言为根因，也不执行在线告警抑制。

用真实运行摘要的副本注入五类异常并重放：

```bash
python3 tools/evaluate_diagnosis.py /path/to/one_run/profile/fast \
  --output-dir /path/to/new_evaluation_dir \
  --peer-groups 0,1,2,3,4,5,6,7
```

评估器写出 `evaluation.json` 及每个情景的最小输入副本和完整诊断 JSON。输出目录必须尚不存在，避免覆盖历史证据。实测结果见 [RESULTS.md](RESULTS.md) 的“前三类诊断离线回放”小节。
