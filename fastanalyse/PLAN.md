# FastAnalyse 实施计划

目标：从已经封口的 Ascend 原始 profiling segment 直接生成紧凑摘要；在线处理速度持续快于训练窗口产生速度，并避免 `Profiler.analyse()`、timeline 和全量 CSV/SQLite。

## 流程

1. **离线原型**：固定 CANN 8.0.RC3 数据格式，直接解析 `ts_track` 和 `stars_soc`，先输出 step 边界、device task 配对、耗时与覆盖率。
2. **名称关联**：按需解析 `task_track`、API/node/context/HCCL 元数据，将 device task 关联为算子/kernel；无法命名的记录保留稳定指纹，不静默丢弃。
3. **紧凑聚合**：按 step、名称/指纹、stream、task type 输出 count/total/p50/p95/p99/max、top-K、结构 hash 和解析质量指标。
4. **回放校验**：用已有 40 个 rank-segment 与官方 `step_trace`、`task_time`、`op_summary` 对照；所有差异量化输出。
5. **吞吐基准**：8 个常驻 worker 并行回放。目标 8 Rank 窗口 p99 ≤ 5 s；硬门槛为小于实测训练窗口间隔的 50%，且队列不积压。
6. **Sidecar 接入**：`ready.json` 出现后，快速解析与持久化归档并行读取封口目录；只有归档和解析均完成后才释放 tmpfs。完整官方分析仅作离线兜底。
7. **在线 A/B**：验证端到端延迟、队列深度、tmpfs 占用和训练吞吐影响，达标后再启用在线诊断。

当前进度：第 1–6 步已经实现并通过 40 个保留样本及 sidecar 生命周期测试；第 7 步已完成 2 卡真实 MatMul + AllReduce smoke 的在线 A/B，目标 Llama7B 8 卡多次 A/B 尚待执行。量化结果见 [RESULTS.md](RESULTS.md)。

## 实现原则

- C++17 负责定长二进制扫描、事件配对和流式统计；Python只负责调度、跨 Rank 聚合与报告。
- 原始数据只读、版本/schema 强校验、失败闭合；摘要必须报告配对率、命名率和 step 归属率。
- 先实现和验证独立 CLI，再修改现有 sidecar；未通过正确性与吞吐门槛前不进入训练链路。
