# ARGUS Ascend Profiling 工具

本目录包含 Llama2-7B 在 8×Ascend 910B2 上使用的训练 profiling、旁路归档、快速解析、在线诊断和可视化代码。项目将 profiling 原始数据先写入 `/dev/shm`，再由独立 CPU 进程完成持久化和解析，避免在训练 Rank 中同步执行耗时的 `analyse()`。

## 首轮上传结构

```text
pretrain/
├── scripts/                    # 训练与 sidecar 启动入口
│   ├── run_llama7b.sh
│   ├── run_llama7b_sidecar.sh
│   └── run_llama7b_fastanalyse.sh
├── configs/                    # 本项目维护的训练配置
│   ├── README.md
│   └── llama2/
│       ├── pretrain_llama2_7b_20steps.yaml
│       └── pretrain_llama2_7b_sidecar_20steps.yaml
├── profiler_sidecar/           # 采集、归档、诊断和仪表盘主体
│   ├── launch.py
│   ├── training_entry.py
│   ├── callback.py
│   ├── worker.py
│   ├── fast_worker.py
│   ├── diagnosis_worker.py
│   ├── dashboard_*.py
│   ├── dashboard.html
│   ├── tests/
│   └── research/
├── fastanalyse/                # C++ 原始 Trace 快速解析器及聚合工具
│   ├── src/
│   ├── tools/
│   ├── tests/
│   └── Makefile
├── benchmarks/                 # benchmark 入口和结果汇总脚本
│   ├── benchmark_entry.py
│   ├── benchmark_llama7b.py
│   └── summarize_benchmark.py
├── .gitignore
└── README.md
```

首轮不上传以下本地运行内容：

- `pyspy_overhead/`
- `compile_cache/`
- `profiler_sidecar/runs/`
- 历史 benchmark Trace 和训练输出
- 快速解析的历史 benchmark/validation JSON
- `__pycache__/`、日志和临时文件
- 编译生成的 `fastanalyse/fast_parse`
- 数据集、模型权重和 checkpoint

这些内容已在 `.gitignore` 中排除。`fast_parse` 应从 `fastanalyse/src/fast_parse.cpp` 在目标机器重新构建。

MindFormers 不作为普通源码复制到本仓库。当前验证版本的包版本为 `1.3.2`，源码提交为 `f90b5b9cfa3330f562d59fc19a8a3d326699c5c0`；建议通过独立 fork 或 Git submodule 固定该提交。训练数据、Tokenizer 和缓存路径见 `configs/README.md`。

## 执行链

```text
scripts/run_llama7b_fastanalyse.sh
              │
              ▼
scripts/run_llama7b_sidecar.sh
              │
              ▼
profiler_sidecar/launch.py
   │
   ├── 启动 worker.py（独立 CPU sidecar）
   ├── 启动 dashboard_server.py（可选独立服务）
   └── 通过 MindFormers msrun_launcher.sh 启动 8 个训练 Rank
                               │
                               ▼
                     training_entry.py
                               │
                  安装 SidecarProfileMonitor
                               │
                               ▼
                    MindFormers 正常训练
                               │
                 callback.py 按 step 窗口采集
                               │
                               ▼
        /dev/shm/.../segment_N/rank_R/ready.json
                               │
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
      worker.py 持久化                    fast_worker.py 调度
      segments/ 原始归档                  fastanalyse/fast_parse
                                                 │
                                                 ▼
                                     fast/ 跨 Rank 窗口摘要
                                                 │
                                                 ▼
                                      diagnosis_worker.py
                                                 │
                                                 ▼
                                      diagnosis/ 在线诊断
                                                 │
                                                 ▼
                           dashboard_data.py + dashboard_server.py
                                                 │
                                                 ▼
                                    在线页面 / DASHBOARD.html
```

职责边界：

- `launch.py` 管理整次运行的配置、目录、编译缓存、后台进程和分布式训练启动。
- `training_entry.py` 在每个训练 Rank 内安装 sidecar callback，然后执行 MindFormers 的 `run_mindformer.py`。
- `callback.py` 只负责训练进程内的窗口采集、封口和 `ready.json` 发布，不执行官方 `analyse()`。
- `worker.py` 在训练进程外完成原始数据归档，并调度官方解析、快速解析和在线诊断。
- `fast_worker.py` 调用 `fastanalyse/fast_parse` 并生成 Rank 摘要和跨 Rank 窗口摘要。
- 仪表盘只读取已经原子发布的 `fast/`、`diagnosis/` 和 `status.json`，不进入训练进程，也不读取尚未封口的数据。

窗口边界处的 `Profiler.stop()` 和原生 `finalize()` 仍发生在训练进程内；磁盘复制、`fsync`、快速解析、官方 `analyse()` 和可视化均在训练进程外执行。因此该方案降低了解析阻塞，但不宣称 profiling 零开销。

## 启动入口

普通 MindFormers profiler：

```bash
bash /root/work/cyf/pretrain/scripts/run_llama7b.sh
```

Sidecar 加官方离线解析：

```bash
bash /root/work/cyf/pretrain/scripts/run_llama7b_sidecar.sh
```

推荐的快速解析与仪表盘入口：

```bash
bash /root/work/cyf/pretrain/scripts/run_llama7b_fastanalyse.sh
```

快速入口在 sidecar 基础上启用 `--fast-analyse`、关闭官方 `analyse()`、要求完整 step trace，并自动启动只读仪表盘。详细参数、恢复方式和验证结果见 `profiler_sidecar/README.md` 与 `fastanalyse/RESULTS.md`。

## 主要运行产物

一次完整运行的输出目录通常包含：

```text
run/
├── sidecar_config.json         # 本次 sidecar 配置
├── train_config.yaml           # 实际训练配置
├── training_run.json           # 训练启动和退出信息
├── status.json                 # sidecar 最终状态
├── segments/                   # 持久化原始 profiling 数据
├── fast/                       # 快速解析与跨 Rank 摘要
├── diagnosis/                  # 在线诊断和数据质量事件
├── logs/                       # MindFormers 日志
├── msrun_log/                  # 分布式启动日志
└── DASHBOARD.html              # 离线仪表盘快照
```

运行产物用于本地验收和复核，不应直接提交到源码仓库。
