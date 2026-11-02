"""Render the completed training comparison; report async backlog explicitly."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import statistics


def read(path):
    return json.loads(path.read_text())


def duration(seconds):
    return f'{int(seconds//60)} 分 {seconds%60:.1f} 秒'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    root = args.output
    data = {r['mode']:r for r in read(root/'summary.json')}
    modes = ['no_profile','offload','default']
    labels = {'no_profile':'不开 profile','offload':'独立进程卸载','default':'默认同步 profiler'}
    for mode in modes:
        row = data[mode]
        assert row['returncode'] == 0 and row['worker_rows'] == 80
        assert len(row['ranks']) == 8 and all(r['last_step'] == 20 for r in row['ranks'])
        hits = read(root/mode/'cache_reuse.json')
        assert len(hits) == 8 and all(h['frontend'] and h['backend'] for h in hits.values())
    baseline = data['no_profile']
    exit_status = read(root/'offload/status_at_training_exit.json')
    current_status = read(root/'offload/status.json')
    ready_by_rank = defaultdict(list)
    archives = []
    for path in (root/'offload/segments').glob('segment_*/rank_*/archive.json'):
        archive = read(path)
        archives.append(archive)
        ready_by_rank[archive['segment']['rank']].append(archive['segment'])
    assert set(ready_by_rank) == set(range(8)) and len(archives) == 40
    for segments in ready_by_rank.values():
        assert [(r['first_step'],r['last_step']) for r in sorted(segments,key=lambda r:r['first_step'])] == [
            (1,4),(5,8),(9,12),(13,16),(17,20)]
    producer = {field:statistics.mean(sum(r[field] for r in segments)
                                    for segments in ready_by_rank.values())
                for field in ('start_seconds','stop_seconds','finalize_seconds')}
    blocking = [r['seconds']['profiler_stop']+r['seconds']['profiler_analyse']
                for r in data['default']['event_phase_seconds'] if 'profiler_analyse' in r['seconds']]
    phase0 = next(r['seconds'] for r in data['default']['event_phase_seconds'] if r['rank']=='rank_0')
    log0 = (root/'default/log/worker/worker_0.log').read_text()
    cann_seconds = float(re.search(r'CANN export done, cost time ([\d.]+) s', log0).group(1))
    inventory = read(root/'default/timeline_inventory.json')
    raw_verification = read(root/'offload/raw_verification.json')
    assert not raw_verification['raw_or_coverage_errors']
    first_parse = read(root/'offload/segments/segment_000000/rank_0/parse_result.json')
    parser_timings = first_parse['timings']
    saved = data['default']['launcher_seconds']-data['offload']['launcher_seconds']
    lines = [
        'Llama2 7B，8 卡，20 step：编译缓存复用后的 profiling 三组对比（2026-09-18）', '',
        f"三组训练均正常完成 20 step。卸载组训练启动器完整退出用时 {duration(data['offload']['launcher_seconds'])}，"
        f"默认同步 profiler 为 {duration(data['default']['launcher_seconds'])}，训练进程提前释放 {duration(saved)}。"
        '卸载组退出时后台仍有解析积压，这个退出时间不表示全部 timeline 已经就绪。', '',
        '| 模式 | worker 日志秒/step | 含回调间隔秒/step | 含间隔相对基线 | samples/s/卡（日志） | 启动器完整退出 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for mode in modes:
        r = data[mode]
        overhead = (r['wall_steady_seconds']/baseline['wall_steady_seconds']-1)*100
        lines.append(f"| {labels[mode]} | {r['logged_steady_seconds']:.4f} | {r['wall_steady_seconds']:.4f} | "
                     f"{overhead:+.2f}% | {r['logged_steady_samples_per_second_per_device']:.5f} | {duration(r['launcher_seconds'])} |")
    lines += ['',
        '数据直接来自每组 `log/worker/worker_0.log` 至 `worker_7.log`。`sink_size=2`，每条 per_step_time 是两步的平均值。'
        '表中正常训练时间取实际第 3–18 step，共 8 条记录/rank，再对 8 rank 求平均。'
        '含回调间隔口径是 `(step18 日志时间戳 − step2 日志时间戳) / 16`，覆盖 loss monitor 计时区间之外的回调和 profiler 启动。'
        '首批次含缓存加载与运行时准备；默认组最后一条日志包含同步解析，均单独处理。全部 20 step 都被采集。', '',
        '本版本日志输出 samples/s/p，没有 FLOPs、TFLOPS 或 MFU 字段，没有将样本吞吐误标为计算量。'
        '全局 batch=64、序列长度=4096，输入工作量为每 step 262,144 token。', '',
        '三组使用同一模型、数据、优化器、batch、20 step 上限和 checkpoint 设置。'
        '两组 profiling 均为 Level0、通信开启、内存专项和 Python 堆栈关闭，init_start_profile=True，包含初始化与缓存加载阶段。'
        '默认组用原框架 stop/analyse 同步路径；卸载组每 4 step 封口一次，由后台写盘和解析。'
        '本次默认组指默认同步处理机制，采集选项与卸载组对齐；它不是原始 YAML 的 profile_memory=True 配置。', '',
        '执行顺序：2 step 缓存预热、2 step 缓存验证、不开 profile、默认 profiler、卸载。'
        '两个预热任务不计入表格，正式任务均保留 checkpoint。将卸载放在最后，避免其后台解析影响另外两组。'
        '每组仅运行一次，百分比是本轮观察值，不是多次试验得到的稳定估计。', '',
        '完整退出时间还受缓存加载、checkpoint 写盘并发和进程清理影响；不能把两组总时间相差约 1.5 秒解释成精确的 profiler 固有开销。'
        '三组末次 loss 分别为 '+', '.join(f"{labels[m]} {data[m]['ranks'][0]['final_loss']:.3f}" for m in modes)+'。'
        '20 个训练 step 中存在一次梯度 overflow，最终 optimizer global_step 为 19。', '',
        '缓存验证：正式三组的 8/8 rank 均同时记录前端缓存命中和后端 kernel graph 缓存构建成功。'
        '缓存验证运行前后 57 个缓存文件没有被重写；首次与复用时 rank 0 首批次分别为 556.916 秒、82.668 秒。'
        f"固定缓存目录：`{read(root/'comparison.json')['cache']}`。入口脚本和模型源码在测量期间保持不变，没有强制绕过缓存一致性检查。", '',
        f"默认组最慢 rank 在最终 stop + analyse 中阻塞 {max(blocking):.3f} 秒。"
        f"rank 0 的 stop 为 {phase0['profiler_stop']:.3f} 秒，analyse 为 {phase0['profiler_analyse']:.3f} 秒；"
        f'其中 CANN 导出日志为 {cann_seconds:.2f} 秒。'
        '剩余 analyse 时间还包含 MindSpore 后续处理等工作，不能全部归因于 JSON 编码。', '',
        f"卸载组每个 rank 累计的 start/stop/finalize 平均为 "
        f"{producer['start_seconds']:.3f}/{producer['stop_seconds']:.3f}/{producer['finalize_seconds']:.3f} 秒。"
        '这些采集器操作仍在训练侧，已经计入训练完成时间；后台独立不等于零资源竞争或零开销。', '',
        f"卸载组在训练退出时：已归档 {exit_status['archived_segments']} 段，"
        f"待解析 {exit_status['parse_pending']} 段（包括正在解析的段），已解析状态 {exit_status['parse_counts']}，"
        f"RAM 原始数据占用 {exit_status['raw_bytes']/2**20:.2f} MiB。",
        f"报告生成时：已归档 {current_status['archived_segments']} 段，待解析 {current_status['parse_pending']} 段，"
        f"已解析状态 {current_status['parse_counts']}。"
        '40 个原始数据段覆盖全部 8 rank 的 step 1–20，无分段缺口或重叠；后台吞吐仍未达到持续实时解析要求。', '',
        f"已归档原始数据 {sum(a['bytes'] for a in archives)/2**30:.3f} GiB，"
        f"归档记录中的复制、摘要计算和文件 fsync 总耗时 {sum(a['archive_seconds'] for a in archives):.3f} 秒。"
        f"已重新读取并核验 {raw_verification['raw_bytes_verified']:,} 字节的原始文件，SHA-256 全部一致。"
        '后台后续仍继续处理，不将未测完的队列排空时间写成已完成结果。', '',
        f"后台首段（rank 0，step 1–4，含初始化）的实际解析总耗时 {first_parse['elapsed_seconds']:.3f} 秒："
        f"输入副本复制 {parser_timings['input_copy_seconds']:.3f} 秒，"
        f"框架准备 {parser_timings['framework_setup_seconds']:.3f} 秒，"
        f"analyse {parser_timings['analyse_seconds']:.3f} 秒，"
        f"JSON 验证 {parser_timings['validate_seconds']:.3f} 秒。"
        '后台仍然每次只解析一个 rank 的一个分段，实际积压证明其吞吐不够；该首段包含初始化，不能将其耗时直接当作所有稳态分段的耗时。', '',
        f"默认组生成了 {len(inventory)} 个 ascend_timeline_display JSON，总计 "
        f"{sum(i['bytes'] for i in inventory)/2**30:.3f} GiB；已检查文件数组首尾边界，未对这些大文件作完整 JSON 验证。"
        'Level0 缺少 op_statistics，标准统计汇总并不完整；不能把启动器返回 0 等同于全部分析产品完整。'
        '卸载后台则对每个已完成的 timeline 做流式 JSON 和 duration 事件验证，结果记录为 timeline_only 或 complete。', '',
        '产物：[逐 rank 汇总](summary.json)、[worker 原始记录 CSV](worker_steps.csv)、[对比图](comparison.png)、'
        '[采集配置差异](offload_vs_default_config.json)、[卸载退出时后台状态](offload/status_at_training_exit.json)、'
        '[卸载当前后台状态](offload/status.json)、[原始数据和覆盖校验](offload/raw_verification.json)。',
    ]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (root/'worker_steps.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1,2,figsize=(11,4.2))
    english = {'no_profile':'No profiler','offload':'Offload','default':'Default profiler'}
    colors = {'no_profile':'#4878a8','offload':'#2a9d8f','default':'#d66a41'}
    for mode in modes:
        points = defaultdict(list)
        for r in rows:
            if r['mode'] == mode and 4 <= int(r['step']) <= 18:
                points[int(r['step'])].append(float(r['logged_seconds_per_step']))
        xs = sorted(points)
        axes[0].plot(xs,[statistics.mean(points[x]) for x in xs],marker='o',
                     label=english[mode],color=colors[mode])
    axes[0].set(xlabel='Sink batch ending at step (2 steps per point)',ylabel='Worker log seconds / step',
                title='Training steps: average of 8 ranks')
    axes[0].legend();axes[0].grid(alpha=.2)
    bars = axes[1].bar([english[m] for m in modes],[data[m]['launcher_seconds']/60 for m in modes],
                       color=[colors[m] for m in modes])
    axes[1].bar_label(bars,fmt='%.2f min',padding=3)
    axes[1].set(ylabel='Minutes',title='Launcher exit (offload parsing continues)')
    axes[1].set_ylim(0,max(data[m]['launcher_seconds']/60 for m in modes)*1.15)
    fig.tight_layout()
    fig.savefig(root/'comparison.png',dpi=160)
    fig.savefig(root/'comparison.svg')
    plt.close(fig)
    print(root/'report.md')


if __name__ == '__main__':
    main()
