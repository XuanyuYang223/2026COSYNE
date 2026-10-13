from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .common import ROOT, config, sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def build_report():
    cfg = config()
    status = read_json(ROOT / 'results' / 'status.json')
    manifest = read_json(ROOT / 'split_manifest.json')
    env = read_json(ROOT / 'results' / 'environment.json')
    budget = read_json(ROOT / 'results' / 'budget.json') if (ROOT / 'results' / 'budget.json').exists() else {'gpu_seconds': 0}
    lines = [f"# 执行报告：{status['status']}", '',
             '日期：2026-10-12（America/Los_Angeles）。方案 v1.0。', '',
             '研究问题：相同完整记忆内容、不同形成历史的候选记忆状态，其几何相似度能否预测保留接收方控制状态后的功能可互换性？', '',
             '本目录是独立实验实现；没有修改既有实验或论文结论，没有推送或发布。', '',
             '## 实际完成范围与门槛', '',
             '| 项目 | 状态 |', '|---|---|',
             '| 数组真值与独立矩阵评价；n=6/7 轨道复核 | 通过 |',
             '| 轨道级 T/V/G/B 划分、内容匹配和时间检查 | 通过 |',
             '| 控制模块不接收置换或 m；状态缓存内容隔离 | 通过 |',
             '| sham、全状态与不中断续跑实现对照 | 通过 |',
             '| float64 CKA 双公式、尺度/同行重排与已知矩阵测试 | 通过 |',
             '| LOSO/历史标签隔离与折内缺失值处理 | 合成数据实现测试通过；不是研究结果 |',
             f"| GPU | {env.get('gpu', '不可用')} |", '',
             '独立复核：S₇ 的 5,040 个置换形成 694 条轨道；S₆ 为 115 条。轨道大小为 2、4 或 8。', '',
             '65 个训练词覆盖 7 种净变换；16 个四 A/B 测试词全部是新序列，仅 ABAB、BABA 引入第 8 种净变换。这是数据生成的数学性质，不说明网络学会群结构。', '',
             '| 集合 | 轨道数 | 置换数 |', '|---|---:|---:|']
    for split in ('T', 'V', 'G', 'B'):
        row = manifest['splits'][split]
        lines.append(f"| {split} | {row['orbit_count']} | {row['permutation_count']} |")
    lines += ['', f"正式行为子集预先固定为 {manifest['behavior_permutation_count']} 个置换、{len(manifest['behavior_orbit_ids'])} 条完整轨道；其余 B 内容保留。", '',
              f"split_seed={cfg['split_seed']}；按规范化轨道列表打乱，T/V/G 向下取整，余数归 B。文件 SHA-256：`{sha256(ROOT / 'split_manifest.json')}`。", '',
              '所有输入在刺激阶段结束后均不再提供置换；查询在交换时点之后才出现。网络仅使用最终查询交叉熵，未接受中间监督、交换训练或几何损失。', '',
              '## Pilot：原任务先于科学分析', '',
              '| memory_dim | 更新数 | 最佳步数 | 固定 V 准确率 | V CE (nat) | 训练秒数 | 峰值 torch 分配显存 (MiB) | ≥97% |',
              '|---:|---:|---:|---:|---:|---:|---:|---|']
    pilots = []
    for name in status.get('pilot_runs', []):
        run = ROOT / 'runs' / name
        summary = read_json(run / 'training_summary.json')
        curve = read_csv(run / 'training.csv')
        peak = max(float(r['peak_gpu_mb']) for r in curve)
        best = summary['best_validation']
        lines.append(f"| {summary['memory_dim']} | {summary['updates']} | {best['step']} | {100*best['accuracy']:.4f}% | {best['cross_entropy']:.6f} | {summary['training_seconds']:.3f} | {peak:.2f} | {'通过' if summary['validation_target_passed'] else '未通过'} |")
        pilots.append((run, summary))
    lines += ['', 'pilot seed=0；同一固定验证集包含 8,192 个试次。检查点只按验证准确率、再按交叉熵选择；没有根据交换结果选择容量或随机种子。', '',
              '每种容量完整运行 20,000 次 Adam 更新；未进行额外容量、初始化、课程训练或超参数搜索。', '',
              '![Pilot 训练曲线](results/figures/pilot_training.png)', '']
    if status['status'] == 'BLOCKED' and status['reason'] == 'both_prespecified_pilots_failed_task_gate':
        lines += ['默认与唯一备用配置均未达到原任务门槛，已按方案第十二/十三节停止正式运行。', '',
                  '**H1 无有效结论；H2、H3 未检验。没有正式 G/B 状态测量，没有正式六种子标签、预测误差、显著性或科学反例。**', '',
                  '正式种子 101、102、103、104、105、106 均未运行，原因相同：pilot 任务学习失败。', '',
                  '## 失败诊断：仅使用 V', '',
                  '机会水平为 1/7=14.2857%；均匀输出交叉熵 ln(7)=1.945910 nat。两种配置的验证结果接近此水平。参数确实更新，梯度有限；失败不等于优化器没有执行。', '',
                  '| memory_dim | 移除刺激后回答改变率 | 打乱刺激后回答改变率 | 移除刺激最大 logit 差 | 第 8 步内容影响 L2 | 第 37 步内容影响 L2 |',
                  '|---:|---:|---:|---:|---:|---:|']
        for run, summary in pilots:
            diag = read_json(run / 'diagnostics.json')
            perturb = diag['stimulus_perturbations']
            influence = diag['content_sensitivity']['selected']['mean_memory_l2_by_step']
            lines.append(f"| {summary['memory_dim']} | {100*perturb['zero']['prediction_change_rate']:.4f}% | {100*perturb['roll']['prediction_change_rate']:.4f}% | {perturb['zero']['max_logit_difference']:.3g} | {influence[7]:.6g} | {influence[36]:.6g} |")
        lines += ['', '此处内容影响是同一 V 试次、控制输入完全相同时，原始刺激与全零刺激对应 m 的 L2 差。第 8 步是刺激结束，第 37 步是 d=0 的最终查询。最终输出对初始内容几乎不敏感，和内容信息在保持/更新期间衰减的解释相符；这项诊断不能单独确定因果性的训练失败机制。', '',
                  '![内容影响随时间衰减](results/figures/pilot_content_sensitivity.png)', '',
                  'V 的前 52 条轨道用于几何速度和探针诊断（392 个内容）；后半部分的前若干完整轨道用于干预诊断（34 个内容、6 条轨道）。两部分轨道不相交。探针只在 T 上拟合 alpha=1 ridge；没有用 V/G/B 拟合。', '',
                  '| memory_dim | sham 最大 logp 误差 | 全状态最大 logp 误差 | V 干预最差原始层准确率 | V 探针逐项目准确率 | V 探针完整置换准确率 |',
                  '|---:|---:|---:|---:|---:|---:|']
        for run, summary in pilots:
            behavior = read_json(run / 'analysis' / 'intervention_summary.json')
            geometry = read_json(run / 'analysis' / 'geometry_summary.json')
            lines.append(f"| {summary['memory_dim']} | {behavior['sham_max_logp_error']:.3g} | {behavior['full_max_logp_error']:.3g} | {100*behavior['native_min_stratum_accuracy']:.4f}% | {100*np.mean(geometry['probe_item_accuracy']):.4f}% | {100*np.mean(geometry['probe_permutation_accuracy']):.4f}% |")
        lines += ['', '上述缓存包含自然形成状态、逐槽真值轨迹、供体初始置换，以及全部查询的原始/sham/匹配/错误内容/同上下文全状态输出。错误内容来自固定跨轨道无自匹配置换，记录接收方答案与供体反事实答案。全部原始数据均保留，但只作实现诊断。', '',
                  '**诊断中的匹配与错误内容替换表现相同；新增错误接近零、概率损害极小不能称为学会内容后的兼容性上限效应。原网络已经无法完成任务，没有辨识 H1/H2/H3 的条件。**', '',
                  '未计算 pilot 的相似度—损害相关系数或预测模型；未挑选高 CKA 反例。原任务未通过，所以四步未见词扩展也未运行。', '',
                  '## 计算成本与进入正式运行的决定', '']
        for run, summary in pilots:
            intervention = read_json(run / 'analysis' / 'intervention_summary.json')
            timing = run / 'analysis' / 'geometry_timing.json'
            geom_seconds = read_json(timing)['wall_seconds'] if timing.exists() else None
            train_speed = summary['updates']/summary['training_seconds']
            estimated = 6*(summary['training_seconds'] + intervention['elapsed_seconds']*manifest['behavior_permutation_count']/intervention['contents'] + 120)
            geom_text = f"{geom_seconds:.3f} 秒" if geom_seconds is not None else '未单独计时'
            geometry = read_json(run / 'analysis' / 'geometry_summary.json')
            lines += [f"- {summary['memory_dim']} 维：训练约 {train_speed:.1f} 更新/秒；完整 V 干预评价 {intervention['elapsed_seconds']:.3f} 秒（约 {918*2*intervention['contents']*14/intervention['elapsed_seconds']:.0f} 个主匹配查询/秒，实际计时还含其他对照及保存），V 几何/初始化/探针诊断 {geom_text}。两种评价峰值 torch 分配显存分别为 {intervention.get('peak_gpu_mb', 0):.2f}/{geometry.get('peak_gpu_mb', 0):.2f} MiB。按内容数线性外推并为每个正式种子保留 120 秒几何余量，六种子训练与主评价估计 {estimated/3600:.3f} GPU 小时；不含 CPU bootstrap 与文件写入的机器差异。"]
        lines += ['', '上述估算只检验预算可行性；原任务门槛失败已经足以阻止正式运行。GPU 条件与预算不是本轮阻塞原因。', '']
    elif status['status'] in ('COMPLETED', 'EXPLORATORY'):
        lines += formal_text(cfg, status)
    else:
        lines += [f"阻塞原因：`{status['reason']}`。未将 smoke test 作为研究结果。", '']
    lines += [f"累计 GPU 操作墙钟记账：{budget['gpu_seconds']:.3f} 秒（{budget['gpu_seconds']/3600:.4f} 小时），预算上限 12 小时。记账包含训练、验证、缓存及 GPU 诊断的外围开销；曾短暂重叠的训练/诊断时间各自记账，因此是保守成本估计。", '',
              '## 可复现产物与限制', '',
              '- `config.yaml`、`split_manifest.json` 与 `results/frozen_plan.json` 保存方案、轨道、固定 B 子集和源码散列。',
              '- `runs/pilot_128/`、`runs/pilot_256/` 各含 `initial.pt`、`best.pt`、`latest.pt`、训练曲线、完整优化器和可恢复 Python/NumPy/Torch/CUDA 随机数状态。',
              '- `analysis/raw_d*_S*.npz` 按供体、接收方、内容、查询类型、查询项目保存输出与真值；`orbit_blocks.npz` 保存每条轨道、每个方向与查询类型的统计和查询计数。',
              '- `directions.csv`、`native_strata.csv`、`labels.csv` 保留方向/上下文/S/d 分层。原始数据足以重算指标，不能把共享状态的查询当成独立 CKA 样本。',
              '- 预测实现固定 alpha=1 ridge。外层留出模型种子，拟合只用 91 个不涉及 AB/BA 的配对；未见 62 对含跨上下文 32 对。同一代码包含训练折内预处理、初始化 CKA 检查及种子/行为轨道层级重采样。实现测试使用合成数据，其数值未写作研究结果。', '',
              '候选 m/c 模块由架构预先指定；单向控制并不能证明天然的独立记忆脑区。数学任务的精确内容标签不等于网络学会代数结构，也不能外推真实神经系统。', '',
              '本轮没有检索更新相关工作或判断原创性。方案中的文献链接仅作原始定位，论文提交前仍需核对版本与近邻工作。', '',
              '![任务与干预示意](results/figures/01_task.png)', '',
              f"**最终状态：{status['status']}。**"]
    (ROOT / 'report.md').write_text('\n'.join(lines) + '\n')
    # Hash every delivered artifact (excluding this index itself and transient caches).
    artifacts = {}
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.name != 'artifact_manifest.json':
            artifacts[str(path.relative_to(ROOT))] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    write_json(ROOT / 'results' / 'artifact_manifest.json', artifacts)


def formal_text(cfg, status):
    lines = ['## 正式六种子与预测结果', '',
             '| 种子 | 最佳验证准确率 | 主行为最差层准确率 | 原任务门槛 |', '|---:|---:|---:|---|']
    ceiling = []
    for seed in cfg['seeds']:
        run = ROOT / 'runs' / f'seed_{seed}'
        training = read_json(run / 'training_summary.json')
        behavior = read_json(run / 'analysis' / 'intervention_summary.json')
        passed = training['validation_target_passed'] and behavior['native_strata_passed']
        ceiling.append(behavior['compatibility_ceiling'])
        lines.append(f"| {seed} | {100*training['best_validation']['accuracy']:.3f}% | {100*behavior['native_min_stratum_accuracy']:.3f}% | {'通过' if passed else '原任务掌握不足，保留此种子'} |")
    lines += ['', '主检验同时留出种子与涉及 AB/BA 的历史配对，仅看跨上下文 32 对；其他三个泛化分层均另存 scores.csv。', '',
              '| 种子 | 模型 | MAE (nat) | RMSE (nat) | 对 M0 的 MAE 改善 | 对 M0 的 RMSE 改善 |',
              '|---|---|---:|---:|---:|---:|']
    score_root = ROOT / 'results' / 'prediction'
    scores = read_csv(score_root / 'scores.csv')
    for row in scores:
        if row['segment'] == 'unseen_cross':
            lines.append(f"| {row['seed']} | {row['model']} | {float(row['mae']):.6g} | {float(row['rmse']):.6g} | {float(row['mae_improvement']):.6g} | {float(row['rmse_improvement']):.6g} |")
    intervals = read_json(score_root / 'bootstrap_intervals.json')
    primary = next(r for r in intervals if r['segment'] == 'unseen_cross' and r['model'] == 'M_CKA')
    lines += ['', f"CKA 相对 M0 的主测试 MAE 改善 95% 层级 bootstrap 区间：{primary['mae_improvement_95ci']}；RMSE 改善区间：{primary['rmse_improvement_95ci']}。", '',
              '重采样以模型种子和 B 置换轨道为块，折内重新拟合预测器，保留方向、查询类型及条件配对的关联；6 个种子仅代表这个有限模型集合，不是普遍定律。', '',
              'M_init_CKA 相对于 M_init 的额外改善用于判断训练后 CKA 是否超过初始架构/输入几何；不能把初始网络的低行为准确率纳入主交换标签。', '']
    if status['status'] == 'EXPLORATORY':
        lines += ['至少一个正式种子未通过原任务门槛。全部保留，结果仅为探索性，不作强机制结论。', '']
    if all(ceiling):
        lines += ['全部种子满足预设新增错误/损害 IQR 的上限效应规则。任务掌握通过时，这说明交换普遍可行；它不能证明 CKA 有排序预测力。', '']
    rows = read_csv(score_root / 'full_distribution.csv')
    variance = np.var([float(r['loss_positive']) for r in rows])
    if variance < cfg['analysis']['negligible_variance']:
        lines += ['行为标签方差接近零，本设计的预测问题缺乏可识别性。相关系数在退化层内保存为 null。', '']
    lines += ['高 CKA/高新增错误反例按每个 seed×d×S×上下文类别内的两个第 75 分位数及 R≥0.5% 固定规则选取；完整候选列表与全部分布一同保存，不选择个别支持观点的查询。', '',
              '概率损害 L 与新增错误 R 分别报告；有概率损害但没有新增错误时，仅能表述为概率损害。', '',
              '![几何与损害](results/figures/02_geometry_harm.png)', '',
              '![留出预测误差](results/figures/03_prediction_errors.png)', '',
              '![保持间隔与行为](results/figures/04_delay_behavior.png)', '']
    return lines


if __name__ == '__main__':
    build_report()
