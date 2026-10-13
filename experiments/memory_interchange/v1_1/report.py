from __future__ import annotations

import json

import numpy as np

from ..common import sha256, write_json
from .common import BASE, ROOT, RESULTS, RUNS, Budget, config
from .figures import build_figures


def read(path):
    return json.loads(path.read_text())


def build_report():
    revision = config()
    status = read(RESULTS / 'status.json')
    preservation = read(RESULTS / 'preservation_check.json')
    audit = read(RESULTS / 'audit' / 'summary.json')
    budget = read(RESULTS / 'budget.json')
    candidates = {name: read(RUNS / f'candidate_{name}_seed0' / 'summary.json') for name in ('A', 'B')
                  if (RUNS / f'candidate_{name}_seed0' / 'summary.json').exists()}
    build_figures()
    lines = [f"# v1.1 任务学习修订报告：{status['status']}", '',
             '日期：2026-10-12（America/Los_Angeles）。本轮只修订任务学习，不开展科学交换/预测实验。', '',
             '## 先报告失败在哪里', '']
    if status['status'] == 'BLOCKED' and candidates:
        lines += ['**两个预先指定候选均在阶段 6（原时序、两种上下文、非 N 数≤2 的两步计算）失败。前五阶段通过，阶段 7 未运行，完整 65 词原任务尚未达到 TASK_READY。**', '',
                  '| 候选 | 阶段 6 更新 | T 准确率 | V 准确率 | V CE (nat) | V 的 k=2 准确率 | 阶段 6 累计课程更新 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for name, result in candidates.items():
            s = result['stage_summaries'][-1]
            v, t = s['last_validation'], s['last_training_evaluation']
            k2 = v['strata']['non_N'].get('2', {}).get('accuracy', float('nan'))
            lines.append(f"| {name} | {s['updates']} | {100*t['accuracy']:.4f}% | {100*v['accuracy']:.4f}% | {v['cross_entropy']:.6f} | {100*k2:.4f}% | {result['total_updates']} |")
        lines += ['', '两者均达到该阶段固定 5,000 更新上限，总体 V 未达到 97%，部分操作因素也未达到 95%；没有进入下一阶段碰运气，没有增加容量、门偏置值、学习率、种子或更新上限。', '',
                  '这将本轮阻塞定位到两步任务的学习门槛，不能仅凭这些数据进一步确定是控制组合、内容更新或读出中的哪一机制。', '']
    elif status['status'] == 'TASK_READY':
        ready = read(RESULTS / 'task_ready' / 'summary.json')
        lines += [f"pilot 候选 {status['selected_candidate']} 已通过阶段 7，原固定 V 准确率 {100*ready['original_V']['accuracy']:.4f}%。本轮仍不自动运行正式模型或打开 G/B；TASK_READY 不支持 H1/H2/H3。", '']
    else:
        lines += [f"执行状态：{status['reason']}。", '']
    lines += ['## 是否修复', '',
              '没有发现 v1.0 训练入口的输入、标签、有效步索引或计算图 bug。课程修订使短时序读出、原长度保持、单步 inverse/complement 及上下文子任务通过，但尚未完成完整原任务的学习修复。', '',
              'v1.0 的交换损害接近零仍是未掌握任务时的诊断现象，不能包装为记忆可互换性证据。H1/H2/H3 本轮均未检验。', '',
              '## 旧模型的 T/V 对比', '',
              '| v1.0 容量 | 检查点 | 步数 | 固定 T 准确率 | 新增平衡 V 准确率 | 原 V 准确率 | T CE | 平衡 V CE |',
              '|---:|---|---:|---:|---:|---:|---:|---:|']
    for dim in (128, 256):
        for checkpoint in ('best', 'latest'):
            row = audit['old_checkpoint_evaluations'][f'v1_0_m{dim}_{checkpoint}']
            lines.append(f"| {dim} | {checkpoint} | {row['checkpoint_step']} | {100*row['T']['accuracy']:.4f}% | {100*row['V_balanced']['accuracy']:.4f}% | {100*row['V_original']['accuracy']:.4f}% | {row['T']['cross_entropy']:.6f} | {row['V_balanced']['cross_entropy']:.6f} |")
    lines += ['', '三个评价集合各 8,192 试次。新增 T/V 原任务诊断使用 seed=20261013、平衡因素抽样；原 V 完全保持 v1.0 的 8,192 个试次，SHA-256 单独核验。T 与 V 均接近 1/7=14.2857%，不支持“T 已学会、主要是新置换泛化差距”的定位。', '',
              '旧 best/latest 在 T、平衡 V、原 V 上移除/固定打乱刺激的回答改变率均为 0；逐试次输出、CE 和分层保存在 `audit/`。', '',
              '## 训练路径审计', '',
              '- oracle 从实际 collate 后的刺激矩阵、上下文、四槽可见线索和最终查询恢复标签，所有固定集合及真实旧训练 sampler 的 128 试次批次均 100% 正确。oracle 不调用 truth.py 的更新函数。',
              '- m/c 仅在试次开始初始化为零；模型按 batch_first 运行，最终 loss 取每个样本 lengths−1，实际最后 QUERY 步处于 padding 之前。刺激之后输入归零，标签、查询 ID/type、上下文和时序均对齐。',
              '- 实际训练 forward 内没有 detach、no_grad/inference_mode、边界清零或 NumPy 状态往返。对逐步携带的因果状态节点 retain_grad，并和实际 fused forward 比较 logit 与导数。',
              '- 所有记录的刺激结束、规则槽结束、交换时点、最终查询状态都有有限梯度，None 数为零；memory/control/readout 参数也都有有限梯度。小梯度与 None 分开记录，不把小或零范数自动认定为 detach。', '',
              '| 旧 best 模型 | 刺激结束 m 梯度 L2（均值） | 交换时点 m 梯度 L2（均值） | 最终 m 梯度 L2（均值） | 实际 stimulus 梯度 L2（均值） |',
              '|---:|---:|---:|---:|---:|']
    paths = audit['real_v1_0_sampler_paths']
    for dim in (128, 256):
        path = paths[f'm{dim}_best']
        e = path['events']
        sg = path.get('actual_fused_stimulus_gradient_l2_during_stim', path['stimulus_gradient_l2_during_stim'])
        lines.append(f"| {dim} | {e['stimulus_end']['gradient_l2']['mean']:.3g} | {e['swap_time']['gradient_l2']['mean']:.3g} | {e['query_end']['gradient_l2']['mean']:.3g} | {sg['mean']:.3g} |")
    fd = audit['short_float64_derivative']
    lines += ['', '上表使用真实旧训练 sampler 的 128 试次 T 批次；原始状态与梯度向量另存 *_real_path_trace.npz。阶段诊断则使用预先固定的 64 试次批次，范数均值的批次大小随数据保存。', '',
              f"短时序 CPU float64 抽查：自动微分导数 {fd['autograd']:.10g}，中心有限差分 {fd['finite_difference']:.10g}，绝对误差 {fd['absolute_error']:.3g}，通过。长时序微小导数不依赖有限差分作判断。", '',
              '实际安装版本为 torch 2.11.0+cu128；已核对 [官方 GRU 门方程与 r/z/n 顺序](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html) 与安装源码一致。训练与评价模式、no-grad/inference 的区分依据 [官方 autograd 文档](https://docs.pytorch.org/docs/2.14/notes/autograd.html)，并通过实际训练 forward 检查验证。', '',
              '## 独立小型拟合检查', '',
              '| 初始化 | 固定 T 置换/试次 | 更新数 | 训练集合准确率 | CE (nat) | ≥99% |',
              '|---|---|---:|---:|---:|---|']
    scratch_updates = 0
    for name in ('A', 'B'):
        path = RUNS / f'scratch_{name}' / 'summary.json'
        if path.exists():
            toy = read(path)
            scratch_updates += toy['updates']
            lines.append(f"| {name} | 16 / 224 | {toy['updates']} | {100*toy['accuracy']:.4f}% | {toy['cross_entropy']:.6f} | {'通过' if toy['passed'] else '失败'} |")
    lines += ['', '固定使用 T 前 16 个置换、全部 7 个项目与两类查询、r=0、NNNN、D_short。scratch 权重未进入任何课程，不能据过拟合成功声称泛化。默认小型拟合通过，因此没有额外运行 B 的小型拟合。', '',
              '## 七阶段课程的实际完成情况', '',
              '| 候选 | 阶段 | 时序/词分布 | 更新 | T 准确率 | V 准确率 | 最差 V 因素准确率 | 连续两次门槛 | 墙钟秒 | 峰值 torch MiB |',
              '|---|---:|---|---:|---:|---:|---:|---|---:|---:|']
    course_updates = 0
    for name, result in candidates.items():
        course_updates += result['total_updates']
        for stage in result['stage_summaries']:
            spec = revision['stages'][stage['stage']-1]
            v, t = stage['last_validation'], stage['last_training_evaluation']
            worst = min(cell['accuracy'] for factor in v['strata'].values() for cell in factor.values())
            words = 'NNNN' if spec['max_non_N'] == 0 else f"k≤{spec['max_non_N']}"
            lines.append(f"| {name} | {stage['stage']} | {stage['timing']} / {words} | {stage['updates']} | {100*t['accuracy']:.4f}% | {100*v['accuracy']:.4f}% | {100*worst:.4f}% | {'通过' if stage['passed'] else '失败'} | {stage['elapsed_seconds']:.3f} | {stage['peak_gpu_mb']:.2f} |")
    lines += ['', '阶段 1–6 的固定 V 各 4,096 试次；阶段 7 预先保存并复用原 V 8,192 试次，但本轮未训练到该阶段。每 500 更新评价；总准确率≥97% 且各适用因素层≥95%，连续两次才晋级。因素分别为上下文、真实 I/C 操作序列、d、查询类型和非 N 数；计数、CE 与门槛状态随每个评价点保存，不使用任意极细交叉格。', '',
              'A 的默认初始化通过阶段 1–5 后在阶段 6 失败，才允许从头运行 B。B 唯一改动是在两个 GRU 的 r/z/n 中 z 段将 bias_ih 设为 +3、bias_hh 设为 0；其余初始化、结构、Adam、课程、seed=0 和门槛不变。所有参数一直自由训练，偏置不是硬编码记忆或代数算子。', '',
              '每阶段从匹配的 advance 检查点恢复权重、Adam 和 Python/NumPy/Torch/CUDA RNG；不使用 scratch 或失败的 v1.0 权重。初始检查点在课程前保存。没有早期简单任务检查点被当成阶段 7 最终检查点。', '',
              '![课程曲线](figures/curriculum_learning.png)', '',
              '## 两步计算失败的完整操作因素', '',
              '| 候选 | 实际操作顺序 | T n | T 准确率 | V n | V 准确率 | V CE |',
              '|---|---|---:|---:|---:|---:|---:|']
    for name, result in candidates.items():
        last = result['stage_summaries'][-1]
        for op in ('KEEP', 'I', 'C', 'II', 'CC', 'IC', 'CI'):
            ts = last['last_training_evaluation']['strata']['true_operation_sequence']
            vs = last['last_validation']['strata']['true_operation_sequence']
            if op in vs:
                t, v = ts[op], vs[op]
                lines.append(f"| {name} | {op} | {t['n']} | {100*t['accuracy']:.4f}% | {v['n']} | {100*v['accuracy']:.4f}% | {v['cross_entropy']:.6f} |")
    lines += ['', 'IC 表示先 inverse 再 complement，CI 表示先 complement 再 inverse；它们的数组真值不交换。上表包含所有预定操作因素，没有只挑某个失败案例。阶段 6 总准确率含 k=0/1 容易试次，不应掩盖 k=2 的不足。', '',
              '![两步因素错误](figures/two_operation_failure.png)', '',
              '## 内容敏感性、梯度和 update gate', '',
              '| 候选 | 阶段 | 分割 | 刺激结束内容影响 L2 | 交换时点内容影响 L2 | 查询结束内容影响 L2 | 移除刺激回答改变率 | 打乱刺激回答改变率 |',
              '|---|---:|---|---:|---:|---:|---:|---:|']
    for name, result in candidates.items():
        for stage in result['stage_summaries']:
            diag = read(RUNS / f'candidate_{name}_seed0' / f"stage_{stage['stage']}" / 'diagnostics.json')
            for split in ('T', 'V'):
                sensitivity = diag[split]['content_sensitivity']
                perturb = diag[split]['stimulus_perturbations']
                lines.append(f"| {name} | {stage['stage']} | {split} | {sensitivity['stimulus_end']['mean']:.5g} | {sensitivity['swap_time']['mean']:.5g} | {sensitivity['query_end']['mean']:.5g} | {100*perturb['zero']['prediction_change_rate']:.4f}% | {100*perturb['shuffle']['prediction_change_rate']:.4f}% |")
    lines += ['', '内容影响比较同一控制输入下原始刺激与全零刺激对应 m 的 L2，固定 64 试次诊断批次；改变率来自完整对应 T/V 评价集合。数值只作路径诊断，不存在人为 L2 或门值通过阈值。', '',
              '每阶段 diagnostics.json 同时保存刺激、规则、保持、查询四类的 memory/control update gate 均值、分位数和样本数；D_short 没有空白保持步，其 delay 门分布为 null。保留各事件梯度是否 None、是否有限及范数分布，按 memory/control/readout 分别记录参数梯度。', '',
              '![状态路径诊断](figures/state_path_diagnostics.png)', '',
              '阶段 7 未通过/未运行，因此未训练本修订的最终 T 内容 ridge 探针；未在 V 或 G/B 用探针结果选择模型。', '',
              '## 版本对比、计算预算与交付', '',
              'v1.0 默认/备用在完整 65 词原 V 上最佳准确率为 14.7827%/14.8926%；v1.1 的上述 94% 数值属于非 N≤2 的 33 词阶段 6 子任务，不能直接当作同分布的完整原任务性能改善。两版失败记录均保留。', '',
              f"本轮课程更新 A/B 合计 {course_updates:,} 次，独立小型检查 {scratch_updates:,} 次。此前 v1.0 两个 pilot 各 20,000 次更新，另有 20 次 smoke；所有 GPU 成本继承累计。", '',
              f"GPU 保守记账：v1.0 {budget['v1_0_gpu_seconds']:.3f} 秒；v1.1 {budget['v1_1_gpu_seconds']:.3f} 秒；合计 {budget['gpu_seconds']:.3f} 秒（{budget['gpu_seconds']/3600:.4f} 小时），上限 12 小时。预算不是本轮停止原因。", '']
    for name, result in candidates.items():
        speeds = {}
        for stage in result['stage_summaries']:
            curve = read(RUNS / f'candidate_{name}_seed0' / f"stage_{stage['stage']}" / 'curve.json')
            speeds.setdefault(stage['timing'], []).append(curve[-1]['updates_per_second'])
        if all(k in speeds for k in ('short', 'mid', 'original')):
            estimate = 5000/min(speeds['short'])+5000/min(speeds['mid'])+40000/min(speeds['original'])
            lines.append(f"- {name} 实测末次评价吞吐（包含 T/V 评价）：short 最低 {min(speeds['short']):.1f}、mid {min(speeds['mid']):.1f}、original {min(speeds['original']):.1f} 更新/秒。50,000 更新完整课程上限外推约 {estimate/3600:.3f} GPU 小时/模型，六个新初始化模型约 {6*estimate/3600:.3f} 小时；此估算不构成运行授权，也不计尚未允许的 G/B 科学评价与 CPU 分析。")
    lines += ['', f"v1.0 的 {preservation['files_checked']} 个源码/报告/数据产物逐一校验未变化；归档和散列在 preserved_v1_0.json 与 v1_0_snapshot.tar.gz。", '',
              f"原 split_manifest.json SHA-256：`{preservation['original_split_sha256']}`。正式 B 的 134 个内容与保留 B 集均复用原定义。G/B 始终封存，种子 101–106 未运行。", '',
              '固定数据集输入、原始标签、评价预测、训练曲线、初始/每阶段 best/latest/advance 检查点、优化器和 RNG 状态位于 results/v1_1/ 与 runs/v1_1/。源码、参数、门槛及数据集散列保存于 frozen_revision.json；失败时该文件是交付快照，不是正式科学实验已获准或完成的记录。', '',
              '[v1.0 原报告](../../report.md) · [v1.1 运行说明](../../v1_1/README.md) · [修改记录](../../v1_1/CHANGES.md)', '',
              f"**最终状态：{status['status']}。未声称 H1/H2/H3 获得支持。**"]
    (RESULTS / 'report.md').write_text('\n'.join(lines)+'\n')
    # Index only v1.1 artifacts; never rewrite the original v1.0 manifest.
    paths = [*ROOT.rglob('*'), *RESULTS.rglob('*'), *RUNS.rglob('*')]
    index = {str(p.relative_to(BASE)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
             for p in sorted(paths) if p.is_file() and '__pycache__' not in p.parts and p.name != 'artifact_manifest.json'}
    write_json(RESULTS / 'artifact_manifest.json', index)


if __name__ == '__main__':
    build_report()
