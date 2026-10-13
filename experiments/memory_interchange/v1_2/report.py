import csv
import json
from pathlib import Path
import numpy as np
import torch
from ..common import sha256, write_json
from .common import BASE, ROOT, RESULTS, RUNS, config
from .figures import learning, formal
from .diagnostics import metadata, table, save_csv

def read(path):return json.loads(path.read_text())
def percent(value):return f'{100*value:.4f}%'

def enrich_diagnostic_tables():
    for folder in [RESULTS/'two_step_diagnostics',RUNS/'candidate_A_seed0/stage_6/two_step_diagnostics',
                   RUNS/'candidate_A_seed0/stage_7/two_step_diagnostics']:
        if not folder.exists():continue
        for path in folder.glob('*_raw.npz'):
            x=dict(np.load(path));meta=metadata(x);rows=table(meta,x)
            correct=x['prediction']==x['target']
            for factors in [('real_operation','context'),('visible_order','context'),('slots','delay'),
                            ('real_operation','slots'),('real_operation','query_type')]:
                keys=np.array(['|'.join(meta[k][i] for k in factors) for i in range(len(correct))])
                for key in np.unique(keys):
                    mask=keys==key
                    rows.append({'group':'cross:'+'|'.join(factors),'value':key,'n':int(mask.sum()),
                                 'accuracy':float(correct[mask].mean()),'CE':float(x['cross_entropy'][mask].mean())})
            save_csv(path.with_name(path.name.replace('_raw.npz','_two_step_strata.csv')),rows)

def checkpoint_inventory():
    rows=[]
    for p in sorted(RUNS.glob('*/stage_*/*.pt')):
        s=torch.load(p,map_location='cpu',weights_only=False)
        steps=sorted(set(int(v['step']) for v in s['optimizer']['state'].values()))
        assert steps==[s['total_updates']],str(p)
        rows.append({'checkpoint':str(p.relative_to(BASE)),'kind':p.stem,'sha256':sha256(p),
                     'stage':s['stage'],'stage_step':s['step'],'curriculum_updates':s['total_updates'],
                     'Adam_parameter_steps':steps,'RNG_components':list(s['rng']),
                     'consecutive_passes':s['consecutive_passes']})
    write_json(RESULTS/'checkpoint_inventory.json',rows)
    return rows

def diagnostic_text(lines,label,path):
    s=read(path)
    lines.extend(['',f'### {label}：固定 T/V 两步诊断','',
                  '| 集合 | 两步数 | 单查询正确率 | 正向完整映射正确 | 反向完整映射正确 | 双向同时正确 | 正向双射 | 反向双射 | 查询互逆 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for split,r in s.items():
        m=r['full_mapping'][0]
        values=[m[k] for k in ('forward_correct','inverse_correct','both_correct','forward_bijection','inverse_bijection','mutually_inverse')]
        acc=r['full_task']['strata']['non_N']['2']['accuracy']
        lines.append(f"| {split} | {r['two_step_n']} | {percent(acc)} | "+' | '.join(percent(v) for v in values)+' |')
    lines.extend(['','候选答案分别为正序两步、仅第一步、仅第二步、不更新、反序。保存每个原试次全部候选答案/映射及两两重合关系。下表仅统计候选答案与正确答案不同的查询；空集不作判断。','',
                  '| 集合 | 真实顺序 | 候选 | 可区分查询数 | 预测符合率 |','|---|---|---|---:|---:|'])
    for split,r in s.items():
        for op in ('II','CC','IC','CI'):
            for candidate,values in r['candidate_agreement'][op].items():
                if candidate=='correct':continue
                v=values['differs_from_correct']
                lines.append(f"| {split} | {op} | {candidate} | {v['n']} | {percent(v['rate']) if v['rate'] is not None else '不可区分'} |")
    lines.extend(['','所有查询的符合率、在全部候选中唯一的子集符合率、候选完整映射符合率分别保存在诊断 summary.json 和 NPZ；不能把候选重合产生的符合称为跳步或反序证据。II/CC 的反序与正确执行相同。IC/CI 也只在预先定义的可区分子集描述行为。完整查询映射利用已有读出，未向隐藏状态写入真值或新增训练损失。','',
                  '真实操作与可见词同时保存：IC 在 r=0 为 AB、r=1 为 BA；CI 相反。分层 CSV 保存真实操作、上下文、可见顺序、六种槽位对、d、查询类型及描述性交叉格的计数、准确率与 CE。任何分层模式只能提出进一步检验方向，不能确定内部计算机制。',''])
    for split,r in s.items():
        lines.append(f"{split} 未达到原 95% 因素门槛的真实操作层：{', '.join(r['failed_operation_layers']) or '无'}。这些层分别保存刺激结束、交换时点、最终查询的敏感性、实际计算图梯度和门值分布；选取各层固定顺序的至多 64 个试次，不设置 L2/门值阈值。")
    raw_path=path.with_name(path.name.replace('_summary.json','_V_raw.npz'))
    if raw_path.exists():
        x=dict(np.load(raw_path));meta=metadata(x);correct=x['prediction']==x['target']
        lines.extend(['','| V 真实顺序 | r | 可见顺序 | n | 准确率 | CE nat |','|---|---:|---|---:|---:|---:|'])
        for op in ('II','CC','IC','CI'):
            for context in (0,1):
                mask=(meta['real_operation']==op)&(x['context']==context)
                visible=''.join(('A' if (letter=='I')==(context==0) else 'B') for letter in op)
                lines.append(f"| {op} | {context} | {visible} | {int(mask.sum())} | {percent(correct[mask].mean())} | {x['cross_entropy'][mask].mean():.6g} |")
    lines.extend(['','非零内容敏感性说明刺激影响输出，不能证明保存了正确更新后的完整内容。正确行为也可能由控制状态保存历史、最终读出时计算得到；原状态交换分析仍然必要。',''])

def science_text(lines,status):
    from .science import scientific_config
    from ..predict import read_csv
    cfg=scientific_config();summary=read(RESULTS/'prediction/summary.json')
    lines.extend(['## 正式六种子：先报告原始行为','',
                  '| 种子 | 课程更新 | 课程通过 | 选定原 8192 V | 原 V 因素层通过 | B 主行为最差层 | B 各层通过 |',
                  '|---:|---:|---|---:|---|---:|---|'])
    for seed in cfg['seeds']:
        run=RUNS/f'seed_{seed}';train=read(run/'summary.json');val=read(run/'selected_validation.json');b=read(run/'analysis/intervention_summary.json')
        lines.append(f"| {seed} | {train['total_updates']} | {train['passed']} | {percent(val['original_V']['accuracy'])} | {val['original_V_factor_gate_passed']} | {percent(b['native_min_stratum_accuracy'])} | {b['native_strata_passed']} |")
    lines.extend(['','全部种子新初始化，不从 pilot 微调；课程前 initial.pt 用于初始 CKA。全部保留，原始行为不足的种子明确标记，不能因交换结果删去。','',
                  '## 内容匹配、错误内容与实现对照','',
                  '![任务与干预](../results/v1_2/figures/01_task.png)','',
                  '| 种子 | 匹配正确率 | 错内容原答案正确率 | 错内容反事实正确率 | sham 最大 logp 差 | 全状态最大 logp 差 | L IQR | 兼容性上限效应 |',
                  '|---:|---:|---:|---:|---:|---:|---:|---|'])
    for seed in cfg['seeds']:
        b=read(RUNS/f'seed_{seed}/analysis/intervention_summary.json')
        lines.append(f"| {seed} | {percent(b['mean_matched_accuracy'])} | {percent(b['mean_wrong_accuracy'])} | {percent(b['mean_wrong_counterfactual_accuracy'])} | {b['sham_max_logp_error']:.3g} | {b['full_max_logp_error']:.3g} | {b['loss_iqr']:.6g} | {b['compatibility_ceiling']} |")
    rows=read_csv(RESULTS/'prediction/full_distribution.csv')
    group_rows=[]
    for seed in cfg['seeds']:
        for cross in (0,1):
            for d in (0,10):
                for suffix in cfg['suffixes']:
                    g=[r for r in rows if int(float(r['seed']))==seed and int(float(r['cross_context']))==cross and int(float(r['d']))==d and r['S']==suffix]
                    group_rows.append({'seed':seed,'cross_context':cross,'d':d,'S':suffix,'pair_count':len(g),
                       **{key:float(np.mean([float(r[key]) for r in g])) for key in ('native_correct','swap_correct','wrong_correct','wrong_cf_correct','wrong_correct_different','wrong_cf_correct_different','new_error','loss_positive','signed_loss')}})
    save_csv(RESULTS/'formal_control_summary.csv',group_rows)
    advantages=[]
    for seed in cfg['seeds']:
        g=[r for r in rows if int(float(r['seed']))==seed]
        advantages.append(float(np.mean([float(r['swap_correct'])-float(r['wrong_correct']) for r in g])))
    lines.extend(['',f'匹配相对错误内容的准确率差，六种子分别为 {[round(v,6) for v in advantages]}。H1 的内容转移解释同时参考可区分答案上的供体反事实正确率，不能只根据原答案下降。完整 same/cross×d×S 分层见 formal_control_summary.csv；两方向、查询类型和 B 轨道记录保留。','',
                  'L 为逐查询正向 CE 增量的双向平均，单位 nat；R 为原答正确而交换错误的比例。改善不抵消损害；只有 L 变化而 R 不变时只称概率损害。','',
                  '## 留出种子与历史配对的主要预测','',
                  'alpha=1 ridge；拟合折仅用其余五个种子的 91 个已见配对。主测试为留出种子的 32 个未见跨上下文配对 ×2 个 d ×3 个 S（192 标签/种子）。标签聚合两个方向、全部 B 内容与 14 查询，查询不是独立 CKA 样本。其他三类测试、全部折内系数与预处理参数一并保存。','',
                  '| 种子 | 模型 | MAE nat | RMSE nat | MAE 改善 | RMSE 改善 |','|---|---|---:|---:|---:|---:|'])
    for r in summary['scores']:
        if r['segment']=='unseen_cross':
            lines.append(f"| {r['seed']} | {r['model']} | {r['mae']:.6g} | {r['rmse']:.6g} | {r['mae_improvement']:.6g} | {r['rmse_improvement']:.6g} |")
    lines.extend(['','| 模型 | MAE 改善 95% 区间 | RMSE 改善 95% 区间 | 实际重采样次数 |','|---|---|---|---:|'])
    for r in summary['intervals']:
        if r['segment']=='unseen_cross':lines.append(f"| {r['model']} | {r['mae_improvement_95ci']} | {r['rmse_improvement_95ci']} | {r['replicates']} |")
    primary=next(r for r in summary['intervals'] if r['segment']=='unseen_cross' and r['model']=='M_CKA')
    scores={r['model']:r for r in summary['scores'] if r['seed']=='overall' and r['segment']=='unseen_cross'}
    extra=scores['M_init']['mae']-scores['M_init_CKA']['mae']
    lines.extend(['',f"训练后 CKA 相对初始 CKA 模型的额外 MAE 改善为 {extra:.6g} nat。初始几何若已解释同样效果，应收窄训练塑造几何的解释。",'',
                  '层级 bootstrap 以种子和 B 轨道为块，保持方向、查询和配对关联，每次重新拟合折内预测器。六种子的区间仅描述有限模型集合，不支持普遍定律。Spearman 仅按固定种子、d、S、上下文类别补充报告；退化层保存 null。',''])
    if status['status']=='EXPLORATORY':
        lines.append('至少一个原始行为门槛不足；全部结果仅为探索性，H2/H3 不作强机制结论。')
    elif primary['mae_improvement_95ci'][0]>0 and primary['rmse_improvement_95ci'][0]>0:
        lines.append('主留出测试的 CKA 增量预测改进在两项误差上均为正，且所报区间下界为正；这些数据支持本任务与干预定义中的 H2/H3。')
    else:
        lines.append('所报主留出误差与区间未提供两项均稳定为正的 CKA 增量改进；不能宣称 H2/H3 已得到稳定支持。')
    ceilings=[read(RUNS/f'seed_{s}/analysis/intervention_summary.json')['compatibility_ceiling'] for s in cfg['seeds']]
    if all(ceilings):lines.append('全部种子满足预登记兼容性上限效应：交换普遍可行，不能据此证明 CKA 有排序预测力。')
    lines.extend(['',f"预登记高 CKA/高新增错误规则共选中 {summary['counterexamples']} 个标签；完整列表和全部分布均交付，不选单个查询制造反例。规则为每个 seed×d×S×上下文类别内 CKA 和 R 均不低于第 75 分位，且 R≥0.5%。",'',
                  f"四步未见序列扩展执行：{status['extension_executed']}。仅 ABAB/BABA 引入新净变换；扩展原始行为不足的行带 performance_floor，不能据此否定兼容性。",'',
                  '![CKA 与损害](../results/v1_2/figures/cka_damage.png)','',
                  '![留出预测](../results/v1_2/figures/held_out_prediction.png)','',
                  '![损害与反事实](../results/v1_2/figures/damage_and_counterfactual.png)',''])

def build_report():
    learning();formal();enrich_diagnostic_tables();inventory=checkpoint_inventory()
    status=read(RESULTS/'status.json');restore=read(RESULTS/'restore_consistency.json');revision=config()
    lines=[f"# v1.2 执行报告：{status['status']}",'','日期：2026-10-12（America/Los_Angeles）。',
           '',f"本轮状态 **{status['status']}**；原因 `{status['reason']}`。原研究问题、主干预和科学指标沿用 v1.0。",'',
           '## 恢复、保留与曲线审查','',
           f"v1.0/v1.1 共 {read(RESULTS/'preservation_check.json')['files_checked']} 个源码、配置、检查点、审计与报告文件散列校验未变；原 split 与 134 个正式 B 内容保持不变。候选 B 没有续训。",'',
           'A 的优先依据是旧 V 两步 84.9195%、最差真实操作层 78.0255%，相较 B 的 83.3821%/56.25% 更均衡且沿用默认初始化；未依据几何、交换或正式种子结果选择。单个 pilot 差异不能证明初始化普遍优劣。','',
           f"恢复源：`{restore['source_checkpoint']}`。SHA-256：`{restore['source_sha256']}`。",'',
           f"source latest 的阶段步数 5,000，课程累计 18,500，所有 Adam 参数 step=18,500；权重和 Python/NumPy/Generator/Torch/CUDA RNG 精确匹配。固定 T/V 准确率及各层计数一致；CE 在预登记 1e-6 nat 容差内一致。",'',
           f"T/V 复算绝对误差：{ {s:r['absolute_difference'] for s,r in restore['fixed_evaluation'].items()} }。数值舍入与恢复错误分别检查，未用阶段 5 检查点替代。旧阶段 6 报告使用 endpoint/latest；best 也位于 5,000。",'',
           '已有阶段 6 全部十个 500 步记录，以及最近 2,000 更新的 T/V、CE、k、真实操作、上下文、d、查询类型均保存。最近 2,000 更新 V 总准确率 +4.1992 百分点，两步 +11.4202 百分点，CE −0.101599 nat；采样评价点持续改善。此趋势不单独确定失败机制。','',
           '![旧曲线审查](../results/v1_2/figures/previous_curve_review.png)','',
           '## 课程与 checkpoint 门槛','',
           '每 500 更新固定 V 评价，总准确率 ≥97%、各原因素层 ≥95%，连续两次才晋级。阶段 6 上限包含已有 5,000，最多新增 45,000；阶段 7 上限 50,000，使用原 65 词和固定 8,192 试次 V。正式早期各阶段 20,000；单模型最多 200,000。没有新增初始化、容量、学习率、重复刺激、辅助损失或错误定向采样。','',
           '| 阶段 | 任务 | 阶段实际更新 | 课程累计 | V 准确率 | V CE nat | 最差真实操作层 | 两次晋级通过 |',
           '|---:|---|---:|---:|---:|---:|---:|---|']
    pilot=read(RUNS/'candidate_A_seed0/summary.json')
    for stage in pilot['stage_summaries']:
        v=stage['last_validation'];minimum=min(c['accuracy'] for c in v['strata']['true_operation_sequence'].values())
        lines.append(f"| {stage['stage']} | k≤{revision['stages'][stage['stage']-1]['max_non_N']} / {stage['timing']} | {stage['updates']} | {stage['total_updates']} | {percent(v['accuracy'])} | {v['cross_entropy']:.6g} | {percent(minimum)} | {stage['passed']} |")
    lines.extend(['',f"pilot 总更新 {pilot['total_updates']}；本轮新增 {pilot['total_updates']-18500}。阶段 1–5 从旧版本保留，没有回头重训。每阶段 latest/best/advance 的真实步数、散列、Adam/RNG 在 checkpoint_inventory.json；未晋级的阶段没有伪造 advance。",'',
                  '![连续课程曲线](../results/v1_2/figures/continuous_learning.png)','',
                  '阶段 6 的 33 词结果不等于完整任务；主实验 H+S 可含三次非 N 更新，必须通过阶段 7。latest、best、advance 分别保留，不把较高总准确率但因素层不足的 best 当成晋级检查点。',''])
    if (RUNS/'candidate_A_seed0/stage_7/summary.json').exists():
        stage=read(RUNS/'candidate_A_seed0/stage_7/summary.json')
        lines.extend(['### 阶段 7 终止点完整操作分层','',
                      '| 真实操作 | T n | T 准确率 | T CE nat | V n | V 准确率 | V CE nat |','|---|---:|---:|---:|---:|---:|---:|'])
        for op,v in stage['last_validation']['strata']['true_operation_sequence'].items():
            t=stage['last_training_evaluation']['strata']['true_operation_sequence'][op]
            lines.append(f"| {op} | {t['n']} | {percent(t['accuracy'])} | {t['cross_entropy']:.6g} | {v['n']} | {percent(v['accuracy'])} | {v['cross_entropy']:.6g} |")
        lines.extend(['','| k | T n | T 准确率 | V n | V 准确率 |','|---:|---:|---:|---:|---:|'])
        for k,v in stage['last_validation']['strata']['non_N'].items():
            t=stage['last_training_evaluation']['strata']['non_N'][k]
            lines.append(f"| {k} | {t['n']} | {percent(t['accuracy'])} | {v['n']} | {percent(v['accuracy'])} |")
        lines.extend(['',f"阶段 7 best：{stage['best_validation']['step']} 步、V {percent(stage['best_validation']['accuracy'])}；本表使用 latest 终止点，未把 best 与 latest 混写。",''])
    diagnostic_text(lines,'恢复点 step 5,000',RESULTS/'two_step_diagnostics/restore_step5000_summary.json')
    for sid in (6,7):
        path=RUNS/f'candidate_A_seed0/stage_{sid}/two_step_diagnostics/endpoint_summary.json'
        if path.exists():diagnostic_text(lines,f'阶段 {sid} 晋级/终止点',path)
    lines.extend(['### 两步诊断的描述性定位','',
                  '恢复点 V 的真实 IC 在 r=0/1 分别为 82.86%/82.38%；真实 CI 为 87.92%/69.09%。可见 AB 在两个上下文对应不同真实操作，分别为 82.86%/69.09%。这些结果同时存在操作与上下文/线索的不均衡，不能仅归因于某个数学操作。','',
                  '恢复点 d=0/10 两步 V 为 85.97%/83.91%；槽位 (1,2) 为 88.16%，(2,4) 为 82.46%。槽位与 d 的交叉格并非一致地随 d 恶化，因此不能凭这些表认定单一保持衰减机制。计数和全部交叉格均保存。','',
                  '恢复点 CI 的“只执行第一步”在答案可区分的 V 查询上符合率为 20.09%（46/229），但在候选完整映射可区分的子集上，没有一个双向完整预测映射整体等于该候选。此处有查询层面的行为模式，尚不足以证明内部跳过第二步。','',
                  '同一阶段 6 固定 V 的两步双向完整映射正确率从 16.47% 升至晋级点 73.06%，两步学习随预算延长明显改善，仍不能据此证明逐步更新。原任务终止点主要门槛缺口为三步操作因素，而非 k=0/1 遗忘。','',
                  'recovery_diagnostics/ 另保存三个诊断时点固定首 64 行的完整 T/V 敏感性、梯度、门分布；权重、Adam 全部状态/参数组与全部 RNG 均再次精确核验。没有优化或改写检查点。',''])
    if status.get('formal_G_B_opened'):
        science_text(lines,status)
    else:
        lines.extend(['## 本轮未启动科学分析','',
                      f"pilot 失败阶段：{pilot['failed_stage']}。若达到阶段上限，仅说明本预算内未达到门槛，不能声称默认模型无法学习该函数。G/B 未计算，正式 101–106 未运行；H1/H2/H3 均未检验。完整 500 步学习曲线、全部操作因素、latest/best 的权重、Adam、RNG 和诊断原始数据已保留。",''])
    budget=read(RESULTS/'budget.json');cost=read(RESULTS/'training_cost_estimate.json')
    lines.extend(['## 成本、可复现与解释边界','',
                  f"恢复前吞吐实测 {cost['updates_per_second']:.2f} 更新/秒（200 步临时权重/Adam/RNG 全部丢弃并恢复）。训练、验证、保存、缓存及 GPU 分析的累计保守记账 {budget['gpu_seconds']:.3f} 秒（{budget['gpu_seconds']/3600:.4f} 小时），其中此前 {budget['v1_0_v1_1_gpu_seconds']:.3f} 秒；总上限 12 小时。科学 CPU bootstrap 单列墙钟，不充作 GPU 时间。",'',
                  '执行命令：`python -m experiments.memory_interchange.v1_2.run`。程序按已完成 summary/检查点恢复；校验旧版本与冻结源，停止点由登记预算和门槛决定。测试结果在 results/v1_2/tests.txt；恢复单独复算和实际源散列在 restore_consistency.json。','',
                  '配置：revision_v1_2.yaml；改动：CHANGES.md。datasets 沿用 results/v1_1/datasets，不生成新的 V；原 8,192 V 与 split 文件散列冻结。学习曲线包含阶段 6 的旧 5,000；每点评价因素的 n、accuracy、CE 全部保存。','',
                  '候选 m/c 来自预先指定结构，不证明天然的独立记忆脑区。即使正确行为达到门槛，也不能证明逐步可复用的内部更新；状态交换及后续行为仍须检验。数学性质不等于网络学会群结构，本实验不能外推真实神经系统。未检索更新相关工作或判断原创性，论文提交前仍需核对。','',
                  f"**最终状态：{status['status']}。**"])
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    (RESULTS/'report.md').write_text('\n'.join(lines).replace('../results/v1_2/figures/','figures/')+'\n')
    artifacts={}
    for folder in (ROOT,RESULTS,RUNS):
        for p in sorted(folder.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json':
                artifacts[str(p.relative_to(BASE))]={'bytes':p.stat().st_size,'sha256':sha256(p)}
    write_json(RESULTS/'artifact_manifest.json',artifacts)

if __name__=='__main__':build_report()
