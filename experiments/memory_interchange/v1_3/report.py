import csv
import json
import numpy as np
from ..common import write_json,sha256
from ..predict import save_csv
from .common import BASE,ROOT,RESULTS,RUNS
from .predict import merged_rows
from .figures import intervention_figures,training,prediction

def read(p):return json.loads(p.read_text())
def pct(v):return f'{100*v:.4f}%'

def intervention_text(out):
    result=read(out/'intervention_summary.json');g=read(out/'geometry_summary.json');c=read(out/'baseline_controls.json')
    with (out/'labels.csv').open() as f:label_rows=list(csv.DictReader(f))
    mean=lambda key:np.mean([float(r[key]) for r in label_rows if r[key]])
    lines=['| 处理 | 全查询平均准确率 | 全查询 CE nat |','|---|---:|---:|',
           f"| 原始 | {pct(mean('native_correct'))} | {mean('native_ce'):.6g} |",
           f"| sham | {pct(mean('native_correct'))} | {mean('native_ce'):.6g} |",
           f"| 匹配 | {pct(mean('matched_accuracy_all'))} | {mean('matched_CE_all'):.6g} |",
           f"| 错内容 | {pct(mean('wrong_accuracy_all'))} | {mean('wrong_CE_all'):.6g} |",'',
           '| mask | 保留比例 | 匹配准确率 | 错内容准确率 | 匹配 L nat | 错内容 L nat | 匹配 R | 错内容 R | 修复原始错误 | 错内容反事实符合 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode,v in result['mode_means'].items():
        lines.append(f"| {mode} | {pct(result['coverage'][mode])} | {pct(v['matched_accuracy'])} | {pct(v['wrong_accuracy'])} | {v['matched_L']:.6g} | {v['wrong_L']:.6g} | {pct(v['matched_R'])} | {pct(v['wrong_R'])} | {pct(v['repair'])} | {pct(v['wrong_cf_agree'])} |")
    lines+=['',f"原始 condition×S×d×查询类型的最差准确率为 {pct(result['native_min_stratum_accuracy'])}。每层计数/CE/全部真实操作与原始答对质量档保存 native_strata.csv，不用整体平均遮盖差层。",'',
            'rec 的分母为接收方原始答对的查询；both 额外要求匹配供体在自身上下文、相同 S/查询下答对；three 再要求错误供体自身答对。供体答案无需与接收方相同。匹配与错误内容处理共用先冻结的同一 mask。修复率以该 mode 的查询为分母，因此 rec/both 上恒为零。所有样本仍展示；共享状态与轨道的查询不是独立统计单位。','',
            f"GPU 续跑/sham/全状态登记对照通过。CPU float32 严格 logit 容差通过：{c['CPU_uninterrupted']['float32_strict_tolerance_passed']}，最大差 {c['CPU_uninterrupted']['float32_max_logit_error']:.6g}、回答改变 {c['CPU_uninterrupted']['float32_prediction_changes']}。相同系数转 float64，在原 2e-7/2e-6 容差下最大差 {c['CPU_uninterrupted']['float64_max_logit_error']:.6g}；实际干预仍是 GPU float32，阈值未放宽。",'',
            '| 探针目标 | 逐项目准确率（条件/间隔等权） | 完整映射准确率 |','|---|---:|---:|']
    for prefix,name in [('current','当前 q'),('initial','初始 p_a')]:
        lines.append(f"| {name} | {pct(np.mean(g[prefix+'_probe_item_accuracy']))} | {pct(np.mean(g[prefix+'_probe_full_accuracy']))} |")
    lines+=['','两个探针在完全相同的 T 自然状态上拟合，相同中心化、alpha=1。初始内容更容易线性解码只表明可读性差异，不确定网络使用何种信息。','',
            '初始输入反事实为 F_b(G_b(p_a))；当前内容目标为 F_b(q)。raw NPZ 同时保存完整映射、逐查询答案与重合关系；directions.csv 给出全体及可区分子集，mapping_diagnostics.csv 给出双向完整映射符合率。符合初始反事实与初始/历史依赖编码一致，但不能证明计算全部推迟到最后，也不能排除历史相关坐标。','',
            '| 上下文关系 | mask | 可区分查询数 | 当前目标符合 | 初始反事实符合 | 错内容反事实（答案可区分） |','|---|---|---:|---:|---:|---:|']
    with (out/'directions.csv').open() as f:dr=list(csv.DictReader(f))
    summary=[]
    for cross in (0,1):
        for mode in ('all','rec','both','three'):
            group=[r for r in dr if (int(r['r_donor'])!=int(r['r_receiver']))==bool(cross) and r['mask']==mode]
            count=sum(int(r['initial_distinct_n']) for r in group)
            wrong_count=sum(int(r['wrong_distinct_n']) for r in group)
            weighted=lambda key,nkey,den:sum(float(r[key])*int(r[nkey]) for r in group if r[key])/den if den else None
            cur=weighted('current_agree_distinct','initial_distinct_n',count);ini=weighted('initial_agree_distinct','initial_distinct_n',count)
            wrong=weighted('wrong_cf_agree_distinct','wrong_distinct_n',wrong_count)
            summary.append({'cross_context':cross,'mode':mode,'initial_distinct_queries':count,
                            'current_agree':cur,'initial_agree':ini,'wrong_distinct_queries':wrong_count,'wrong_cf_agree':wrong})
            lines.append(f"| {'跨' if cross else '同'} | {mode} | {count} | {pct(cur) if cur is not None else '空'} | {pct(ini) if ini is not None else '空'} | {pct(wrong) if wrong is not None else '空'} |")
    save_csv(out/'counterfactual_summary.csv',summary)
    lines+=['','| 供体r→接收方r | all L nat | all 新增错误 | 有符号 CE 变化 nat |','|---|---:|---:|---:|']
    ordered=[]
    for a in (0,1):
        for b in (0,1):
            group=[r for r in dr if int(r['r_donor'])==a and int(r['r_receiver'])==b and r['mask']=='all']
            total=sum(int(r['n']) for r in group)
            vals={k:sum(float(r[k])*int(r['n']) for r in group)/total for k in ('matched_L','matched_R','matched_signed_CE')}
            ordered.append({'donor_r':a,'receiver_r':b,'n':total,**vals})
            lines.append(f"| {a}→{b} | {vals['matched_L']:.6g} | {pct(vals['matched_R'])} | {vals['matched_signed_CE']:.6g} |")
    save_csv(out/'ordered_context_summary.csv',ordered)
    lines+=['','所有表均是预定配对/方向的描述性汇总；完整 d/S/H/查询类型分层及数量保存，不将共享状态查询视为独立样本。']
    return lines

def exploratory():
    out=RESULTS/'V_exploration';lock=read(RESULTS/'checkpoint_lock.json')
    with (out/'geometry.csv').open() as f:g=list(csv.DictReader(f))
    index={(int(r['a']),int(r['b']),int(r['d'])):r for r in g}
    with (out/'labels.csv').open() as f:r=[{**index[(int(v['a']),int(v['b']),int(v['d']))],**v} for v in csv.DictReader(f)]
    save_csv(out/'full_distribution.csv',r);intervention_figures(out,r)
    lines=['# v1.3 V 探索报告：EXPLORATORY','',
           'V 属于模型选择数据，不能作为正式留出确认。旧 v1.2 BLOCKED 原样保留；seed0 没有追加训练。','',
           f"锁定 best 阶段步数 {lock['stage_step']}，课程累计 {lock['total_updates']}，原 V {pct(lock['fixed_evaluation']['V']['accuracy'])}。旧细层门槛通过：{lock['old_fine_gate_passed']}；新版粗门槛及此前连续点通过：{lock['pilot_TASK_READY_v13']}。",'',
           f"checkpoint SHA：`{lock['sha256']}`；权重 SHA：`{lock['model_state_sha256']}`；Adam SHA：`{lock['optimizer_state_sha256']}`。",'',
           f"几何 V：旧前52轨道/{lock['geometry_V_contents']} 内容；行为 V：旧后半6轨道/{lock['behavior_V_contents']} 内容。与旧清单逐项比较，未根据输出重新选内容。",'']
    lines+=intervention_text(out)
    lines+=['','这些结果用于描述干预和检查定义，不给单模型大样本 p 值、不声称跨种子泛化；不会据 V 效果选择指标、模型或决定正式运行。正式主标签仍为 L_all，主基线控制 D_input，敏感性分析保留 rec/both。','',
            '![CKA](figures/cka_all_damage.png)','', '![原生距离](figures/raw_all_damage.png)','',
            '![Procrustes](figures/proc_all_damage.png)','', '![行为与覆盖率](figures/behavior_and_coverage.png)','']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'status.json',{'status':'EXPLORATORY','formal_G_B_opened':False,'single_seed_V_only':True})

def final():
    status=read(RESULTS/'status.json');lock=read(RESULTS/'checkpoint_lock.json');budget=read(RESULTS/'budget.json')
    lines=[f"# v1.3 执行报告：{status['status']}",'','日期：2026-10-12（America/Los_Angeles）。','',
           'v1.2 的 BLOCKED 未回写。v1.3 明确修改阶段 7 晋级规则，旧细操作层仍逐项展示，整体准确率不替代条件表现。seed0 固定 best 49,500，不续训或更换。','',
           'V 探索单列 [V 探索报告](V_exploration/report.md)。科学指标、训练与分析规则在正式 G/B 打开之前冻结；探索结果不决定是否继续正式运行。','',
           f"累计 GPU 保守记账 {budget['gpu_seconds']:.3f} 秒（{budget['gpu_seconds']/3600:.4f} 小时），继承此前 {budget['inherited_gpu_seconds']:.3f} 秒，上限12小时。CPU bootstrap 墙钟另列。",'']
    if not status.get('formal_G_B_opened'):
        lines+=['未打开正式 G/B；不能给确认结论。',json.dumps(status,ensure_ascii=False),'']
    else:
        training();prediction()
        distribution=merged_rows([RUNS/f'seed_{s}' for s in range(101,107)])
        intervention_figures(RESULTS,distribution)
        lines+=['## 六种子原任务与细层状态','',
                '| 种子 | 课程更新 | 新课程通过 | 选定原 V | 新粗层通过 | 旧细层通过 | B 最差条件层 |',
                '|---:|---:|---|---:|---|---|---:|']
        for seed in range(101,107):
            run=RUNS/f'seed_{seed}';t=read(run/'summary.json');v=read(run/'selected_validation.json');b=read(run/'analysis/intervention_summary.json')
            lines.append(f"| {seed} | {t['total_updates']} | {t['passed']} | {pct(v['original_V']['accuracy'])} | {v['coarse_gate_passed']} | {v['old_fine_gate_passed']} | {pct(b['native_min_stratum_accuracy'])} |")
        lines+=['','全部种子新初始化，不从 pilot 微调，失败种子保留、没有替换或依据交换续训。每个真实操作词、k、r、d、查询层的 n/accuracy/CE 保存在 selected_validation.json、每500步曲线和 native_strata.csv；全部检查点含 Adam 与 RNG。初始 CKA 使用课程开始前权重。','',
                'B 原始质量按 receiver×S×d×query_type，以及标签的双向原始平均正确率分别固定 ≥95%、90–95%、<90%；分档只用于展示，不能筛训练模型或排除不利测试行。','']
        for seed in range(101,107):
            lines += [f'## 种子 {seed}：干预与反事实','']+intervention_text(RUNS/f'seed_{seed}/analysis')+['']
        p=read(RESULTS/'prediction/summary.json')
        lines+=['## 留出预测：主标签 L_all','',
                'alpha=1 ridge，LOSO 只在其余五种子的91个不含AB/BA配对拟合；主测试为留出种子的32个未见跨上下文配对×2d×3S。折内中心化、缩放与缺失值处理。D_input 仅由 G 的初始 one-hot 输入距离计算，不看 B 标签。','',
                '| 种子 | 模型 | MAE nat | RMSE nat | 相对 M0_v13 的 MAE 改善 | 相对 M0_old 的 MAE 改善 |',
                '|---|---|---:|---:|---:|---:|']
        for r in p['scores']:
            if r['mode']=='all' and r['segment']=='unseen_cross' and r['quality']=='all':
                lines.append(f"| {r['seed']} | {r['model']} | {r['mae']:.6g} | {r['rmse']:.6g} | {r['mae_improvement']:.6g} | {r['mae_improvement_vs_old']:.6g} |")
        lines+=['','| 口径 | 模型 | MAE 改善95%区间（相对新版基线） | RMSE 改善95%区间 | 相对旧基线 MAE 区间 |','|---|---|---|---|---|']
        for r in p['intervals']:
            if r['segment']=='unseen_cross':lines.append(f"| {r['mode']} | {r['model']} | {r['mae_improvement_95ci']} | {r['rmse_improvement_95ci']} | {r['mae_improvement_vs_old_95ci']} |")
        lines+=['','rec/both 敏感性标签及原始质量三档的全部折/总体 MAE/RMSE 见 scores.csv，不改变主标签。初始 CKA+训练后 CKA 对比及配对重采样区间同时保留；若加入 D_input 后 CKA 不再改善，则未排除初始输入结构解释。模型种子与 B 轨道是重采样块，每次重新拟合；六种子仅代表有限模型集合，不把查询计成独立实验。','',
                f"固定分位规则共选中 {p['counterexample_count']} 个高 CKA/高新增错误标签；完整分布与所有符合者交付，不按故事挑内容。Spearman 只按固定 d/S/上下文类别补充；方差退化的层为 null。",'',
                f"CPU bootstrap/预测墙钟 {p['CPU_wall_seconds']:.3f} 秒。H1 参考匹配与错误内容以及反事实；H2/H3 参考新版基线下的留出改进，不能用课程或 V 探索替代。",'',
                '![课程](figures/formal_learning.png)','', '![预测误差](figures/held_out_prediction.png)','',
                '![CKA 与主损害](figures/cka_all_damage.png)','', '![行为与口径](figures/behavior_and_coverage.png)','']
        primary=next(r for r in p['intervals'] if r['mode']=='all' and r['segment']=='unseen_cross' and r['model']=='M_CKA_v13')
        lines+=[f"主检验 MAE/RMSE 改善区间：{primary['mae_improvement_95ci']} / {primary['rmse_improvement_95ci']}。只有在实际行为质量和有限模型范围内解释这些数据；不外推脑机制或抽象群结构。",'']
    lines+=['所有原始/sham/匹配/错内容/全状态输出、自然真值、共用 mask、轨道统计与方向均保留；正确行为本身不证明逐步可复用的更新。没有发布或推送。','',f"**最终状态：{status['status']}。**"]
    (RESULTS/'report.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'report.md').write_text('\n'.join(lines).replace('(V_exploration/','(../results/v1_3/V_exploration/').replace('(figures/','(../results/v1_3/figures/')+'\n')
    artifacts={}
    for folder in (ROOT,RESULTS,RUNS):
        for f in folder.rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and f.name!='artifact_manifest.json':artifacts[str(f.relative_to(BASE))]={'bytes':f.stat().st_size,'sha256':sha256(f)}
    write_json(RESULTS/'artifact_manifest.json',artifacts)
