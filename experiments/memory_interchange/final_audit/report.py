"""Render the final finite audit, including limitations and reproducible links."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])+'\n'

def pct(v):return f'{100*float(v):.4f}%'
def number(v):return '—' if v=='' else f'{float(v):.6g}'

def select(rows,**where):
    chosen=[r for r in rows if all(str(r[k])==str(v) for k,v in where.items())]
    assert len(chosen)==1,(where,len(chosen));return chosen[0]

def figures():
    out=OUT/'figures';out.mkdir(exist_ok=True)
    rows=csv_read(OUT/'translation_seed_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,cross in zip(axes,(1,0)):
        for j,(mode,color) in enumerate([('all','#2864ad'),('rec','#c76521'),('E_sem','#378858')]):
            rr=[select(rows,cohort='B_reserved',seed=s,cross=cross,mode=mode,S='ALL',d='ALL') for s in SEEDS]
            ax.plot(np.arange(6)+(j-1)*.12,[100*float(r['delta_accuracy_original']) for r in rr],'o',color=color,label=mode)
        ax.axhline(0,color='gray',lw=1);ax.set_xticks(range(6),SEEDS);ax.set_xlabel('Seed')
        ax.set_title('Cross context' if cross else 'Same context control');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Patch translation gain on original target (pp)');axes[1].legend()
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(out/f'translation_gain.{ext}',dpi=200)
    plt.close(fig)
    geo=csv_read(OUT/'geometry_decomposition.csv');fig,axes=plt.subplots(1,2,figsize=(10,4))
    groups=[(0,0),(0,10),(1,0),(1,10)];labels=['same d0','same d10','cross d0','cross d10']
    for ax,key,label in zip(axes,('cka','mean_fraction'),('Centered linear CKA','D_mean / D_raw')):
        data=[[float(r[key]) for r in geo if int(r['cross'])==c and int(r['d'])==d] for c,d in groups]
        ax.violinplot(data,showmedians=True);ax.set_xticks(range(1,5),labels,rotation=15);ax.set_ylabel(label)
        ax.set_ylim(0,1.02);ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(out/f'geometry.{ext}',dpi=200)
    plt.close(fig)
    fixed=json.loads((OUT/'fixed_prediction_summary.json').read_text())
    fig,ax=plt.subplots(figsize=(7,3.8));v=[r['paired_MAE_improvement'] for r in fixed['folds']]
    ax.bar([str(s) for s in SEEDS],v,color=['#378858' if x>0 else '#2864ad' for x in v])
    ax.axhline(0,color='gray');ax.set_ylabel('Fixed M0 MAE - fixed M_CKA MAE (nat)');ax.set_xlabel('Held-out seed')
    ax.set_title('Original fixed predictions: 1 positive / 5 negative folds');ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(out/f'fixed_prediction_scores.{ext}',dpi=200)
    plt.close(fig)

def run():
    overall=csv_read(OUT/'translation_overall_summary.csv');byseed=csv_read(OUT/'translation_seed_summary.csv')
    cf=csv_read(OUT/'candidate_overall_summary.csv');native=csv_read(OUT/'native_seed_summary.csv')
    probe=csv_read(OUT/'context_probe_results.csv');inventory=csv_read(OUT/'context_probe_fit_inventory.csv')
    geo=csv_read(OUT/'geometry_distributions.csv');contrast=csv_read(OUT/'context_raw_contrasts.csv')
    fixed=json.loads((OUT/'fixed_prediction_summary.json').read_text());replay=json.loads((OUT/'bootstrap_replay_summary.json').read_text())
    plan=json.loads((OUT/'frozen_plan.json').read_text());budget=json.loads((OUT/'budget.json').read_text())
    ci=json.loads((OUT/'translation_block_intervals.json').read_text())['intervals']
    delta=select(overall,cohort='B_reserved',cross=1,mode='all',S='ALL',d='ALL')
    same=select(overall,cohort='B_reserved',cross=0,mode='all',S='ALL',d='ALL')
    distinct=select(cf,unit='query',cross=1,treatment='matched',subset='all_three_distinct',S='ALL')
    confirmed=float(delta['delta_accuracy_original'])>0 and float(same['delta_accuracy_original'])<0
    text=[]
    def add(s):text.append(s.strip()+'\n')
    add('# v1.3 Final audit\n\n日期：2026-10-12（America/Los_Angeles）。状态：**COMPLETED：三项有限复核完成，收口**。原六个 RNN 未再训练，旧 v1.3 点结果、主指标、全部 bootstrap、106 和数值核验失败记录均保留；没有发布或推送。')
    add('## 摘要\n\n1. **内容匹配的干预优势保留。** 旧六模型等权原始 98.1683%、匹配 70.1305%、错误内容 19.0569%，匹配比错误内容高 51.0736 个百分点。这支持本设计中的内容相关干预优势，不证明 m 仅承载纯当前内容。\n'
        '2. **原生 m 交换依赖规则上下文。** 旧同上下文平均新增错误约 1.697%，跨上下文约 53.398%；保留物理 c 组件不足以推断接收方的功能规则也被保留。\n'
        f'3. **供体规则解释得到{ "新集合的行为确认" if confirmed else "有限/未得到新集合的行为确认" }。** B1 的三候选两两不同查询中，供体规则符合率 {pct(distinct["donor_rule_agree"])}。在独立 B_reserved 上，cross-context 的冻结主要准确率差为 **{100*float(delta["delta_accuracy_original"]):+.4f} 个百分点**；同上下文翻译对原目标的差为 {100*float(same["delta_accuracy_original"]):+.4f} 个百分点。这里确认的是外部线索翻译对混合状态规则界面的补偿，与原直接交换成功不同。\n'
        f'4. **CKA 的原主检验仍为负，统计精度有限。** 固定六 fold 的平均 MAE 改善 {fixed["mean_MAE_improvement"]:+.8f} nat，1 正、5 负；条件经验区间 [{fixed["conditional_empirical_95ci"][0]:.5f}, {fixed["conditional_empirical_95ci"][1]:.5f}]。原重拟合 bootstrap 巨大区间保留，典型/极端副本复现且未发现本次检查范围内的实现错误。不能推出 CKA 永远不能预测功能。')
    add('## 冻结范围、独立确认与样本依赖\n\n原 B 为 496 内容 / 70 轨道，已打开 B1 为 134 / 19；确切集合差为 **362 内容 / 51 完整轨道**。全部剩余轨道都使用，没有按 B1 结果抽选。冻结清单、成员、散列和旧自然状态缓存访问审计见 [B_reserved_manifest.json](B_reserved_manifest.json)、[成员 NPZ](B_reserved_members.npz)、[frozen_plan.json](frozen_plan.json)、[冻结回执](freeze_receipt.json)。保存的历史自然状态缓存没有这些剩余内容，未发现其先前用于规则选择或调参的记录；这项判断限于保留的本地记录，不声称可以重建未保存的外部访问。\n\n六模型均来自原正式初始化和原选中检查点；T 仅用于固定探针拟合，G 仅用于几何。B1 新机制分析为 **EXPLORATORY**，B_reserved 为冻结翻译对比的独立确认，旧 CKA 主检验不被替代。')
    add(table(['seed','原检查点（相对实验目录）','stage7 step','SHA256'],[[p['seed'],p['path'],p['stage7_step'],p['sha256']] for p in plan['checkpoints']]))
    add('覆盖所有 18 个 (r,H)、153 无序条件对的 306 个方向、d=0/10、S=NN/NA/NB、每个内容的 7 个项目和 7 个位置查询。主差按六种子等权，两个方向等权，原 d/S/query 分布不变，NN 也计入主要 S 混合。条件结果先在每个方向、d、S 的同一 mask 内按保留查询计算，再对这些单元等权；没有把每个查询当独立模型或独立统计样本。轨道/种子 bootstrap 保留方向、查询类型和整条轨道。\n\n规则变化仅为新增第三反事实、A/B 后缀翻译、原生输出定义的 E_sem、G 均值分解、固定上下文探针、原 bootstrap 复播和固定预测描述。没有变更旧学习、门槛、主预测变量、alpha、方差地板或样本。完整规则见 [revision_final_audit.yaml](revision_final_audit.yaml)。')
    add('## 复核一：供体规则反事实与线索翻译\n\n以供体 a、接收方 b、共同当前完整内容 q 为索引，三目标分别为：`Q_receiver=F_b(S)(q)`、`Q_initial=F_b(S)(G_b(p_a))`、`Q_donor_rule=F_a(S)(q)`。第三项没有与初始输入解释混写。错误内容 q′ 的候选为 `F_b(S)(q′)` 和 `F_a(S)(q′)`，没有为它虚构初始输入候选。B1 重用全部原匹配/错内容 log-probabilities；另独立生成并核验初始输入数学映射。')
    rows=[]
    for treatment in ('matched','wrong'):
        for subset in ('all','receiver_initial_distinct','receiver_donor_distinct','initial_donor_distinct','all_three_distinct'):
            if treatment=='wrong' and subset not in ('all','receiver_donor_distinct'):continue
            r=select(cf,unit='query',cross=1,treatment=treatment,subset=subset,S='ALL')
            rows.append([treatment,subset,r['n_observations_all_seeds'],pct(r['receiver_agree']) if r['receiver_agree']!='' else '—',
                         pct(r['initial_agree']) if r['initial_agree']!='' else '—',pct(r['donor_rule_agree']) if r['donor_rule_agree']!='' else '—'])
    add('B1 跨上下文探索性结果（率为种子等权；数量汇总六种子，包含依赖查询）：\n\n'+table(['处理','候选可区分子集','查询数','接收规则','初始输入解释','供体规则'],rows))
    add('全部同/跨上下文、每对候选可区分子集、三候选两两不同子集、查询类型与方向计数见 [候选总体表](candidate_overall_summary.csv)、[每种子表](candidate_seed_summary.csv)、各 `B1/seed_*/candidate_agreement.csv`。完整映射是否正确、是否双射、两个读出是否互为逆以及候选完整映射符合率见 [映射表](candidate_mapping_seed_summary.csv) 和每种子的 `mapping_agreement.csv`；每试次预测、全部候选与重合关系保存在 `counterfactual_d*_S*.npz`。\n\nNN 的供体/接收规则完全重合；单操作时某些查询也重合，因此 all 的高符合率不能单独识别机制。错误内容 q′ 是固定双射重排：在固定 (a,b,d,S) 下，其候选符合率汇总可与匹配 q 的率完全相同，这不是额外独立证据。这里最有辨别力的是可区分子集与新增行为干预，不能由初始 p 可读性宣布 q 完全没更新。')
    add('翻译把 A/B 对换、N 保留。cross-context 时数学关系 `F_a(swap(S))(q)=F_b(S)(q)` 已对全部 5,040 个置换独立核验。每个状态都评价 native+S、native+swap(S)、匹配 m patch+S、匹配 m patch+swap(S)、sham。保留 **原语义目标** `F_b(S)(q)` 与 **新字面目标** `F_b(swap(S))(q)`；两者没有合并。\n\nall/rec/both 沿用旧口径；E_sem 由 receiver native 原 S 正确、且 donor native 能执行相同语义运算定义：cross 用翻译 S，same 用原 S。所有六模型的原生输出和 masks 在任何新增 m patch 之前保存（[全模型回执](all_native_masks_frozen.json)）。原 visible-S 的 both 只要求各自在自身上下文中答对同一可见 S，与执行同一语义运算有区别。所有干预共享原 mask。')
    rows=[]
    for cohort in ('B1','B_reserved'):
        for cross in (1,0):
            for mode in MODES:
                r=select(overall,cohort=cohort,cross=cross,mode=mode,S='ALL',d='ALL')
                rows.append([cohort,'cross' if cross else 'same',mode,pct(r['coverage']),pct(r['original_accuracy_original']),
                    pct(r['translated_accuracy_original']),f'{100*float(r["delta_accuracy_original"]):+.4f}',
                    f'{float(r["delta_CE_original"]):+.5f}',f'{100*float(r["delta_new_error_original"]):+.4f}'])
    add(table(['集合','上下文','mask','覆盖率','原 S 准确率','翻译 S 对原目标','差(pp)','CE 差(nat)','新增错误差(pp)'],rows))
    add('新增错误沿用 native 原 S 正确且 patch 对原目标错误的定义；rec/E_sem 的表值是条件概率。CE 差与新增错误差为翻译减原 S，负值表示减少。覆盖率是原生输出决定的共同样本比例，两个处理不会各选一批试次。\n\nB_reserved 冻结主要 all 对比的种子/轨道成对经验 95% 区间为 **['+f'{100*ci["all"][0]:.4f}, {100*ci["all"][1]:.4f}'+'] 个百分点**。这是固定网络与本集合下的块重采样描述，不包括重新训练或未知模型总体的保证；所有 masks 区间见 [translation_block_intervals.json](translation_block_intervals.json)。')
    rows=[]
    for seed in SEEDS:
        rr=[select(byseed,cohort='B_reserved',seed=seed,cross=1,mode=m,S='ALL',d='ALL') for m in MODES]
        sc=select(byseed,cohort='B_reserved',seed=seed,cross=0,mode='all',S='ALL',d='ALL')
        rows.append([seed,*[f'{100*float(r["delta_accuracy_original"]):+.4f}' for r in rr],
                     f'{float(rr[0]["delta_CE_original"]):+.5f}',f'{100*float(rr[0]["delta_new_error_original"]):+.4f}',
                     f'{100*float(sc["delta_accuracy_original"]):+.4f}'])
    add(table(['seed','cross all(pp)','cross rec','cross both','cross E_sem','cross CE差','cross新增错差(pp)','same all(pp)'],rows))
    rows=[]
    for s in SUFFIXES:
        r=select(overall,cohort='B_reserved',cross=1,mode='all',S=s,d='ALL')
        rows.append([s,pct(r['original_accuracy_original']),pct(r['translated_accuracy_original']),
            f'{100*float(r["delta_accuracy_original"]):+.4f}',pct(r['original_donor_rule_agree']),
            pct(r['original_donor_rule_agree_distinct']) if r['original_donor_rule_agree_distinct']!='' else '—',
            pct(r['translated_receiver_rule_agree']),pct(r['translated_donor_rule_agree'])])
    add(table(['S','patch 原目标','翻译后原目标','差(pp)','原S供体规则','可区分查询供体规则','翻译后接收字面规则','翻译后供体规则'],rows))
    add('NN 的原/翻译 patch log-probabilities **逐元素一致**，native/sham 也逐元素一致；B1 新翻译输出与原保存的对应 S′ 输出在原 GPU 登记容差内一致。NA/NB、两个 d、两个方向和全部 H/query 结果均保留，没有从中挑最好的一项。完整成对准确率/CE/新增错差、规则符合率与覆盖率见 [全部种子汇总](translation_seed_summary.csv)、[总体汇总](translation_overall_summary.csv)、各 `translation_d*_S*.csv`；全部 14 查询读出与候选保存在相应 NPZ，完整映射描述见 [translation_full_mapping_diagnostics.csv](translation_full_mapping_diagnostics.csv)。')
    rows=[]
    for seed in SEEDS:
        nr=select(native,cohort='B_reserved',seed=seed,treatment='native')
        nt=select(native,cohort='B_reserved',seed=seed,treatment='native_translated')
        rows.append([seed,pct(nr['original_target_accuracy']),pct(nt['original_target_accuracy']),pct(nt['literal_target_accuracy']),
                     pct(nr['min_actual_task_accuracy']),f"r={nr['minimum_r']}, H={nr['minimum_H']}, S={nr['minimum_S']}, d={nr['minimum_d']}, type={nr['minimum_query_type']}",nr['minimum_n']])
    add('B_reserved 原生质量及预定 native 翻译对照：\n\n'+table(['seed','native 原任务','native 翻译对原目标','native 翻译对字面目标','原任务最差层','最差条件','n'],rows))
    add('原生翻译主要符合新字面目标，而对原目标下降；因此翻译不是普遍提高网络准确率的处理。106 及其差条件完整保留。所有 native H×r×d×S×query 类型准确率与 CE 见 [native_all_strata.csv](native_all_strata.csv)，原生质量汇总见 [native_seed_summary.csv](native_seed_summary.csv)。\n\n'+('跨上下文翻译改善、同上下文翻译破坏原目标，并且 B1 可区分查询表现出高供体规则符合率，支持 **规则界面随 m 交换改变** 的解释。' if confirmed else '新集合结果没有完整满足“cross 改善、same 破坏”的预定方向；供体规则迁移不足以完整解释交换失败。')+' 这不证明 r 只在 m、c 完全无用，也不排除历史相关内容格式、初始输入、联合编码或最终一次性计算。新增处理包含外部语义翻译；不能写成原直接交换已经成功。')
    add('## 复核二：CKA、原生均值与上下文可读性\n\nG 上几何统计去重到 (seed,condition_pair,d)，没有把相同值跨 S 的重复计为新样本，共 1,836 个值。')
    rows=[]
    for cross in (0,1):
        for d in (0,10):
            r=select(geo,seed='all',cross=cross,d=d,metric='cka')
            rows.append(['same' if cross==0 else 'cross',d,r['n'],*[f'{float(r[k]):.6f}' for k in ('min','q05','q25','median','q75','q95','max')]])
    add(table(['上下文','d','n','min','q05','q25','median','q75','q95','max'],rows))
    add('原固定规则选出的全部 288 个高 CKA/高新增错案例都列在 [original_fixed_counterexamples.csv](original_fixed_counterexamples.csv)，保留实际 CKA、原分位数门槛、新增错误、native/pair 质量、方向最差 query 类型质量和均值分解。分位数高不自动等于 CKA 接近 1；这是预先固定规则的案例清单，不是挑某个看起来漂亮的交换。')
    examples=csv_read(OUT/'original_fixed_counterexamples.csv')
    add('以下仅显示原保存顺序的前八行；没有按新增复核结果重选，完整 288 行见链接：\n\n'+table(
        ['seed','(a,b)','d','S','实际 CKA','native(pair)','最差方向/type native','新增错误 R'],
        [[int(float(r['seed'])),f"({int(float(r['a']))},{int(float(r['b']))})",int(float(r['d'])),r['S'],
          f"{float(r['cka']):.6f}",pct(r['native_correct']),pct(r['native_min_direction_query_accuracy']),pct(r['new_error'])] for r in examples[:8]]))
    add('原生坐标均值分解使用原 D_raw 的同一分母：`D_mean=||mu_a-mu_b||²/den`，`D_center=mean_q||(m_a-mu_a)-(m_b-mu_b)||²/den`。没有对干预状态平移、旋转或标准化。独立计算逐项验证 `D_raw=D_mean+D_center`，最大绝对误差 '+f'{json.loads((OUT/"geometry_checks.json").read_text())["max_decomposition_error"]:.3g}'+ '。完整值与分母见 [geometry_decomposition.csv](geometry_decomposition.csv)。')
    rows=[]
    for cross in (0,1):
        for d in (0,10):
            rr=[select(geo,seed='all',cross=cross,d=d,metric=m) for m in ('D_raw','D_mean','D_center','mean_fraction')]
            rows.append(['same' if cross==0 else 'cross',d,*[number(r['median']) for r in rr]])
    add(table(['上下文','d','D_raw 中位数','D_mean 中位数','D_center 中位数','D_mean/D_raw 中位数'],rows))
    add('中心化消去所有 q 共享的平移，是数学事实：若 `m_b(q)=m_a(q)+v`，中心化线性 CKA=1，v 仍可能影响未来 A/B 的解释。独立合成检验得到 CKA≈1。这只是可能机制；真实 RNN 的 D_center 不为零，不能说网络只是相差一个平移。原保留未中心化信息的 D_raw 在旧主留出检验同样没有稳定改善（M_raw MAE 约 2.69497 nat，M0 2.68475）。这些分解是描述，未临时替换主预测变量。')
    add('上下文探针为固定线性 ridge，alpha=1、r→±1、阈值 0，仅用 T 原 512 内容自然状态拟合，r/H/d 平衡，排除 AB/BA 作历史留出。只用 T 拟合坐标中心与标准差；仅“精确为零”的标准差置 1。m 每条件内容等权；c 不把重复 q 当独立状态：pooled 拟合为 28 个 (r,H,d)，单 d 为 14 个，实际独特向量数另存。pooled、fit-d0、fit-d10 全部固定展示，未按分数选择方向或更复杂探针。\n\nG AB/BA 历史/内容留出上的 pooled 探针：')
    rows=[]
    for seed in SEEDS:
        rr=[select(probe,seed=seed,component=c,variant='pooled',split='G',d=d,history_group='AB_BA') for c in ('m','c') for d in (0,10)]
        rows.append([seed,*[pct(r['accuracy']) for r in rr],*[number(r['signed_margin_mean']) for r in rr]])
    add(table(['seed','m d0','m d10','c d0','c d10','m 边际 d0','m 边际 d10','c 边际 d0','c 边际 d10'],rows))
    rows=[]
    for seed in SEEDS:
        inv=select(inventory,seed=seed,component='c',variant='pooled')
        values=[r for r in contrast if int(r['seed'])==seed and r['component']=='c' and r['split']=='G']
        mvals=[r for r in contrast if int(r['seed'])==seed and r['component']=='m' and r['split']=='G']
        rows.append([seed,inv['fitting_rows'],inv['actual_distinct_state_vectors'],
                     number(min(float(r['paired_context_L2_mean']) for r in values)),
                     number(max(float(r['paired_context_L2_mean']) for r in values)),
                     number(min(float(r['paired_context_L2_mean']) for r in mvals)),
                     number(max(float(r['paired_context_L2_mean']) for r in mvals)),inv['nonzero_coordinate_std_min']])
    add(table(['seed','c 独立拟合数','c 精确独特向量数','c raw差 L2 min','c raw差 L2 max','m raw差 L2 min','m raw差 L2 max','c 最小非零 T std'],rows))
    add('完整 T/G、同/不同 d、非混合/AB/BA、每个 H 的边际和准确率见 [context_probe_results.csv](context_probe_results.csv)，raw L2 见 [context_raw_contrasts.csv](context_raw_contrasts.csv)，独特状态数量与标准差见 [context_probe_fit_inventory.csv](context_probe_fit_inventory.csv)。拟合参数和每个自然状态的分数、边际、标签均保存在 `context_probes/seed_*/`。标准化可以放大很微小的残留，边际大小不能直接作为神经功能贡献。\n\nm 中 r 可读性及线索翻译与规则界面解释一致；c 的此固定 AB/BA 探针处于机会水平不能证明 c 无任何信息或因果作用。解码 r 不等于网络因果使用 r，c 可探到某些信号也不证明它控制未来规则。几何、probe、语义翻译补偿均为这组人工网络的证据，不能直接推断真实脑机制；正确原始行为也不排除历史记在 m、末尾一次性算答案。')
    add('## 复核三：重拟合不稳定与固定预测点结果\n\n原主检验 M0_v13 MAE=2.6847533452 nat，M_CKA_v13=2.6996467606 nat；加入 CKA 的均值改善为负。原 refit bootstrap MAE 改善 95% 区间 **[−42.55952, 87.77731] nat**、RMSE 改善区间 **[−172.43769, 320.18672] nat** 全部保留。这些区间很宽，不能视为有足够精度，也不能当成 CKA 普遍无效的证据。')
    add('复播副本 '+', '.join(map(str,replay['replicate_ids']))+f'（原数据定义的典型 {replay["typical"]}，最低 {replay["minimum"]}、最高 {replay["maximum"]} MAE 改善，加原 0/285），共 {replay["fold_fits"]} 个 all/rec/both×留出模型×M0/CKA 拟合。留出原种子的所有复制整体排除；每个模型权重、独特训练模型数、每个特征的 mean/scale/raw 范围/test z 范围、ridge 系数、最大预测、对应条件与有限性均保存。配对 M0/CKA 的训练标签及单位散列一致。复播 MAE 与原保存值最大差为 '+number(replay['max_MAE_replay_difference'])+'。')
    add('副本 285 的留出 106 拟合样本为原种子 `[101,101,105]`，仅两个独特模型；控制距离近恒定，使测试 z-score 极端外推。已复现最大 |z| '+f'{replay["max_test_abs_z"]:.6f}'+ '，最大 |预测| '+f'{replay["max_absolute_prediction"]:.6f}'+ ' nat，预测仍全部有限。没有发现种子副本泄漏或不成对重采样；数值外推本身不是自动的实现 bug。没有删 106、裁剪预测、改 alpha 或提高方差地板。全部记录见 [bootstrap_replay_metadata.json](bootstrap_replay_metadata.json)、[每特征表](bootstrap_feature_diagnostics.csv)、[复播分数](bootstrap_replay_scores.csv)、原 [bootstrap 副本](../results/v1_3/prediction/bootstrap.csv)。')
    extreme=[r for r in json.loads((OUT/'bootstrap_replay_metadata.json').read_text()) if r['replicate']==285 and r['mode']=='all' and r['held_seed']==106]
    add(table(['rep285 held106','拟合模型权重','control 原训练范围','原训练 std / scale','原测试范围','max |test z|','max |预测| nat'],
        [[r['model'],'101×2,105×1',f"[{r['control_raw_train_min']:.3g}, {r['control_raw_train_max']:.3g}]",
          f"{r['control_raw_train_std']:.3g} / {r['control_raw_scale']:.3g}",
          f"[{r['control_raw_test_min']:.3g}, {r['control_raw_test_max']:.3g}]",number(r['max_test_abs_z']),
          number(r['maximum_absolute_prediction'])] for r in extreme]))
    add(table(['留出种子','M0 MAE','M_CKA MAE','成对 MAE 改善(nat)','成对 RMSE 改善'],[[r['seed'],f'{r["M0_MAE"]:.8f}',f'{r["M_CKA_MAE"]:.8f}',f'{r["paired_MAE_improvement"]:+.8f}',f'{r["paired_RMSE_improvement"]:+.8f}'] for r in fixed['folds']]))
    add(f'固定预测点结果：均值 {fixed["mean_MAE_improvement"]:+.8f} nat，范围 [{fixed["range"][0]:+.8f}, {fixed["range"][1]:+.8f}]，仅 104 正、其余 5 负。六个已观察 fold score 成对重采样 100,000 次所得条件经验区间为 [{fixed["conditional_empirical_95ci"][0]:.8f}, {fixed["conditional_empirical_95ci"][1]:.8f}] nat。它条件于这六个已训练模型/预测器和原测试数据；没有重拟合，没有重新训练，不包括未知模型总体或训练不确定性，不替代原 refit bootstrap。见 [fixed_prediction_summary.json](fixed_prediction_summary.json)。\n\n“重拟合算法在退化拟合集上不稳定”与“固定预测平均未改善”是两个结果。在本设计、既定基线和原留出条件下，没有观察到稳定的 CKA 增量预测收益；不能宣称 CKA 永远无法预测功能。')
    add('## 数值核验、预算与停止\n\n原数值注记继续保留：**GPU 登记对照通过；CPU float32 原严格 logit 容差未通过；同一系数的 float64 一致、原比较 argmax 不变。** COMPLETED 表示有限实验/复核完成，不表示所有精度核验通过。新 GPU 原生缓存/不间断计算按原 atol=rtol=1e−5 通过、argmax 未变、sham 与 NN 精确一致；未用新宽容差抹掉旧 CPU 失败。')
    add(f'继承 GPU 保守记账 {budget["inherited_gpu_seconds"]:.3f} 秒。收尾后累计 **{budget["gpu_seconds"]:.3f} 秒（{budget["gpu_seconds"]/3600:.4f} GPU 小时）**，上限 12 小时；新复核登记 {budget["final_audit_gpu_seconds"]:.3f} 秒。成本口径和原始计时见 [budget.json](budget.json)。原始检查点、源代码、曲线、主预测、全部 bootstrap 的散列验证见 [preservation_check.json](preservation_check.json)。')
    add('复现命令与产物索引见 [README.md](README.md)，复核源代码在 [final_audit/](../final_audit/)，规则与成员冻结文件均在本目录；[artifact_manifest.json](artifact_manifest.json) 登记全部新产物散列。所有六种子输出都交付。完成这三项后停止：没有新模型、参数扫描、适配器、神经元消融或四步序列扩展，负结果亦保留。')
    add('## 可导出的图\n\n![独立集合的线索翻译效应](figures/translation_gain.png)\n\n![去重几何与均值占比](figures/geometry.png)\n\n![原固定预测的六个 fold](figures/fixed_prediction_scores.png)\n\n每图同时提供 PDF/SVG。')
    (OUT/'report.md').write_text('\n'.join(text))
    write_json(OUT/'conclusions.json',{'status':'COMPLETED','three_finite_audits_complete':True,
        'old_CKA_main_test_unchanged':True,'B1_new_hypothesis_exploratory':True,
        'B_reserved_translation_direction_confirmed':confirmed,
        'cross_primary_accuracy_gain':float(delta['delta_accuracy_original']),
        'same_context_control_accuracy_change':float(same['delta_accuracy_original']),
        'cross_primary_95ci':ci['all'],'source_rule_CF_three_distinct_B1':float(distinct['donor_rule_agree']),
        'fixed_MAE_improvement':fixed['mean_MAE_improvement'],'retraining':False,'all_6_seeds_retained':True,
        'CPU_float32_strict_tolerance_failure_retained':True,'stop_after_requested_scope':True})
    figures()

if __name__=='__main__':run()
