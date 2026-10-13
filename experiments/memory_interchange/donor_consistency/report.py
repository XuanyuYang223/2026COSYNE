"""Finite descriptive report and inventory; preserve all earlier findings."""
import sys
import platform
import numpy as np
from .common import *

def find(rows,**where):
    rr=[r for r in rows if all(str(r[k])==str(v) for k,v in where.items())]
    assert len(rr)==1,(where,len(rr));return rr[0]
def percent(v):return '—' if v=='' else f'{100*float(v):.8f}%'
def num(v):return '—' if v=='' else f'{float(v):.8g}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])+'\n'

def run():
    plan=load_plan();audit=read_json(OUT/'pairing_audit.json');rows=csv_read(OUT/'seed_summary.csv')
    intervals=read_json(OUT/'block_intervals.json')['intervals'];controls=csv_read(OUT/'supplemental_controls.csv')
    crows=csv_read(OUT/'control_state_differences.csv');receipt=read_json(OUT/'compute_receipt.json')
    text=[]
    def add(s):text.append(s.strip()+'\n')
    add('# 状态交换与供体原生输出一致性复核\n\n日期：2026-10-13（America/Los_Angeles）。状态：**'+audit['status']+'**。本轮只读现有六个冻结模型的 B1/B_reserved 缓存，没有网络前向、重新训练、增加种子或重拟合探针/预测器；没有推送、发布或投稿。')
    add('## 配对覆盖率与来源\n\n项目实际目录为 `experiments/memory_interchange/`。已读取根 README、v1.3 README 和 Final audit；没有发现适用 AGENTS.md。B1=134 内容/19 完整轨道，B_reserved=362/51，成员与原 manifest 完全相同且集合不交。**两个集合均已打开，本次只是新增描述性复核，B_reserved 不再称为尚未打开的独立确认集。**')
    add(table(['集合','应有非自身查询记录','实际可配对','缺失','覆盖率'],[[d,audit['expected_primary_queries'][d],audit['paired_primary_queries'][d],audit['expected_primary_queries'][d]-audit['paired_primary_queries'][d],percent(audit['paired_primary_queries'][d]/audit['expected_primary_queries'][d])] for d in DATASETS]))
    add('主均值沿用 153 无序条件对的两个方向（306 非自身方向）、六种子、d、原 S 和查询类型的等权口径。数组另有 18 个自身方向，全部保留作对照，共 '+str(sum(audit['diagonal_expected_queries'].values()))+' 个查询记录，不作为跨条件新证据。“同上下文且同历史”的非自身格为空，是条件定义导致，不是缺失。\n\n原生键由 dataset、seed、供体 (r,H)、完整内容 ID、d/交换事件、实际 effective_suffix、查询类型和项目组成；patch 键另外包含接收方 (r,H)、处理类型和原 S。条件/内容轴由真实状态缓存元数据及保存源码核对；每个原生键唯一，每个 patch 键唯一。完整数据中每个原生键受控地展开为 17 个接收方×两个处理，基数 34；没有意外多对多。连接覆盖明细见 [pairing_coverage.csv](pairing_coverage.csv) 和 [pairing_audit.json](pairing_audit.json)。')
    add('未翻译配对 `(m_a(q),c_b),S` 对 `(m_a(q),c_a),S`；翻译配对 `(m_a(q),c_b),T(S)` 对 `(m_a(q),c_a),T(S)`。因此原 S=NA 的翻译 patch 使用 **供体 NB**，原 S=NB 使用 **供体 NA**；没有使用接收方原生输出，也没有把翻译结果接到供体原 S。供体历史、初始输入 `p_a=G_a^{-1}(q)` 和共享当前 q 均已检查。\n\n实际交换后后缀长 16 步，没有外部上下文重现；两槽线索分别位于相对第 1/5 步、每槽后保持 3 步、末尾保持 5 步、查询相对第 14–16 步。交换事件为 21+d，首次查询 35+d，最终读出 37+d。保存的 `native_logp` 来自供体自然状态的原生续跑，与 patch 的 effective_suffix 输入、保持和查询时点一致；不间断原生输出另外作连接/数值对照。没有因外部输入不一致而排除任何条件。')
    add('## 答案一致率与概率差异\n\n主比较为 all 可配对查询，不使用处理后正确率或一致率选择样本。argmax 固定取最大值中类别索引最小者，类别顺序 0–6；两侧相同。概率由 float64 `exp(logp)` 得到，先检查有限与原总和误差≤2e−6，再按行归一化。逐查询比较最大绝对类别概率差及 TV；不比较带共同平移的 raw logits。精确相等、容差接近和答案相同分别报告，不是新的科学成功门槛。\n\n下表率按六模型/方向等权，概率均值同口径。括号中的区间只对固定六模型做共同轨道块重采样：')
    concise=[]
    for dataset in DATASETS:
        for treatment in TREATMENTS:
            for context in ('same','cross'):
                r=find(rows,dataset=dataset,seed='equal_weight',treatment=treatment,context=context,history='all',S='all',d='all',query_type='all',donor_subset='all')
                ci=next(v['95ci'] for v in intervals if v['dataset']==dataset and v['treatment']==treatment and v['context']==context and v['history']=='all')
                concise.append([dataset,treatment,context,percent(r['answer_agreement_equal_cells']),
                    f'[{100*ci[0]:.8f}, {100*ci[1]:.8f}]',percent(r['full14_agreement']),
                    num(r['max_abs_probability_difference_mean_equal_cells']),num(r['TV_mean_equal_cells']),num(r['max_abs_probability_difference_max'])])
    add(table(['集合','处理','上下文','答案一致','经验95%区间(%)','14查询全一致','max概率差均值','TV均值','max概率差最大'],concise))
    add('查询分子/分母与等权均值是不同对象。完整 all 的各方向等量，因此合并查询率与等权率相同；在供体正确/错误子集或缺失时可能不同，均分别保存。六种子汇总的 `answer_agreement_equal_cells` 是每种子各非空方向×d/S×类型单元率的等权均值再等权六种子；`answer_agreement_pooled` 是总分子/总分母。概率均值保留同样区分；中位数与95/99分位数是合并实际查询的描述性分布，没有把它们当独立统计样本，也没有用“平均各模型中位数”冒充合并中位数。')
    add('## 六种子、同/异历史及数值细节')
    detailed=[]
    for dataset in DATASETS:
        for seed in SEEDS:
            for treatment in TREATMENTS:
                r=find(rows,dataset=dataset,seed=seed,treatment=treatment,context='all',history='all',S='all',d='all',query_type='all',donor_subset='all')
                detailed.append([dataset,seed,treatment,r['answer_agree_n'],r['n_subgroup_queries'],r['answer_disagree_n'],
                    percent(r['answer_agreement_equal_cells']),percent(r['full14_agreement']),num(r['max_abs_probability_difference_max'])])
    add(table(['集合','seed','处理','相同数','查询数','答案分歧数','答案一致','14全一致','最大概率差'],detailed))
    hist=[]
    for dataset in DATASETS:
        for treatment in TREATMENTS:
            for context in ('same','cross'):
                for history in ('same','different'):
                    r=find(rows,dataset=dataset,seed='equal_weight',treatment=treatment,context=context,history=history,S='all',d='all',query_type='all',donor_subset='all')
                    hist.append([dataset,treatment,context,history,r['n_subgroup_queries'],percent(r['answer_agreement_equal_cells']),
                        percent(r['item7_agreement']),percent(r['position7_agreement']),percent(r['full14_agreement']),num(r['TV_mean_equal_cells'])])
    add(table(['集合','处理','上下文','历史','n','答案一致','item7全一致','position7全一致','14全一致','TV均值'],hist))
    probability=[]
    for dataset in DATASETS:
        for treatment in TREATMENTS:
            r=find(rows,dataset=dataset,seed='equal_weight',treatment=treatment,context='all',history='all',S='all',d='all',query_type='all',donor_subset='all')
            for metric in ('max_abs_probability_difference','TV'):
                probability.append([dataset,treatment,metric,num(r[metric+'_mean_equal_cells']),num(r[metric+'_median']),
                    num(r[metric+'_q95']),num(r[metric+'_q99']),num(r[metric+'_max'])])
    add(table(['集合','处理','概率指标','均值','中位数','q95','q99','最大'],probability))
    tolerance=[]
    for dataset in DATASETS:
        for treatment in TREATMENTS:
            r=find(rows,dataset=dataset,seed='equal_weight',treatment=treatment,context='all',history='all',S='all',d='all',query_type='all',donor_subset='all')
            tolerance.append([dataset,treatment,r['probability_n'],percent(r['probability_exact_query_rate']),percent(r['probability_exact_element_rate']),
                percent(r['probability_close_1e6_rate']),percent(r['probability_close_1e5_rate']),percent(r['probability_close_1e4_rate']),
                r['patch_tie_n'],r['donor_tie_n']])
    add(table(['集合','处理','概率可比数','全部7概率精确相同','逐元素精确相同','≤1e−6','≤1e−5','≤1e−4','patch并列最大','donor并列最大'],tolerance))
    add('完整向量一致率指 14 个 argmax 答案逐项相同，并分别保留两个 7 查询向量；它不要求预测合法。非双射或两类预测不互为逆的向量也保留，不能把“两个错误/非法向量相同”写成完整置换正确。相关数量、所有 S/d/query 类型、空交叉层及六种子见 [seed_summary.csv](seed_summary.csv) 与 [strata_summary.csv](strata_summary.csv)。S 分层按原接收方字面 S，effective_suffix 独立保存。')
    add('## 供体错误是否复现\n\n供体正确/错误根据其实际 effective_suffix 的自身真值 `F_a(effective_suffix)(q)` 分类，而非原 S 或接收方目标。该真值只给实际供体输出贴标签，没有充当缺失预测。扩展查询计数含同一个原生输出被多个接收方复用，不能当独立错误样本。')
    errors=[]
    for dataset in DATASETS:
        for treatment in TREATMENTS:
            for subset in ('correct','error'):
                r=find(rows,dataset=dataset,seed='equal_weight',treatment=treatment,context='all',history='all',S='all',d='all',query_type='all',donor_subset=subset)
                errors.append([dataset,treatment,subset,r['n_subgroup_queries'],r['answer_agree_n'],r['answer_disagree_n'],
                    percent(r['answer_agreement_pooled']),percent(r['answer_agreement_equal_cells']),num(r['max_abs_probability_difference_mean_pooled']),num(r['TV_mean_pooled'])])
    add(table(['集合','处理','供体子集','n','复现答案数','分歧数','合并查询一致','等权非空单元一致','max差均值','TV均值'],errors))
    add('供体答错时与供体相同的 patch 答案就是复现相同错误，而非“供体规则真值正确”。这与上一轮的源规则真值符合率和翻译对原目标的恢复率是三个不同指标。供体错误子集为空的细格保持为空；计数和缺失逐层列出，没有删除 106。')
    add('## 控制状态与实现核验')
    ctable=[]
    for dataset in DATASETS:
        for history in ('same','different'):
            selected=[r for r in crows if r['dataset']==dataset and r['is_diagonal']=='0' and r['same_history']==str(int(history=='same'))]
            equal=sum(r['c_exact_equal']=='True' for r in selected)
            ctable.append([dataset,history,len(selected),equal,num(np.mean([float(r['raw_c_L2']) for r in selected])),num(max(float(r['raw_c_L2']) for r in selected))])
    add(table(['集合','历史','seed×d×方向数','c原坐标精确相等数','L2均值','L2最大'],ctable))
    add('原始 c 差异来自已保存自然状态，没有生成新状态；逐模型/方向见 [control_state_differences.csv](control_state_differences.csv)。若 c_a=c_b 且实际未来输入相同，起点就与供体原生起点相同，确定性数学输出相同是预期结果，不是独立新机制发现；不同批量 GPU 浮点实现仍可能出现细微概率差。不能由同历史推广到全部异历史，更不能以相同答案推断 c 普遍无因果作用。')
    add('NN 全量检查共 '+str(audit['NN_entries_checked'])+' 个 dataset×seed×d，S 与 T(S) 使用同一个供体来源，全部 324 个缓存方向的 patch 输出与主比较结果一致。原生/sham、不间断原生、B1 已存同后续输入全状态对照的 argmax 都一致；可用对照概率差另见 [supplemental_controls.csv](supplemental_controls.csv)。B_reserved 没有保存全状态对照，该可选项没有伪造，也不影响已完整保存的主要配对。\n\n独立直读小样本在看输出之前冻结，覆盖六种子、两集合、同/跨上下文、所有可实现的同/异历史、全部 S/d/处理和两查询类型。'+str(audit['independent_direct_sample_checked'])+' 个键以真实 q/(r,H) 重新搜索源索引，重算预测与概率并核对向量化连接；见 [direct_index_validation.json](direct_index_validation.json)。')
    add('## 不确定性、产物与限制\n\n固定 RNG seed=20261013，10,000 次轨道块重采样；B1 的 19 条和 B_reserved 的 51 条轨道分别抽取，每次在六模型间共用相同所抽轨道，同时保留其全部内容、方向、处理、d、S、query。六模型固定，不重采样模型、不重新训练；区间条件于这六模型，不包含未知模型总体。见 [block_intervals.json](block_intervals.json)，原始轨道权重与样本均保留。')
    add('冻结口径只约束这次后验分析，不追溯宣称在旧结果出现前已预注册。输入位置/散列、模型散列、数组轴、配对键、类别順序、聚合、容差、重采样和代码散列见 [analysis_plan.json](analysis_plan.json)。'+str(read_json(OUT/'preservation_check.json')['files'])+' 个旧登记文件的前后散列全部一致，旧报告、结果、模型、成员和失败记录未被改写。\n\n保留原数值核验注记：**GPU 登记对照通过；CPU float32 原严格 logit 容差失败；同一系数 float64 一致、argmax 不变。** 本次概率容差不能替代或抹去旧 logit 失败。')
    add('每个可配对查询的预测/供体真值、flags、连续概率差保存在轻量张量中；完整语义键和源概率位置保存在 [input_index.json](input_index.json)。答案分歧或 max 概率差>1e−5 的记录另存两侧全部七类概率、完整键、真值与错误标志；其他微小非零概率差同样可定位并通过 `inspect_record` 取回两侧概率，避免复制整个已有概率库。见 [README.md](README.md) 的命令与索引格式。没有为每个微小差异编造机制解释。')
    english=[]
    for dataset in DATASETS:
        r=find(rows,dataset=dataset,seed='equal_weight',treatment='all',context='all',history='all',S='all',d='all',query_type='all',donor_subset='all')
        english.append(f'{dataset}: {100*float(r["answer_agreement_equal_cells"]):.8f}% of query answers and {100*float(r["full14_agreement"]):.8f}% of complete 14-query prediction vectors agreed, with mean total variation {float(r["TV_mean_equal_cells"]):.6g} and maximum probability difference {float(r["max_abs_probability_difference_max"]):.6g}')
    sentence='Across the six frozen RNNs and tested continuations, memory-state patches reproduced donor-native outputs ('+'; '.join(english)+'), including donor-native errors; these are descriptive comparisons on previously opened data.'
    add('可供摘要使用的英文结果句：\n\n> '+sentence)
    add('这些是已测试后续输入中的输出一致性描述。即使答案高度一致，概率差异仍需按实算值限定；不据此声称 m 仅编码纯内容、规则只存在 m、c 无因果作用、所有未来任务均功能等价、子空间已分离或任何疾病修复机制。原 CKA 主检验与翻译恢复结论均保持原样。完成本次缓存复核后收口。')
    add('CPU 只读分析用时 '+num(receipt['CPU_wall_seconds'])+' 秒（该值为比较阶段）；新增 GPU 时间 **0**，原累计保守记账仍为 '+num(read_json(FINAL/'budget.json')['gpu_seconds'])+' 秒。无新网络调用，无外部发布。')
    (OUT/'report.md').write_text('\n'.join(text));(OUT/'english_result.txt').write_text(sentence+'\n')
    write_json(OUT/'status.json',{'status':audit['status'],'complete_scope':'existing-output pairing, consistency/distributions/full-vectors/donor-errors/strata/fixed-model orbit intervals/controls/hash checks',
        'missing_primary_queries':audit['missing_primary_queries'],'missing_probability_queries':audit['probability_missing_queries'],
        'network_forward_calls':0,'network_training_updates':0,'seeds':list(SEEDS),'new_confirmation':False,'external_publication':False})
    write_json(OUT/'environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'GPU_seconds_added':0})
    manifest={}
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json':
            manifest[str(p.relative_to(OUT))]={'bytes':p.stat().st_size,'sha256':sha(p)}
    write_json(OUT/'artifact_manifest.json',manifest)
    print(json.dumps({'status':audit['status'],'artifact_count':len(manifest),'report':str(OUT/'report.md')}),flush=True)

if __name__=='__main__':run()
