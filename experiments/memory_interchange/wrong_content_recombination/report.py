#!/usr/bin/env python3
"""Render the Chinese report from saved audit summaries, without model execution."""
import csv
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent


def read(name):
    return json.loads((OUT/name).read_text())


def rows(name):
    with (OUT/name).open() as f:
        return list(csv.DictReader(f))


def pct(value):
    return 'null' if value is None else f'{float(value)*100:.8f}%'


def six(row,key):
    return row['six_seed_equal_mean_'+key+'_rate']


def table(headers,values):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
            *['| '+' | '.join(map(str,value))+' |' for value in values]]


def main():
    s=read('summary.json');p=read('analysis_plan.json');v=read('preservation_check.json')
    assert s['status']=='COMPLETED' and v['all_unchanged']
    primary,allq=s['primary'],s['cross_all']
    n=primary['selected_n'];base=primary['base_n']
    keys=['donor_math','receiver_original_math','recombined_math','other']
    labels=['供体内容＋供体规则','接收方内容＋接收方规则','供体内容＋接收方规则','其他答案']
    winner=max(range(4),key=lambda i:six(primary,keys[i]))
    overall=[r for r in rows('overall.csv')]
    # CSV null is explicit; values used below come from JSON or parsed numeric fields.
    def convert(row):
        return {k:(None if val=='null' else val) for k,val in row.items()}
    overall=list(map(convert,overall))
    seed=[convert(r) for r in rows('seed_summary.csv') if r['selection']=='triple_distinct']
    exceptions=rows('donor_native_disagreements_on_primary.csv')
    assert len(exceptions)==n-primary['reproduce_donor_native_n']
    def category(r):
        for key,field in [('donor_math','donor_math'),('receiver_original_math','receiver_math'),('recombined_math','recombined_math')]:
            if r['wrong_answer']==r[field]:
                return key
        return 'other'
    classified=[]
    for seed_id,counts in [('all_six',primary), *[(r['seed'],{k+'_n':int(r[k+'_n']) for k in keys}) for r in seed]]:
        for key in keys:
            exc=[r for r in exceptions if category(r)==key and (seed_id=='all_six' or r['seed']==seed_id)]
            reproduced=counts[key+'_n']-len(exc)
            classified.append({'seed':seed_id,'category':key,'category_n':counts[key+'_n'],
                'donor_correct_reproduced_n':reproduced if key=='donor_math' else 0,
                'donor_error_reproduced_n':reproduced if key!='donor_math' else 0,
                'donor_correct_not_reproduced_n':sum(r['donor_native_answer']==r['donor_math'] for r in exc),
                'donor_error_not_reproduced_n':sum(r['donor_native_answer']!=r['donor_math'] for r in exc)})
    with (OUT/'category_donor_native_cross.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(classified[0]));w.writeheader();w.writerows(classified)
    joint={r['category']:r for r in classified if r['seed']=='all_six'}
    assert sum(r['donor_error_reproduced_n'] for r in joint.values())==primary['donor_error_reproduced_n']
    recombined_error_reproductions=joint['recombined_math']['donor_error_reproduced_n']
    recombined_exceptions=joint['recombined_math']['donor_correct_not_reproduced_n']+joint['recombined_math']['donor_error_not_reproduced_n']
    reuse=rows('donor_query_reuse.csv')
    assert {int(r['receivers_per_native_key']) for r in reuse if r['selection']=='triple_distinct' and int(r['unique_native_keys']) and int(r['receivers_per_native_key'])}=={9}
    assert primary['donor_error_reproduced_n']%9==0
    unique_donor_errors=primary['donor_error_reproduced_n']//9
    main_donor_misses=n-primary['donor_math_n']
    repeated_errors=primary['donor_error_reproduced_n']
    lines=['# Wrong-content：供体规则、接收方规则与内容重组的补充分析',
        '', '**状态：COMPLETED。日期：2026-10-17（America/Los_Angeles）。**', '',
        '本轮仅重新评分已有六个冻结正式模型的 v1.3/B1 缓存。没有训练、模型前向、增加模型、修改任务、重拟合或访问 B_reserved 干预输出；这不是新的独立确认实验。原结果与失败记录均保留，未提交或推送 Git，未修改投稿正文。', '',
        '## 数据覆盖与冻结口径', '',
        f'六种子 101–106 的 36 份 raw 缓存及对应 baseline、mismatch、检查点和运行记录均齐全，输入散列与原 v1.3 产物清单一致。B1 为 134 个完整内容、19 条轨道，18 个条件；本轮科学评分只用 162 个跨规则上下文方向、d=0/10、S=NN/NA/NB、两种查询的全部 7 个项目。', '',
        f'跨上下文扩展查询总数为 **{allq["selected_n"]:,}**，主 triple_distinct 子集为 **{n:,}**，覆盖率 **{pct(primary["coverage"])}**。这些是共享状态、轨道及原生预测的扩展记录，不能视为独立重复。所有种子完整；未将不完整种子集合称为六模型。', '',
        '在读取本轮待评分 wrong/matched/native 输出前，已保存 [analysis_plan.json](analysis_plan.json)、[冻结回执](freeze_receipt.json)、输入清单和 600 个固定直读样本。这仅冻结本轮后验描述口径，不追溯称为原实验预注册。输入准备只读散列、数组头、成员、数学真值及控制状态。', '',
        '设 j 为接收方内容索引，k=mismatch_index[j]。三个数学答案分别是 answers[D,k]、answers[R,j]、counterfactual_answers[R,j]=answers[R,k]；实际供体预测另取 argmax(native_logp[D,k])。所有比较使用相同 seed、D/R、内容、d、实际 S、查询和读出时点。', '',
        '**triple_distinct 仅要求三个数学答案两两不同，不按预测或正确性筛选；原代码 three 是原生正确性掩码，本轮未使用。** 主汇总先在每个种子内合并 eligible 查询分子/分母，再对六种子等权；不按筛选后的方向重新等权。空层保留 n=0、rate=null，NN 没有可区分查询。', '',
        '## 三候选互斥主结果', '']
    lines+=table(['预测类别','扩展计数','六种子等权率'],
        [(label,f'{primary[key+"_n"]:,}',pct(six(primary,key))) for key,label in zip(keys,labels)])
    lines+=['',f'四类计数合计 **{sum(primary[k+"_n"] for k in keys):,} = {n:,}**。占比最高的是 **{labels[winner]}**。与供体网络实际输出的一致率为 **{pct(six(primary,"reproduce_donor_native"))}**，这是预测对预测的比较，独立于供体数学真值符合率。', '',
        '六种子分别如下；计数是该种子的 eligible 扩展查询数，率在种子内合并。', '']
    lines+=table(['种子','主子集 n','覆盖率','供体数学','接收方原任务','重新组合','其他'],
        [(r['seed'],f'{int(r["selected_n"]):,}',pct(r['coverage']),
          *[pct(r[k+'_rate']) for k in keys]) for r in seed])
    lines+=['','## 供体自己的错误是否被复现','']
    lines+=table(['种子','与实际供体一致','供体原生正确率','接收方原生正确率'],
        [(r['seed'],pct(r['reproduce_donor_native_rate']),pct(r['donor_native_correct_rate']),pct(r['receiver_native_correct_rate'])) for r in seed]
        +[('六种子等权',pct(six(primary,'reproduce_donor_native')),pct(six(primary,'donor_native_correct')),pct(six(primary,'receiver_native_correct')))])
    lines+=['']+table(['实际供体原生状态','交换复现供体答案','交换未复现供体答案'],[
        ('正确',f'{primary["donor_correct_reproduced_n"]:,}',f'{primary["donor_correct_not_reproduced_n"]:,}'),
        ('错误',f'{primary["donor_error_reproduced_n"]:,}',f'{primary["donor_error_not_reproduced_n"]:,}')])
    lines+=['',f'以上交叉计数与四分类使用完全相同的主子集，合计仍为 {n:,}。**不符合供体数学答案的 {main_donor_misses:,} 条中，{repeated_errors:,} 条仍然复现供体自己的错误，占 {pct(repeated_errors/main_donor_misses if main_donor_misses else None)}。** 因而“没有答对供体数学真值”不能直接解释为脱离供体或成功重组；供体实际输出与数学答案必须分开。', '',
        f'进一步按四类与完整原生不一致记录交叉核对：重组候选的 {primary["recombined_math_n"]:,} 条符合中，**{recombined_error_reproductions:,} 条同时是供体原生错误的复现**，只有 {recombined_exceptions} 条未复现供体。它们不是大规模成功的供体内容／接收方规则组合证据。此交叉表由已评分主计数及全部例外键推导，不改变筛选或评分，见 [category_donor_native_cross.csv](category_donor_native_cross.csv)。', '',
        '全部 12 条主子集实际供体不一致均来自 seed 106、d=0、S=NA、项目查询和 c 不等条件：6 条符合接收方原任务答案，6 条符合重组候选；供体这时本来答对。它们只涉及两个独特供体原生查询键，各在六个接收方上改变答案。完整键见 [donor_native_disagreements_on_primary.csv](donor_native_disagreements_on_primary.csv)。这些是实际答案差异，不能称为全部精确复现，也不能未经额外核验全部归为浮点舍入。', '',
        '## 全部跨上下文查询与固定分层','',
        '全部查询上三个候选可能重合。下列符合率是分别计算的非互斥比例，不要求合计为 100%；“其他”指三个候选均未符合。', '']
    lines+=table(['全部跨上下文指标','扩展计数','六种子等权率'],
        [(label,f'{allq[key+"_n"]:,}',pct(six(allq,key))) for key,label in zip(keys,labels)]
        +[('与实际供体输出一致',f'{allq["reproduce_donor_native_n"]:,}',pct(six(allq,'reproduce_donor_native')))])
    lines+=['','主子集的预定边际分层如下，完整交叉格、逐种子分子分母和覆盖率见 [strata_summary.csv](strata_summary.csv)；NN 的空层没有被改定义以产生样本。','']
    strata=[r for r in overall if r['selection']=='triple_distinct' and r['family'] in ('d','S','query_type','history_relation')]
    lines+=table(['因素','层','主子集 n','覆盖率','供体数学','重组数学','实际供体一致'],
        [(r['family'],r['stratum'],f'{int(r["selected_n"]):,}',pct(r['coverage']),
          pct(r['six_seed_equal_mean_donor_math_rate']),pct(r['six_seed_equal_mean_recombined_math_rate']),
          pct(r['six_seed_equal_mean_reproduce_donor_native_rate'])) for r in strata])
    lines+=['','## c 精确相等与不相等','',
        '控制缓存轴为 [d,condition,32]，**没有内容轴**。只用已有原坐标判断逐元素精确相等，不加 L2 阈值；距离在转为 float64 后计算。下表是同一 triple_distinct 主子集的条件分层，不据结果选样。百分比仍先在种子内合并再对六种子等权；n 是六种子的扩展计数之和，不能直接用总计数复算等权百分比。合并计数率另在 overall.csv 保存。','']
    byc=[r for r in s['by_control'] if r['selection']=='triple_distinct']
    lines+=table(['c 关系','eligible n','有样本种子','供体数学','重新组合','实际供体一致','查询加权 L2 均值'],
        [(r['stratum'],f'{r["selected_n"]:,}',r['n_seeds_nonempty'],pct(six(r,'donor_math')),
          pct(six(r,'recombined_math')),pct(six(r,'reproduce_donor_native')),
          'null' if r['control_L2_query_weighted_mean'] is None else f'{r["control_L2_query_weighted_mean"]:.9g}') for r in byc])
    lines+=['','独立的控制状态距离单位是 seed×d×跨上下文方向，以下未按 S 或查询重复计数；完整逐对记录见 [control_state_pairs.csv](control_state_pairs.csv)。','']
    crows=[r for r in s['control_distances'] if r['control_relation'] in ('equal','unequal')]
    lines+=table(['历史','c 关系','状态对数','L2 均值','L2 中位数','L2 最小','L2 最大'],
        [(r['history_relation'],r['control_relation'],r['seed_d_direction_n'],
          'null' if r['L2_mean'] is None else f'{r["L2_mean"]:.9g}',
          'null' if r['L2_median'] is None else f'{r["L2_median"]:.9g}',
          'null' if r['L2_min'] is None else f'{r["L2_min"]:.9g}',
          'null' if r['L2_max'] is None else f'{r["L2_max"]:.9g}') for r in crows])
    lines+=['',
        '当 c_D=c_R 且未来输入相同，wrong patch 的起点 (m_D(q[k]),c_R) 就等于供体原生起点；确定性续跑相同是数学预期，不能作为独立机制发现。c 不相等时的一致程度是这六模型、这些后续输入下的行为证据，不能推广成 c 对所有任务无因果作用。同历史和异历史的差别须连同原坐标距离及分层分母解释。', '',
        'c 不等组同时包含极小残差和较大差异，不能将“逐元素不相等”自动解释成很大的功能差异。上面 12 条实际答案差异对应 c 距离约 1.993–2.026；因此本报告保留这些例外，也不以总体高一致率宣称 c 完全无作用。', '',
        '任何条件层若有种子 n=0，严格六种子等权率写 null，另保留明确标注的 available-seed 均值与非空种子数，不把空层补零。这里的空层与主缓存缺失是不同情况。', '',
        '## 同起点重排检查：wrong[D,R,j] 与 matched[D,R,k]','']
    relation=s['same_start_consistency']
    lines+=table(['范围','查询数','答案一致率','max概率差均值','max概率差最大','TV均值','≤1e−5 比例'],[
        (name,f'{r["n"]:,}',pct(r['answer_agreement']),f'{r["max_abs_probability_difference"]["mean"]:.9g}',
         f'{r["max_abs_probability_difference"]["max"]:.9g}',f'{r["TV"]["mean"]:.9g}',pct(r['close_rates']['1e-05']))
        for name,r in relation.items()])
    lines+=['',
        f'特别地，科学评分范围内的全部 {relation["cross_context"]["n"]:,} 条跨上下文 wrong/matched 重排比较，归一化概率向量逐元素精确相同；全 324 方向中有 {relation["all_stored_directions"]["exact_probability_vector_n"]:,} 条概率向量精确相同，其余存在上表所列微小差异，但答案全部相同。这里比较的是概率，不声称 raw logits 逐位相等。', '',
        'c_R 不依赖内容，因此该重排在数学上具有相同交换起点与相同后续输入。概率以 float64 exp(logp) 后检查并归一化，报告实际差异，不要求 raw logits 逐位相同，也没有通过放宽容差隐去失败。1e−6/1e−5/1e−4 的完整描述见 [consistency_checks.json](consistency_checks.json)，答案分歧键见 consistency_disagreements.csv。', '',
        '**本轮主要增加的是已有干预的重新评分，不是新增独立干预证据。** 重新组合候选在原报告中已作为 wrong-content 的接收方规则反事实出现；本轮将它与正确索引的供体规则答案、接收方原任务答案和实际供体输出放在同一可区分子集上比较。固定双射重排意味着总体 wrong 与 matched 现象不应算两次独立证据。', '',
        '## 独特供体查询与索引核验','',
        f'全部跨上下文扩展记录对应 {allq["unique_donor_queries"]:,} 个独特供体原生查询键；主子集对应 {primary["unique_donor_queries"]:,} 个。键为 (seed,D,q[k],d,S,type,item)，不同接收方可复用同一原生预测。每种子的复用次数分布见 [donor_query_reuse.csv](donor_query_reuse.csv)，完整计数张量是 donor_query_usage_seed_*.npz。扩展计数不是独立原生预测数量。', '',
        f'复用张量核验主子集中每个非空供体原生键恰好出现 9 次。因此 {repeated_errors:,} 条供体错误复现记录对应 **{unique_donor_errors:,} 个独特原生错误查询键**，不能当作 {repeated_errors:,} 个独立供体错误。', '',
        f'{s["scored_cells"]} 个 seed×d×S 缓存全部核对内容和条件顺序、跨轨道双射 mismatch、q[k]≠q[j]、反事实等于 answers[R,k]、数组真值与独立矩阵真值、baseline 原生值，以及自然状态初始置换和轨迹。后续 16 步输入重建自原 task.py 的纯输入函数，并与既有登记输入散列一致；没有外部上下文重现，交换 21+d、首次查询 35+d、最终读出 37+d。', '',
        f'{s["direct_index_checks"]} 个预先固定小样本通过独立 ZIP 成员直读、q/条件语义键重新定位、列表实现的单查询真值与标量 argmax 核验；概率归约最大误差 {read("direct_index_checks.json")["max_probability_reduction_error"]:.3g}。详见 [direct_index_checks.json](direct_index_checks.json)、[index_checks.csv](index_checks.csv)。', '',
        f'{v["files"]} 个既有跟踪文件与本轮输入的前后散列一致。新增训练更新 0、网络前向 0、GPU 秒 0；评分 CPU 墙钟 {s["elapsed_CPU_wall_seconds"]:.3f} 秒。没有导入 torch。', '',
        '**原数值注记保留：GPU 登记对照通过；CPU float32 原严格 logit 容差未通过；相同系数 float64 一致、原比较 argmax 不变。** 本轮只是重读缓存，未重跑这些网络核验；概率容差不替代旧 logit 失败。', '',
        '## 对 COSYNE 摘要的影响','',
        '可以保留：在这些冻结模型及已测输入中，内容匹配相较错内容具有原先报告的干预优势；状态交换表现出规则上下文依赖；本轮以可区分候选和实际供体错误的复现进一步定位行为解释。原 CKA 主留出测试未观察到稳定增量预测收益的结论与统计精度限制不变。', '',
        f'需要按实算收窄：本轮主子集最大类别是“{labels[winner]}”，供体实际输出一致率为 {pct(six(primary,"reproduce_donor_native"))}，重新组合答案比例为 {pct(six(primary,"recombined_math"))}，其中 {recombined_error_reproductions:,} 条是供体自身错误的复现，仅 {recombined_exceptions} 条实际偏离供体。当前结果没有展示普遍成功的“供体内容＋接收方规则”组合接口。摘要应给具体口径，不把数学真值符合、原生预测复现或线索翻译恢复写成同一结果。不能据此声称 m 仅编码纯内容、规则只在 m、c 普遍无因果作用、已完成内容/规则子空间分离，或泛化到所有未来任务及真实脑机制。', '',
        '同起点重排的一致和 c 相等时的确定性预期不构成新的独立机制发现；B1 此前已打开，本轮没有新增确认样本。若描述线索翻译恢复，仍须明确那是增加外部语义翻译后的既有结果，不是原直接交换成功。', '',
        '## 英文摘要句','']
    abstract=(f'In a descriptive reanalysis of six frozen RNNs, cross-context wrong-content patches on queries with three pairwise-distinct mathematical candidates matched donor-content/donor-rule answers in {100*six(primary,"donor_math"):.6f}%, receiver-content/receiver-rule answers in {100*six(primary,"receiver_original_math"):.6f}%, and donor-content/receiver-rule recombinations in {100*six(primary,"recombined_math"):.6f}% (equal-weight seed means; {n:,} expanded eligible queries). '
              f'Patch outputs agreed with actual donor-native predictions in {100*six(primary,"reproduce_donor_native"):.6f}%, reproducing all {unique_donor_errors:,} unique donor-native error queries ({repeated_errors:,} expanded records); these descriptive rescoring results on previously opened interventions are not independent confirmation.')
    lines+=['> '+abstract, '',
        '完整产物： [总体](overall.csv)、[逐种子](seed_summary.csv)、[分层](strata_summary.csv)、[候选重合](candidate_overlaps.csv)、[输入与来源](input_inventory.json)、[保全核验](preservation_check.json)。复现命令见 [README.md](README.md)。完成本轮固定分析后停止。','']
    (OUT/'report.md').write_text('\n'.join(lines))
    (OUT/'english_result.txt').write_text(abstract+'\n')
    empty_differences=OUT/'consistency_disagreements.csv'
    if empty_differences.stat().st_size==0:
        empty_differences.write_text('seed,d,S,D,R,j,k,query_type,query_item,cross_context,wrong_answer,matched_reindexed_answer,max_abs_probability_difference,TV\n')
    status=read('status.json');status['report_pending']=False;status['report_complete']=True
    (OUT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    inventory={str(path.relative_to(OUT)):{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
               for path in sorted(OUT.rglob('*')) if path.is_file() and '__pycache__' not in path.parts
               and path.name!='artifact_manifest.json'}
    (OUT/'artifact_manifest.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({'status':'COMPLETED','report':str(OUT/'report.md'),'primary_n':n,
                      'donor_math':six(primary,'donor_math'),'recombined_math':six(primary,'recombined_math'),
                      'donor_native_agreement':six(primary,'reproduce_donor_native')}))


if __name__=='__main__':
    main()
