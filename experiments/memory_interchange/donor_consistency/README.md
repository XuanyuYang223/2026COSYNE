# Donor-native output consistency

本目录是 2026-10-13 新增的只读描述性复核。B1 与 B_reserved 均已在旧分析中打开，本轮不称独立确认。不运行 RNN，不训练或重新拟合，旧文件完全保留。主报告为 [report.md](report.md)。

## 执行

从 `/home/yangx/2026COSYNE` 执行：

```bash
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m experiments.memory_interchange.donor_consistency.test_math
python -m experiments.memory_interchange.donor_consistency.prepare
python -m experiments.memory_interchange.donor_consistency.analyze
python -m experiments.memory_interchange.donor_consistency.summarize
python -m experiments.memory_interchange.donor_consistency.validate
python -m experiments.memory_interchange.donor_consistency.report
```

`prepare` 只读取轴/成员/源码/输入时序和散列，在一致率计算前保存分析计划、代码散列与固定直读样本。已有计划不会覆盖；后续命令先验证代码未变。`analyze` 仅读取已有输出，将全量可配对记录按真实条件/内容键连接。`summarize` 汇总并对固定六模型共同重采样整轨道 10,000 次；不重采样或重训模型。`validate` 独立按 q 与 (r,H) 直读原缓存，复算固定小样本并验证旧文件散列。代码/规则冻结只针对本次分析，不追溯声称旧结果前已预注册。

全部六种子 101–106 都保留。主比较是 306 非自身方向；数组中另外 18 个自身方向均保留作实现对照。条件 (r,H) 和内容来源于原自然状态缓存，不仅凭行号假设配对。

## 真正的供体来源

- B1 patch：`../Final_audit/B1/seed_*/translation_d*_S*.npz`。
- B1 native：`../runs/v1_3/seed_*/analysis/baseline_d*_S*.npz`，成员/条件来自同目录 `behavior_states.npz`。
- B_reserved patch/native：`../Final_audit/B_reserved/seed_*/translation_d*_S*.npz` 与 `baseline_d*_S*.npz`，成员/条件来自同目录 `states.npz`。
- 原 S 的未翻译 patch 配供体 S；翻译 patch 配供体 T(S)，NA→NB、NB→NA、NN→NN。供体自己的历史与 p_a 由状态缓存验证，q 与原 manifest 匹配。

主 native 字段为保存的 `native_logp`（(m_a,c_a) 原生续跑）；`natural_logp` 为不间断对照，不与主数组混用。缺少缓存时不会用真值填供体预测，也不会运行网络补缓存。具体文件、字段、散列和实际 q 连接位置见 `analysis_plan.json` / `input_index.json`。

## 轻量逐查询记录与差异定位

`data/{B1,B_reserved}/seed_*/` 保留全部查询比较：

- `flags.npy`：uint16；轴为 `[d/S cell, treatment, nonself pair, canonical content, query_type, query_item]`。cell 顺序 d0/NN,NA,NB，然后 d10/NN,NA,NB；treatment 为 original,translated；pair 在 `analysis_plan.json` 中明确列出；内容由成员字典及 lexicographic permutation ID 唯一标识。
- `probability_differences.npy`：同轴，最后另有 `[max绝对概率差,TV]`，float64；不可计算时为 NaN。
- `answers.npy`：同轴，最后为 `[patch argmax,donor argmax,donor自身真值,精确相等类别概率元素数]`，uint8；缺失为 255。精确元素数量只在概率可比时解释。
- `orbit_blocks.npz`：每方向/d/S/处理/类型保留轨道分子、分母；完整源查询不被当独立 Bernoulli 样本。
- `differences_d*_S*_{original,translated}.npz`：答案分歧或 max概率差>1e−5 的显式记录，包含完整键、两侧预测与全部7类概率、供体真值/错误标志、max差与TV。未触发该标准的小浮点差并未丢弃，仍完整保存在轻量张量及源概率索引中。

flag 位：available=1，probability_available=2，agree=4，donor_correct=8，exact=16，patch_tie=32，donor_tie=64，close_1e6=128，close_1e5=256，close_1e4=512，patch_tie_known=1024，donor_tie_known=2048。缺失 tie 检测不能当作“没有 tie”。

每个张量平坦索引由可复现解码器还原完整语义键，并从已有源缓存取回两侧概率；不重复复制整个概率库。例如：

```bash
python -m experiments.memory_interchange.donor_consistency.inspect_record \
  --dataset B1 --seed 101 --flat-index 0
```

不一致 NPZ 的 `flat_index` 可以直接用于该命令；其他微小非零差异同样可通过 `probability_differences.npy` 定位。`input_index.json` 保存每个处理的源文件、字段、有效后缀、条件/内容连接与张量位置。概率由 float64 exp(logp)、检查原总和、逐行归一化得到；argmax 是相同的最小类别索引约定。

## 表与统计口径

- `seed_summary.csv`：六种子分别及等权汇总，未翻译/翻译、同/跨上下文、同/异历史、供体正确/错误子集；另有两处理合并的总描述。
- `strata_summary.csv`：固定全交叉 S/d/query类型/上下文/历史/处理/供体子集，包括不可实现或无观测的空格；原 S 与 effective_suffix 都保存。
- `answer_agreement_equal_cells`：每方向×d/S×query类型在固定子集的查询率等权，再在六种子等权。`answer_agreement_pooled`：实际总相同数/总查询数。all 完整等量时相同，错误子集或缺失时可不同。
- 概率均值保存 pooled 与 equal_cells 两种口径；中位数与 q95/q99 是合并实际查询分布，不是平均各模型的中位数。完整14/两个7查询向量一致率是预测向量相同，不是置换正确率；非法预测也保留。
- `block_intervals.json` 与两个 bootstrap NPZ：分别在19/51轨道重采样，每次六模型共用轨道权重并保留方向、处理、d/S/query。固定seed=20261013，每集合独立重置生成器，10,000次。区间条件于六模型，不含重训练或未知模型总体。
- `pairing_audit.json` / `pairing_coverage.csv`：唯一键/连接基数、已有/缺失/对角线对照、输入时序和模型/成员覆盖。
- `normalization_audit.json` / `NN_audit.json` / `supplemental_controls.csv`：有限值/归一化、NN同源同结果、已有 sham/原生/全状态对照。
- `control_state_differences.csv`：只从已保存状态得到的原始 c 差异，不生成新状态。
- `direct_index_validation.json`：冻结的小样本语义直读与向量化结果核对。
- `preservation_check.json`：旧模型、成员、报告、结果与失败记录散列；`artifact_manifest.json` 为全部新增交付散列。

原GPU对照通过、CPU float32原严格logit容差失败、同系数float64一致/argmax不变的注记继续保留。本次1e−6/1e−5/1e−4概率阈值仅描述差异，不能替换旧logit核验，不能作为新的科学成功门槛。完成缓存复核后收口，无外部发布。
