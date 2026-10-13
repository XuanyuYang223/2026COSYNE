# v1.3 Final audit

这是用户授权的三项有限复核。六个正式 RNN 不再训练；旧 v1.3 指标、点结果、全部 bootstrap、seed 106 和 CPU float32 严格容差失败均保留。主报告为 [report.md](report.md)。

## 复现 / 恢复

在仓库根目录，使用原 Python/Torch 环境：

```bash
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m experiments.memory_interchange.final_audit.common
python -m experiments.memory_interchange.final_audit.behavior native >> experiments/memory_interchange/Final_audit/native.log 2>&1
python -m experiments.memory_interchange.final_audit.validate
python -m experiments.memory_interchange.final_audit.behavior CF
python -m experiments.memory_interchange.final_audit.geometry_probe
python -m experiments.memory_interchange.final_audit.statistics
python -m experiments.memory_interchange.final_audit.behavior patch >> experiments/memory_interchange/Final_audit/translation.log 2>&1
python -m experiments.memory_interchange.final_audit.behavior summary
python -m experiments.memory_interchange.final_audit.summarize
python -m experiments.memory_interchange.final_audit.finish check
python -m experiments.memory_interchange.final_audit.finish cost
python -m experiments.memory_interchange.final_audit.report
python -m experiments.memory_interchange.final_audit.finish inventory
```

`native` 完成所有模型的原生输出和 masks，随后才允许 `patch`。已完成模型的 GPU 输出不会重算。`CF` 重用全部原 B1 matched/wrong 输出；新 B1 translated patch 单独前向重放，验证与原 S′ 输出一致。B_reserved 原/翻译 patch 均新计算。精确重做应使用隔离的工作区并保留旧版本完整输入，而非覆盖原报告或检查点。GPU 总预算继承旧账并受 12 小时约束。

## 冻结与来源

- `frozen_plan.json`、`revision_final_audit.yaml`、`freeze_receipt.json`：新增假设、两个目标、所有条件、mask、主要对比、固定 probe/statistics 规则。
- `B_reserved_members.npz` / `B_reserved_manifest.json`：原 B 减 B1 的确切成员，362 内容 / 51 完整轨道；所有剩余轨道使用。B1 为探索，reserved 为翻译确认。
- `all_native_masks_frozen.json` 和每种子 `mask_freeze.json`：新增 patch 前保存的 native masks。
- `preservation_manifest.json` / `preservation_check.json`：旧 v1.0–v1.3 源码、配置、检查点与报告散列。
- `artifact_manifest.json`：新代码与产物完整散列；不登记易变的 Python bytecode。
- `budget.json` / `cost_windows.json`：GPU 计账和包括初始化、保存、CPU 归约/IO、完成观察滞后的保守进程窗口；实际繁忙 GPU 时间更小。

## 数组与口径

`B1/seed_{101..106}` 与 `B_reserved/seed_{101..106}` 全部保留。条件顺序沿用原 `truth.CONDITIONS`：r=0 的九个 H，随后 r=1 的九个 H；H 为 NN, NA, AN, NB, BN, AA, BB, AB, BA。

`translation_d*_S*.npz` 的两个 patch log-probabilities 轴为 `[donor,receiver,content,query_type,query_id,class]`，shape `[18,18,Q,2,7,7]`。query_type=0 是项目→位置，1 是位置→项目；argmax 前两个查询轴重建正/反向映射。`Q_original=F_b(S)(q)` 和 `Q_literal=F_b(swap(S))(q)` 为 `[condition,Q,2,7]`；供体候选按 donor condition 取相同数组，接收候选按 receiver 取。无序条件对的两个方向均保存；对角线虽保存在数组中，不计主要描述。

`counterfactual_d*_S*.npz` 仅 B1：全部单查询和完整映射预测、三匹配候选、两错误内容候选、每对候选不同标记、三候选两两不同标记、内容/轨道与固定错配 index。标记的补集就是相应重合关系。错误内容没有初始输入候选；CSV 的该列留空。

`masks_d*_S*.npz` 的 all/rec/both/E_sem 同为 `[18,18,Q,2,7]`。同一个 mask 用于两个 patch，不按处理结果选择样本。E_sem 由 receiver native 原 S 正确和 donor native 相同语义计算正确定义：cross 使用 donor translated S，same 使用 donor 原 S。`both` 只要求各自对同一可见 S 答对。

`blocks_d*_S*.npz`：`sums[306 directions,4 masks,2 treatments,orbits,8 metrics]`、`counts[306,4,orbits]`。指标含原目标准确率、CE、新增错误、正向损害、新字面准确率/CE、接收/供体规则符合率。整轨道保存所有查询和方向；主要 all 对比为种子等权、方向等权、d/S/query 原分布。条件描述先按每方向固定 mask 汇总，再等权方向、d、S、种子。数量是观察计数，不是独立样本量。

## 表与检查

- `translation_seed_summary.csv` / `translation_overall_summary.csv`：所有 masks、同/跨上下文、S、d；每个原/翻译准确率、CE、新增错误及差。详细 query_type/方向/计数在各 `translation_d*_S*.csv`。
- `translation_full_mapping_diagnostics.csv`：双射、互逆、原/字面/供体完整映射符合率，全部及映射可区分子集。
- `candidate_*summary.csv`：每对候选可区分与三候选不同子集，数量、CE、单查询和完整映射符合率；原始分层表在 B1 各种子目录。
- `native_all_strata.csv`：全部 H/r/d/S/query 类型的原目标与新字面目标，含 native、native translated、sham；`native_seed_summary.csv` 展示 106 差条件。
- `geometry_decomposition.csv` / `geometry_distributions.csv`：去重的 G CKA 分布、D_raw 均值/中心分解和独立数值关系；`original_fixed_counterexamples.csv` 保留旧固定规则全部 288 个案例。
- `context_probe_results.csv`、`context_raw_contrasts.csv`、`context_probe_fit_inventory.csv`：固定 T-only、非混合 H 拟合、AB/BA 留出，pooled 和双单 d 模型；c 不重复 q。每种子的参数/分数/边际在 `context_probes/`。
- `bootstrap_replay_metadata.json`、`bootstrap_feature_diagnostics.csv`、`bootstrap_replay_scores.csv`：典型与极端原副本完整审计，无 held-seed 副本泄漏，M0/CKA 共享重采样、标签与测试单位。未改 alpha、方差地板或预测。
- `fixed_prediction_summary.json`：固定原六个 LOSO 模型/测试预测的成对 fold 分数和条件经验区间，不替代 refit bootstrap，不含未知模型总体/重新训练不确定性。
- `pre_patch_checks.json` / `post_patch_checks.json`：全 5,040 置换的翻译数学关系、mask 冻结与语义、NN 精确不变、独立点归约与块分母验证。

几何、线性可读性、外部语义翻译补偿和真实脑机制是不同层次的证据；保留物理 c 不自动意味着保留接收方功能规则。完成本清单后停止，没有参数扫描、适配器、消融或四步序列扩展。
