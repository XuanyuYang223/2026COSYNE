# 记忆状态交换实验：中文文档

这里汇集仓库的中文阅读版本。原有报告正文大多已为中文，本次补齐英文标题、表格标签、说明文字和常用术语。数值、公式、命令、代码、数据字段名及原图保留，图中的英文术语可查[中文对照表](术语与字段.md)。

每篇文档顶部都可返回冻结原件。原版本的状态、当时的执行范围及“未推送”等历史叙述保留其时间含义。早期受阻报告与后续已完成复核分别对应各自版本；最终核验仍保留 CPU 单精度失败记录。

## 最新结果与执行说明

| 工作 | 中文报告 | 中文执行说明 |
| --- | --- | --- |
| 状态交换与供体原生输出一致性复核 | [复核报告](experiments/memory_interchange/donor_consistency/report.md) | [执行与数据说明](experiments/memory_interchange/donor_consistency/README.md) |
| v1.3 最终三项复核 | [最终复核报告](experiments/memory_interchange/Final_audit/report.md) | [执行与产物说明](experiments/memory_interchange/Final_audit/README.md) |
| v1.3 正式实验 | [正式报告](experiments/memory_interchange/results/v1_3/report.md) | [版本说明](experiments/memory_interchange/v1_3/README.md) |

供体一致性复核比较的是替换输出与供体实际输出；供体规则反事实比较的是真值；线索翻译复核比较的是原目标能否恢复。这三种指标分别回答不同问题。两套 B 数据在最新一致性复核前均已用于分析。

## 正式结果的详细阅读

- [结果解释与数值稳定性核验](experiments/memory_interchange/results/v1_3/interpretation.md)
- [全部正式模型的原始行为因素](experiments/memory_interchange/results/v1_3/all_behavior_factors.md)
- [原始质量档与条件样本掩码](experiments/memory_interchange/results/v1_3/quality_and_mask_summary.md)
- [当前内容与初始输入反事实](experiments/memory_interchange/results/v1_3/counterfactual_all_and_distinct.md)
- [相关方法定位](experiments/memory_interchange/results/v1_3/method_context.md)
- [先导模型验证集探索报告](experiments/memory_interchange/results/v1_3/V_exploration/report.md)
- [供体一致性摘要句的中文译文](experiments/memory_interchange/donor_consistency/english_result.txt)

## 历史版本与研究方案

| 版本 | 说明 | 报告 | 方案与修订 |
| --- | --- | --- | --- |
| v1.0 | [原始实验说明](experiments/memory_interchange/README.md) | [历史受阻报告](experiments/memory_interchange/report.md) | [原始研究方案](experiments/memory_interchange/protocol_v1.txt) |
| v1.1 | [版本说明](experiments/memory_interchange/v1_1/README.md) | [历史受阻报告](experiments/memory_interchange/results/v1_1/report.md) | [方案](experiments/memory_interchange/v1_1/protocol_v1_1.txt)；[修改记录](experiments/memory_interchange/v1_1/CHANGES.md) |
| v1.2 | [版本说明](experiments/memory_interchange/v1_2/README.md) | [阶段七受阻报告](experiments/memory_interchange/results/v1_2/report.md) | [修改记录](experiments/memory_interchange/v1_2/CHANGES.md) |
| v1.3 | [版本说明](experiments/memory_interchange/v1_3/README.md) | [正式报告](experiments/memory_interchange/results/v1_3/report.md) | [方案](experiments/memory_interchange/v1_3/protocol_v1_3.txt)；[修改记录](experiments/memory_interchange/v1_3/CHANGES.md) |

## 文件与复现

源码、配置、JSON/CSV 字段、检查点及原始审计文件保留原路径和格式，以便执行已有命令及散列核验。中文文档的来源和 SHA-256 见[翻译清单](translation_manifest.json)。生成器位于 [tools/build_chinese_docs.py](../../tools/build_chinese_docs.py)，只更新中文阅读版。

```bash
python tools/build_chinese_docs.py
python tools/build_chinese_docs.py --check
```

复现数据的包含范围和二进制恢复要求见[仓库首页](../../README.md)。原报告中的命令有的会执行训练或网络前向；本次中文化没有运行这些命令，也没有重新计算实验结果。
