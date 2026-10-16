# 2026COSYNE：记忆状态交换实验

本仓库收录记忆状态交换研究的实验代码、研究方案、配置、训练记录、报告、汇总表、图表和完整性校验清单。

最新完成的工作是在六个冻结的循环神经网络（RNN）中，比较记忆状态替换后的输出与供体原生输出的一致性。B1 和 B_reserved 此前均已用于分析，因此这次复核属于描述性分析。原有中心化核对齐（CKA）预测检验及此前未通过的数值核验记录均予以保留。

## 中文阅读入口

- [全部中文文档与版本导航](docs/zh-CN/README.md)
- [术语、数据字段与统计口径对照](docs/zh-CN/术语与字段.md)
- [供体原生输出一致性复核，2026-10-13](docs/zh-CN/experiments/memory_interchange/donor_consistency/report.md)
- [v1.3 最终复核](docs/zh-CN/experiments/memory_interchange/Final_audit/report.md)
- [v1.3 正式实验](docs/zh-CN/experiments/memory_interchange/results/v1_3/report.md)
- [v1.3 结果解释](docs/zh-CN/experiments/memory_interchange/results/v1_3/interpretation.md)
- [v1.2 报告](docs/zh-CN/experiments/memory_interchange/results/v1_2/report.md)
- [v1.1 报告](docs/zh-CN/experiments/memory_interchange/results/v1_1/report.md)

早期报告中的 BLOCKED（受阻）状态描述对应版本的历史结果；最新复核的状态见其报告。六个预定种子全部保留，包括种子 106。登记的 GPU 对照通过；CPU float32 的原严格 logit 容差核验未通过；使用相同系数的 float64 结果一致，所比较的最大概率类别（argmax）未改变。

中文阅读版翻译了说明文字、表格标签和术语，并保留数值、公式、命令与程序标识。历史实验文档原件及其散列保留在原路径；中文版本位于独立的 `docs/zh-CN/` 目录，每篇均链接原件。图像沿用原图，图中文字的常用术语可查对照表。

## 代码与复现

实验代码位于 [experiments/memory_interchange](experiments/memory_interchange)。各版本均保留对应的研究方案和执行说明。最新只读复核的命令见[供体一致性复核执行说明](docs/zh-CN/experiments/memory_interchange/donor_consistency/README.md)；此前最终复核的命令见[最终复核执行说明](docs/zh-CN/experiments/memory_interchange/Final_audit/README.md)。

**Git 仓库包含源码与报告快照，完整复现还需要本地二进制产物。** 仓库已包含六个正式模型在课程开始时的 `initial.pt` 及冻结的 `selected_stage7.pt`，共十二个检查点，合计约 9.4 MiB；同时包含源码、文本元数据、已有表格和图表，以及原始自助重采样记录。

其余检查点、自然状态缓存、原始概率数组、逐查询比较张量和归档快照共约 9.57 GiB，保留在原工作区。`.gitignore` 排除 `*.pt`、`*.npz`、`*.npy` 和 `*.tar.gz`，并明确放行上述十二个正式模型检查点。历史文件逐字节保留，包括 CSV 的原换行格式。

复现冻结输出的复核前，需要按原相对路径恢复对应二进制文件，并校验 SHA-256。文件名、集合成员、输入映射和散列见：

- [供体一致性分析计划](experiments/memory_interchange/donor_consistency/analysis_plan.json)与[输入索引](experiments/memory_interchange/donor_consistency/input_index.json)
- [最终复核产物清单](experiments/memory_interchange/Final_audit/artifact_manifest.json)
- [v1.3 产物清单](experiments/memory_interchange/results/v1_3/artifact_manifest.json)
- [未纳入 Git 的本地产物清单](LOCAL_ARTIFACTS.json)

新克隆的仓库包含上述十二个正式检查点；复核所需的输出缓存和归档训练状态仍需另行恢复。上传时未删除本地二进制产物。本次中文化只更新阅读文档和说明，不运行训练或新增科学分析。

## 更新中文文档

```bash
python tools/build_chinese_docs.py
python tools/build_chinese_docs.py --check
```

生成器只写入中文阅读版目录；校验命令检查翻译与来源、数值、代码块、程序标识及链接的一致性。
