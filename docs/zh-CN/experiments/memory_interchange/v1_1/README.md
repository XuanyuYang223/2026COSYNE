> 中文阅读版 · [冻结原件](../../../../../experiments/memory_interchange/v1_1/README.md) · [中文导航](../../../README.md)
>
> 原版本的时间、状态和结论按原记录保留；代码、字段标识、公式和原图沿用原件。

# v1.1：任务学习修订

当前状态：**受阻（BLOCKED），候选 A/B 均在阶段 6 的两步计算门槛失败。** 前五阶段通过，阶段 7 未运行；不属于任务就绪（TASK_READY） 或正式科学结果。

[完整报告](../results/v1_1/report.md)保留两版结果、训练路径审计、全部课程阶段、真实操作因素、梯度/内容/门诊断及计算成本。[修改记录](CHANGES.md)说明与 v1.0 的区别。

## 范围与保留

v1.0 的源码、报告、检查点、数据和原划分均保持原位，101 个原文件另有归档及 SHA-256。正式 B 的 134 个内容完全复用；本轮未对 G/B 运行任何模型、几何/损害分析或科学预测器。

v1.1 新源码在本目录；结果在 `results/v1_1/`，检查点在 `runs/v1_1/`。旧 GPU 成本从不可变 v1.0 记录继承，与本轮增量累计受 12 GPU 小时上限约束。

## 复现入口

从仓库根目录运行，沿用 v1.0 的 torch 2.11.0+cu128、NumPy 和依赖环境。

```bash
python -m experiments.memory_interchange.v1_1.run
```

该入口验证旧产物不变、检查固定数据和实现、复用已完成的审计/从头训练/课程记录。当前失败状态下再次运行不会追加模型训练，不会自动进入阶段 7 或正式种子。

分阶段入口：

```bash
python -m experiments.memory_interchange.v1_1.audit
python -m pytest experiments/memory_interchange/tests experiments/memory_interchange/v1_1/tests -q
python -m experiments.memory_interchange.v1_1.train scratch --candidate A
python -m experiments.memory_interchange.v1_1.train curriculum --candidate A
python -m experiments.memory_interchange.v1_1.train curriculum --candidate B
```

只有 A 的课程失败，或 A 小型拟合失败且 B 小型拟合通过，才允许 B 课程。A 通过时不试 B。连续晋级点、失败上限及优化器/RNG 的中断恢复均受检查点状态约束。

## 固定方法

任务真值、输入编码、m/c 单向结构和最终查询 CE 均复用 v1.0；本轮结构固定 128/32/64。上下文和取逆/位置镜像仍仅在数据生成/独立真值核验器中解释，不进入网络权重公式。

| 阶段 | 时序 | 训练词/上下文 | 上限 |
|---:|---|---|---:|
| 1 | 9 步短时序 | NNNN，r=0 | 5,000 |
| 2 | 23 步中等时序 | NNNN，r=0/1 | 5,000 |
| 3 | 37/47 步原始 | NNNN，r=0/1，d=0/10 | 5,000 |
| 4 | 原始 | k≤1，r=0 | 5,000 |
| 5 | 原始 | k≤1，r=0/1 | 5,000 |
| 6 | 原始 | k≤2，r=0/1 | 5,000 |
| 7 | 原 v1.0 分布 | 完整 65 词，k=0/1/2/3 均匀 | 20,000 |

候选 A 默认初始化；唯一 B 初始时设两个 GRU 的有效更新门值偏置 +3，所有参数继续自由学习。每个候选从种子=0 新初始化开始，从头训练独立、不转移权重；阶段晋级用同一检查点的权重、Adam 和全部 RNG。

每 500 更新评价固定 V，阶段 1–6 各 4,096 试次，阶段 7 必须使用未改动的原 V 8,192 试次。总准确率≥97% 且上下文、真实操作序列、d、查询类型、非 N 数的各因素层≥95%，连续两次才晋级。没有任意细交叉格门槛，也没有下降行为阈值。

所有新评价集合在训练前生成并保存实际批次整理输入及真值核验器标签。额外平衡 T/V 抽样种子=20261013，阶段 V 种子=20261014；新增平衡 V 不替换原 V。`datasets/manifest.json` 保存文件散列和各因素计数。

`audit.py` 检查旧最佳/最新的 T/V、刺激移除/打乱响应、填充和实际训练前向计算。`trace.py` 用实际 GRU 的逐步携带节点 retain_grad，比较融合实现 logit/参数/输入导数，并保存 None/有限性/范数；短试次 CPU float64 有限差分检查独立于长时序小导数。`data.py` 的真值核验器只从实际输入恢复语义，不调用标签更新函数。

每阶段 `curve.json` 保存每 500 步 T/V 分层与计数，`diagnostics.json` 保存完整评价响应、64 试次的状态敏感性、四类阶段的记忆/控制更新 gates 及梯度。所有 `logit/CE/gradient/L2` 指标的诊断用途与行为门槛分开；没有用它们挑几何指标或制造交换损害。

只有阶段 7 成功时，`ready.py` 才可做 T-only 岭回归内容探针与 V 原任务原始行为分层；即使到任务就绪（TASK_READY），本轮也不自动开展正式科学实验。初始 CKA 未来必须使用课程开始前初始化；科学缓存时点、G/B、交换和预测定义仍按 v1.0。
