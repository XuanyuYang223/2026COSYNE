# Memory-state interchange experiment

本轮状态：**BLOCKED — 两个预先指定的 n=7 pilot 均未学会原任务。**
默认 128 维与唯一备用 256 维均完成 20,000 次更新，最佳固定验证准确率分别为 14.7827% 和 14.8926%，门槛为 97%。已按停止规则保留失败结果；正式六种子实验、H2/H3 留出预测和未见四步词扩展均未运行。

详见 [report.md](report.md)。所有 pilot 几何和交换数据仅用于实现、吞吐和失败诊断，不是科学结论。

## 任务与主问题

检验相同完整记忆内容、不同形成历史的自然 RNN 状态，其几何相似度是否预测保留接收方控制状态后的功能可互换性。

置换 `p[i]` 表示项目 i 对应的位置；`I(p)[p[i]]=i` 是双射反向后的数组表示，`C(p)=n-1-p` 是对应位置镜像。上下文 0 中 A/B 表示 I/C，上下文 1 中 A/B 表示 C/I；N 始终保持。inverse 不描述实际物体移动。

上下文展示 3 步、完整刺激 5 步、空白 5 步；四个规则槽各 1 步线索 + 3 步空白。前两槽后保持 d∈{0,10}，在第 21+d 步结束时交换。后两槽与最后 5 步空白之后，查询在第 35+d 步首次出现，在第 37+d 步读出。每个内容枚举项目/位置两类查询的全部 7 个项目。

单向控制 GRU 不接收置换和 m；记忆 GRU 接收初始刺激、当前 c 和阶段标志。训练只用最终查询交叉熵。m/c 是人为指定模块，不能据此证明自然脑区分工或网络自发形成群结构。

## 运行

从仓库根目录 `/home/yangx/2026COSYNE` 执行。使用能支持本机 GPU 的 PyTorch CUDA wheel；本轮为 torch 2.11.0+cu128。

```bash
python -m pip install -r experiments/memory_interchange/requirements.txt
python -m experiments.memory_interchange.truth
python -m pytest experiments/memory_interchange/tests -q
python -m experiments.memory_interchange.run
```

最后一条命令会复用已完成且训练配置一致的 pilot，生成报告并按门槛决定是否进入正式六种子；本轮再次执行会保留 BLOCKED 状态，不会自动尝试其他配置。未完成的训练从 `latest.pt` 恢复，包含优化器和所有随机数状态。没有 GPU 时只运行数学检查与短 CPU smoke test。

分阶段复现：

```bash
python -m experiments.memory_interchange.train --seed 0 --out experiments/memory_interchange/runs/pilot_128 --resume
python -m experiments.memory_interchange.train --seed 0 --memory-dim 256 --out experiments/memory_interchange/runs/pilot_256 --resume
python -m experiments.memory_interchange.diagnose --run experiments/memory_interchange/runs/pilot_128
python -m experiments.memory_interchange.intervene --run experiments/memory_interchange/runs/pilot_128 --pilot
python -m experiments.memory_interchange.geometry --run experiments/memory_interchange/runs/pilot_128 --pilot
```

正式入口受 pilot、实现检查、预算与源码冻结约束。`predict` 要求六个规定种子完整标签，拒绝把 pilot 或缺少种子的结果写作正式分析。`extension` 要求正式主实验原始行为门槛通过。本轮无需运行这两个入口。

## 数据与分析约束

split_seed=20261012；先枚举 I/C 闭包轨道再划分。T/V/G/B 为 451/104/69/70 条轨道，实际置换数为 3274/778/492/496。正式 B 子集是固定轨道顺序首次达到 128 个置换的完整轨道，共 134 个置换；未按任何模型结果选择。

主几何 CKA、原生距离、中心化正交 Procrustes 在 G 的全部内容计算；主干预在固定 B 子集计算。错误内容采用固定跨轨道 derangement；同时保存原答案和供体在接收方规则下的反事实答案。sham 和同上下文全状态仅为实现对照。

`L=mean(max(0,ell_swap-ell_native))` 与 `R=mean(native正确且swap错误)` 同时报告，双向损害不能正负抵消。ridge alpha=1，所有标准化与缺失值处理仅在拟合折内进行。主预测对照为 M0 与 M_CKA；原生距离、Procrustes 和初始化 CKA 是预定比较。外层留出种子、训练排除涉及 AB/BA 的 62 个配对；主要测试为其中 32 个跨上下文配对。未见的是预测器的交换标签，任务网络训练分布包含这些历史。

反例规则固定为每个 seed/d/S/上下文类别（0/0、0/1、1/1）内 CKA 与 R 均达到第 75 分位数、且 R≥0.5%；展示全部合格候选并保存完整分布。内容探针 alpha=1，只在 T 的平衡条件/保持时间缓存上拟合。训练各类别采用均匀随机抽样，批内计数允许随机波动。所有正式种子均保留，包括原任务未过门槛者。

## 文件与数组索引

| 文件 | 内容 |
|---|---|
| `protocol_v1.txt` / `config.yaml` | 原始方案和完整参数 |
| `split_manifest.json` / `.sha256.json` | 全部轨道、成员、划分、B 子集和校验和 |
| `truth.py` / `task.py` | 数组真值、独立矩阵验证、固定时序与抽样 |
| `model.py` / `train.py` | 学习得到的单向 m/c GRU、检查点与可恢复训练 |
| `cache.py` / `intervene.py` | 自然状态、所有对照、真值与轨道统计 |
| `geometry.py` / `predict.py` | float64 几何、T 探针、LOSO 与层级 bootstrap |
| `diagnose.py` / `figures.py` / `report.py` | V 失败诊断、可导出图、报告与产物散列 |
| `run.py` / `extension.py` | 有限预算、pilot 停止、冻结与条件扩展 |

条件编号 0–8 为 r=0、9–17 为 r=1；历史顺序 `NN,NA,AN,NB,BN,AA,BB,AB,BA`。缓存 m 轴为 `[d,条件,q,神经元]`，c 为 `[d,条件,神经元]`。所有数组保存内容和 orbit_id，可追溯到 manifest。

每个 `raw_d*_S*.npz` 的 native/sham 轴为 `[接收方,q,查询类型,查询项目,输出类别]`；matched/wrong 为 `[供体,接收方,q,查询类型,查询项目,输出类别]`。query_type=0/1 分别为项目/位置查询。`logp` 是 float32 自然对数概率。full 另存供体/接收方索引；只能解释同上下文对照。

`orbit_blocks.npz` 的 `metric_sums` 轴为 `[d/S,供体,接收方,轨道,查询类型,指标]`；`query_counts` 为每轨道每种查询类型的数量。过滤答案不同子集时，`wrong_*_different` 是正确数的和，必须除以 `answers_differ` 的和。原始 NPZ 可恢复每个具体查询；CSV 给出方向、查询类型与配对汇总。初始、选定和最新检查点、训练曲线及 V 验证逐试次预测全部保留在 `runs/`。

生成的大文件在 `.gitignore` 中排除，但本地交付完整保留；`results/artifact_manifest.json` 记录每个产物的 SHA-256。不要仅复制源码后声称包含本轮实验数据。

12 GPU 小时是累计上限；不自动扩大预算、调噪声制造损害、增加架构或在失败后挑种子。6 个模型种子的统计仍仅描述有限模型集合。正式数据缺失时不会填入预测效果或显著性。
