# EDANE — 动态属性网络高效嵌入

基于调研报告中 **EDANE**（Efficient Dynamic Attributed Network Embedding）算法设计思想的实验原型实现。

## 核心特性

- **显式优化初始化** — 稀疏随机投影锚点 + 目标函数迭代优化
- **局部增量更新** — 支持 `ΔV+ / ΔV- / ΔE+ / ΔE-` 与属性更新的工程化近似更新
- **可学习双曲门控融合** — 结构与属性在 Poincaré 球空间中自适应融合
- **量化压缩模块** — int8 主量化 + 可选 binary 副本 + 压缩误差统计
- **稀疏矩阵后端** — 基于 SciPy CSR，显著降低大图内存占用

## 项目结构

```
├── README.md                     ← 你在这里
├── requirements.txt              Python 依赖
├── run_all.bat                   一键实验（Windows）
├── all_results.csv               实验汇总表
│
├── docs/                         文档
│   ├── algorithm.md              算法理论说明
│   ├── development.md            开发指南
│   ├── evaluation_metrics_design_usage.md  评估指标设计说明
│   ├── getting_started.md        快速上手
│   ├── pipeline_usage.md         流水线详细用法
│   └── modules_1_4_integrated.md 模块一至模块四整合文档
│
├── dataset/                      项目内原始 OAG 压缩包目录
│   └── OAG/                      存放 v5_oag_publication_*.zip
│
├── data/                         流水线实际使用的数据目录
│   └── OAG/                      固定 CSV 数据集目录（file 模式读取这里）
│
├── src/                          源代码
│   ├── edane.py                  算法核心
│   ├── edane_full_pipeline.py    端到端实验流水线
│   ├── prepare_datasets.py       数据预处理
│   ├── plot_metrics_svg.py       SVG 曲线绘图
│   ├── run_edane_experiment.py   轻量快速验证
│   └── run_stage23_experiments.py 阶段2/3矩阵实验脚本
│
└── outputs/                      实验输出（按时间戳分目录，本地生成，默认不提交）
```

补充约定：

- 当前 file 模式固定使用项目根目录下的 `data/OAG/`；
- `dataset/OAG/` 仅保留 OAG 原始 zip，`prepare_datasets.py` 会将其转换到 `data/OAG/`；
- `dataset/` 与 `outputs/` 现在按本地数据/本地实验产物管理，不建议直接当作远程仓库正文提交。

数据来源说明：

- 固定数据集来源页：`https://open.aminer.cn/open/article?id=67aaf63af4cbd12984b6a5f0`
- 可安全写入项目说明的出处口径：该来源页托管于 **AMiner Open Platform（AMiner开放数据平台）**。

OAG 转换口径说明：

- 当前脚本构造的是一个 **OAG-derived 动态属性引文基准**；
- `references` 被映射为边，`year` 被映射为时间，`venue` 被映射为单标签，`title + abstract + keywords` 被映射为文本哈希特征；
- 这不是 OAG 官方任务定义的逐项复刻，而是为了适配当前统一的 `edges/features/labels` 流水线。

规模提醒：

- 转换脚本支持流式扫描 OAG zip；
- 但当前 `edane_full_pipeline.py` 在 file 模式下仍会一次性读入 CSV，因此**全量 OAG 转换结果未必适合在普通内存环境下直接跑全图**，通常需要结合 `--max-nodes` 使用。

## 对比算法（Baselines）与复现口径

本项目目前支持 3 个可运行的对比/基线模型，通过 `--model` 选择：

- `edane`：本项目算法原型（native 实现）
- `dane`：DANE（paper_approximation）
- `dtformer`：DTFormer-style（paper_approximation）

补充：本次调研/讨论中还提到了一个常见强基线 **MTSN（WWW 2021）**，它与 EDANE 的任务定义更接近，但由于其代码生态较旧（TF1/Python2），本仓库目前未直接接入。详见 `docs/baselines.md`。

### 1) DANE（CIKM 2017）

- 论文：Jundong Li, Harsh Dani, Xia Hu, Jiliang Tang, Yi Chang, Huan Liu. **Attributed Network Embedding for Learning in a Dynamic Environment**. CIKM 2017.
- DOI：10.1145/3132847.3132919；arXiv：https://arxiv.org/abs/1706.01860
- 代码位置：`src/dane.py`
- 复现口径：**paper-inspired 近似实现**（`implementation_fidelity=paper_approximation`）
  - 结构视图与属性视图采用谱嵌入 + 共识融合
  - 属性图使用 top-k cosine 相似图近似（避免全量相似度）
  - 在线更新：扰动式更新 + 不稳定时回退 refit
- 特色：同赛道经典强基线，任务定义接近动态属性网络嵌入
- 不足/风险：谱分解开销大，对大图不友好；数值稳定性依赖求解器与参数

### 2) DTFormer（CIKM 2024）

- 论文：Xi Chen et al. **DTFormer: A Transformer-Based Method for Discrete-Time Dynamic Graph Representation Learning**. CIKM 2024.
- DOI：10.1145/3627673.3679568；arXiv：https://arxiv.org/abs/2407.18523
- Official repo：https://github.com/chenxi1228/DTFormer
- 代码位置：`src/dtformer.py`
- 复现口径：**DTFormer-style 适配器**（`implementation_fidelity=paper_approximation`）
  - 官方 DTFormer 更偏离散时间动态图的链接预测/事件序列训练
  - 本项目为了统一到 “输出 embedding → 统一评估” 流水线，实现了 paper-inspired 的时序聚合与 patching 思路
- 特色：满足 “2024+ 前沿对比” 要求
- 不足/风险：任务/训练目标不完全同构，因此需要在论文中明确是适配版 baseline

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成 OAG 子图（推荐先做可跑子集）
python src/prepare_datasets.py --convert-oag --subset-profile test --overwrite

# 生成更接近正式实验的 OAG 子图（示例：节点>=15000，优先保边）
python src/prepare_datasets.py --convert-oag --overwrite \
  --selection-strategy dense --max-papers 15000 --candidate-multiplier 3 \
  --min-venue-support 5 --keep-unlabeled --max-record-bytes 2000000

# 或仅检查固定数据集 data/OAG
python src/prepare_datasets.py --validate-only

# 3. 运行实验
python src/edane_full_pipeline.py --mode synthetic --model edane --quantize

# EDANE / DANE / DTFormer 对比（统一在 data/OAG 上）
python src/edane_full_pipeline.py --mode file --model edane --snapshots 4 --max-nodes 15000
python src/edane_full_pipeline.py --mode file --model dane --snapshots 4 --max-nodes 15000
python src/edane_full_pipeline.py --mode file --model dtformer --snapshots 4 --max-nodes 15000

# 消融实验
python src/edane_full_pipeline.py --mode file --snapshots 6 --quantize --no-attr
python src/edane_full_pipeline.py --mode file --snapshots 6 --quantize --no-hyperbolic
python src/edane_full_pipeline.py --mode file --snapshots 6 --quantize --no-inc

# 或一键运行全部
run_all.bat
```

## 输出示例

每次运行在 `outputs/<数据集名>_<时间戳>/` 下生成（一眼看出哪个实验）：

| 文件 | 说明 |
|------|------|
| `summary.json` | 实验摘要（时延、F1、AUC/AP、重构AUC、压缩比/误差） |
| `metrics_per_snapshot.csv` | 逐快照指标（含 `link_ap`、`reconstruction_auc`） |
| `metrics_curves.svg` | 曲线图（可直接用于论文） |
| `final_embedding.npy` | 最终嵌入向量 |
| `final_embedding_int8.npy` | int8 量化副本（开启 `--quantize` 时） |
| `final_embedding_scale.npy` | int8 缩放系数 |
| `final_embedding_binary.npy` | binary 副本（开启 `--binary-quantize` 时） |

### 阶段2/3一键矩阵实验

```bash
# 阶段2：更新频率 10/100/1000；阶段3：full + 3个消融自动汇总
python src/run_stage23_experiments.py --mode file --snapshots 3 --max-nodes 3000

# 快速自检（synthetic）
python src/run_stage23_experiments.py --mode synthetic --snapshots 2 --stage2-rates 10,100 --synthetic-rounds 6
```

## 全量 OAG 运行建议

如果你要尝试全量 OAG，请使用：

```bash
python src/prepare_datasets.py --convert-oag --subset-profile full --overwrite
```

但要注意：

- 这更适合高内存/长时间任务环境，不适合普通开发机；
- 建议至少使用 **128GB+ 内存、NVMe SSD、长时可持续运行环境**（更理想：256GB+）；
- 本项目当前 file 模式会整读 `edges.csv/features.csv/labels.csv`，因此“全量转换成功”并不等于“可直接全量训练/评估”。

更现实的全量可行方案（推荐写进论文复现说明）：

1. **大机器完成全量转换 → 再做采样子图**（推荐）
   - 在服务器完成 `--subset-profile full` 或更大规模转换/索引
   - 再按论文实验设定抽取可训练的子图（例如 1e5~1e6 节点级别）
2. **重构加载逻辑为流式/分块**（工程量大）
   - 让 `build_graph_from_files()` 在 `--max-nodes` 前就能完成预筛选/采样
   - 避免全量 CSV 先进入内存
3. **改用更适合超大图的图学习框架/存储**（工程量大）
   - 例如将数据转为 Parquet/Arrow，或使用图数据库/图采样器，再进入训练

输出在 `outputs/stage23_matrix_oag_<timestamp>/`：

- `stage2_rate_results.csv`（阶段2速率矩阵）
- `stage3_ablation_results.csv`（阶段3消融矩阵）
- `stage23_combined_results.csv`（合并表）

## 文档导航

| 文档 | 内容 |
|------|------|
| [快速上手](docs/getting_started.md) | 环境搭建、首次运行 |
| [流水线用法](docs/pipeline_usage.md) | 完整命令、参数、数据格式 |
| [算法说明](docs/algorithm.md) | 理论背景、数学推导 |
| [开发指南](docs/development.md) | 项目架构、扩展方法 |
| [评估指标文档](docs/evaluation_metrics_design_usage.md) | Macro/Micro-F1、AUC、AP、重构与时延指标说明 |
| [评估设置对比](docs/evaluation_setting_comparison.md) | raw-label 与 cleaned-label 两种分类评估口径的对比与解释 |
| [模块整合文档](docs/modules_1_4_integrated.md) | 模块一至模块四的统一设计说明 |
| [对比算法说明](docs/baselines.md) | DANE/DTFormer 等基线的复现口径与限制 |
| [OAG全量可行方案](docs/oag_full_scale.md) | OAG 采样/全量转换与硬件建议 |

## 环境要求

- Python ≥ 3.9
- NumPy
- SciPy（稀疏矩阵）

## 当前实现定位

当前项目是一个**可运行的 EDANE 研究原型**，已经覆盖：

- 数据预处理
- 初始化训练
- 动态增量更新
- 结构/属性融合
- 节点分类与链路预测评估
- 实验结果导出

但它仍然是：

- 单机原型优先；
- NumPy/SciPy 主导，PyTorch 仅为可选 dense 后端；
- 面向毕业设计/课程实验/方法验证；
- 不是工业级分布式生产系统。
