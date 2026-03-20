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
├── all_results.csv               多数据集汇总表
│
├── docs/                         文档
│   ├── algorithm.md              算法理论说明
│   ├── development.md            开发指南
│   ├── evaluation_metrics_design_usage.md  评估指标设计说明
│   ├── getting_started.md        快速上手
│   ├── pipeline_usage.md         流水线详细用法
│   └── modules_1_4_integrated.md 模块一至模块四整合文档
│
├── dataset/                      项目内原始数据目录（本地使用，优先被 prepare_datasets.py 读取）
│
├── src/                          源代码
│   ├── edane.py                  算法核心
│   ├── edane_full_pipeline.py    端到端实验流水线
│   ├── prepare_datasets.py       数据预处理
│   ├── plot_metrics_svg.py       SVG 曲线绘图
│   ├── run_edane_experiment.py   轻量快速验证
│   └── run_stage23_experiments.py 阶段2/3矩阵实验脚本
│
├── data/                         预处理后的数据集
│   ├── reddit_sample/
│   ├── amazon2m_sample/
│   ├── amazon3m_sample/
│   ├── mag_sample/
│   └── twitter_sample/
│
└── outputs/                      实验输出（按时间戳分目录，本地生成，默认不提交）
```

补充约定：

- 当前推荐把原始数据放在项目根目录下的 `dataset/`；
- `prepare_datasets.py` 当前按项目内 `dataset/` 作为默认原始数据目录；
- `dataset/` 与 `outputs/` 现在按本地数据/本地实验产物管理，不建议直接当作远程仓库正文提交。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据（优先从项目内 dataset/ 生成样本）
python src/prepare_datasets.py --prepare-reddit --prepare-amazon --prepare-amazon3m --prepare-mag --prepare-twitter

# 3. 运行实验
python src/edane_full_pipeline.py --mode synthetic --quantize
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 6 --quantize
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 6 --quantize --binary-quantize
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 6 --quantize --update-rate 100
python src/edane_full_pipeline.py --mode file --dataset-preset mag_sample --snapshots 6 --quantize
python src/edane_full_pipeline.py --mode file --dataset-preset twitter_sample --snapshots 6 --quantize
python src/edane_full_pipeline.py --mode file --dataset-preset amazon3m_sample --snapshots 6 --quantize

# 消融实验
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 6 --quantize --no-attr
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 6 --quantize --no-hyperbolic
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 6 --quantize --no-inc

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
python src/run_stage23_experiments.py --mode file --dataset-preset reddit_sample --snapshots 3 --max-nodes 3000

# 快速自检（synthetic）
python src/run_stage23_experiments.py --mode synthetic --snapshots 2 --stage2-rates 10,100 --synthetic-rounds 6
```

输出在 `outputs/stage23_matrix_<dataset>_<timestamp>/`：

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
| [模块整合文档](docs/modules_1_4_integrated.md) | 模块一至模块四的统一设计说明 |

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
