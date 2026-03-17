# EDANE — 动态属性网络高效嵌入

基于调研报告中 **EDANE**（Efficient Dynamic Attributed Network Embedding）算法设计思想的实验原型实现。

## 核心特性

- **稀疏随机投影初始化** — 避免显式构造高阶邻近矩阵，O(q·|E|·d) 复杂度
- **增量动态更新** — 仅更新受影响节点及一跳邻居，无需全图重训练
- **双曲注意力融合** — 结构与属性在 Poincaré 球空间中自适应融合
- **int8 量化存储** — 约 8× 压缩比，精度损失极小
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
│   ├── getting_started.md        快速上手
│   ├── pipeline_usage.md         流水线详细用法
│   └── thesis_evaluation_conclusion_auto.md  论文结论（自动生成）
│
├── src/                          源代码
│   ├── edane.py                  算法核心
│   ├── edane_full_pipeline.py    端到端实验流水线
│   ├── prepare_datasets.py       数据预处理
│   ├── plot_metrics_svg.py       SVG 曲线绘图
│   ├── generate_thesis_conclusion.py  论文结论自动生成
│   └── run_edane_experiment.py   轻量快速验证
│
├── data/                         预处理后的数据集
│   ├── reddit_sample/
│   ├── amazon2m_sample/
│   ├── amazon3m_sample/
│   ├── mag_sample/
│   └── twitter_sample/
│
└── outputs/                      实验输出（按时间戳分目录）
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据（从 D:\毕设资料\dataset 生成样本）
python src/prepare_datasets.py

# 3. 运行实验
python src/edane_full_pipeline.py --mode synthetic --quantize
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 6 --quantize
python src/edane_full_pipeline.py --mode file --dataset-preset mag_sample --snapshots 6 --quantize
python src/edane_full_pipeline.py --mode file --dataset-preset twitter_sample --snapshots 6 --quantize
python src/edane_full_pipeline.py --mode file --dataset-preset amazon3m_sample --snapshots 6 --quantize

# 或一键运行全部
run_all.bat
```

## 输出示例

每次运行在 `outputs/<数据集名>_<时间戳>/` 下生成（一眼看出哪个实验）：

| 文件 | 说明 |
|------|------|
| `summary.json` | 实验摘要（时延、F1、AUC、压缩比） |
| `metrics_per_snapshot.csv` | 逐快照指标 |
| `metrics_curves.svg` | 曲线图（可直接用于论文） |
| `final_embedding.npy` | 最终嵌入向量 |

## 文档导航

| 文档 | 内容 |
|------|------|
| [快速上手](docs/getting_started.md) | 环境搭建、首次运行 |
| [流水线用法](docs/pipeline_usage.md) | 完整命令、参数、数据格式 |
| [算法说明](docs/algorithm.md) | 理论背景、数学推导 |
| [开发指南](docs/development.md) | 项目架构、扩展方法 |

## 环境要求

- Python ≥ 3.9
- NumPy
- SciPy（稀疏矩阵）
