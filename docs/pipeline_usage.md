# EDANE 全流程代码使用说明

## 1. 文件位置

| 脚本 | 路径 | 职责 |
|------|------|------|
| 实验主脚本 | `src/edane_full_pipeline.py` | 端到端实验流水线 |
| 算法核心 | `src/edane.py` | EDANE 模型定义 |
| 数据整理 | `src/prepare_datasets.py` | 原始数据 → CSV 格式 |
| 曲线绘图 | `src/plot_metrics_svg.py` | CSV → SVG 曲线图 |
| 快速验证 | `src/run_edane_experiment.py` | 合成图单文件实验 |

## 2. 已支持的数据集

| 预设名 | 数据目录 | 说明 |
|--------|---------|------|
| `reddit_sample` | `data/reddit_sample/` | Reddit 社交网络子图 |
| `amazon2m_sample` | `data/amazon2m_sample/` | Amazon-2M 商品网络子图 |
| `amazon3m_sample` | `data/amazon3m_sample/` | Amazon-3M 商品标签共现图 |
| `mag_sample` | `data/mag_sample/` | MAG 学术引用网络子图 |
| `twitter_sample` | `data/twitter_sample/` | Twitter 社交网络子图 |

每个目录包含：`edges.csv`、`features.csv`、`labels.csv`、`attr_updates.csv`（部分可选）

## 3. 支持的两种模式

### 3.1 合成数据模式（快速验证）

```bash
python src/edane_full_pipeline.py --mode synthetic --quantize
```

### 3.2 文件数据模式（真实实验）

#### 方式 A：使用预设数据目录（推荐）

```bash
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 6 --quantize
python src/edane_full_pipeline.py --mode file --dataset-preset amazon2m_sample --snapshots 6 --quantize
python src/edane_full_pipeline.py --mode file --dataset-preset mag_sample --snapshots 6 --quantize
python src/edane_full_pipeline.py --mode file --dataset-preset twitter_sample --snapshots 6 --quantize
python src/edane_full_pipeline.py --mode file --dataset-preset amazon3m_sample --snapshots 6 --quantize
```

使用逻辑回归分类评估（推荐）：

```bash
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 6 --classifier logreg --quantize
```

#### 方式 B：手动指定文件路径

```bash
python src/edane_full_pipeline.py ^
  --mode file ^
  --edges-path data/edges.csv ^
  --features-path data/features.csv ^
  --labels-path data/labels.csv ^
  --attr-updates-path data/attr_updates.csv ^
  --snapshots 10 ^
  --snapshot-mode window ^
  --quantize
```

## 4. 从原始数据重新生成样本

```bash
python src/prepare_datasets.py --prepare-reddit --prepare-amazon --prepare-amazon3m --prepare-mag --prepare-twitter
```

可调参数示例：

```bash
python src/prepare_datasets.py --prepare-reddit --reddit-max-nodes 10000
python src/prepare_datasets.py --prepare-amazon --amazon-max-nodes 20000 --amazon-max-edges 200000
python src/prepare_datasets.py --prepare-mag --mag-max-nodes 10000 --mag-max-edges 200000
python src/prepare_datasets.py --prepare-twitter --twitter-max-nodes 12000 --twitter-max-edges 180000
python src/prepare_datasets.py --prepare-amazon3m --amazon3m-max-nodes 10000 --amazon3m-max-edges 200000
```

## 5. 数据文件格式

### 5.1 边文件 `edges.csv`（必需）

| 列 | 说明 |
|----|------|
| `src` | 源节点 ID |
| `dst` | 目标节点 ID |
| `time` | 时间戳（可选，推荐提供以支持动态切分） |

```csv
src,dst,time
u1,u2,1710000000
u2,u5,1710003600
```

### 5.2 特征文件 `features.csv`（可选）

```csv
node_id,f1,f2,f3
u1,0.21,-0.05,1.77
u2,0.14,0.88,-0.33
```

### 5.3 标签文件 `labels.csv`（可选）

```csv
node_id,label
u1,0
u2,1
```

### 5.4 属性更新文件 `attr_updates.csv`（可选）

维度需要和 `features.csv` 一致。

```csv
time,node_id,f1,f2,f3
1710003600,u1,0.20,-0.02,1.80
```

## 6. 输出结果

每次运行会在 `outputs/<数据集名>_<时间戳>/` 下生成（如 `reddit_sample_20260316_170012/`）：

| 文件 | 说明 |
|------|------|
| `summary.json` | 整体实验摘要（时延、最终指标、压缩比） |
| `metrics_per_snapshot.csv` | 每个快照的指标与更新时间 |
| `final_embedding.npy` | 最终浮点嵌入 |
| `final_embedding_int8.npy` | 量化后的嵌入（开启 `--quantize` 时） |
| `final_embedding_scale.npy` | 量化缩放系数 |
| `node_mapping.csv` | 原始节点 ID → 索引映射 |
| `metrics_curves.svg` | 指标曲线图 |

## 7. 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dim` | 嵌入维度 | 64 |
| `--order` | 结构传播阶数 | 2 |
| `--projection-density` | 随机投影稀疏度 | 0.12 |
| `--learning-rate` | 增量更新率 | 0.55 |
| `--snapshots` | 时间切分快照数 | 8 |
| `--snapshot-mode` | `window` 或 `cumulative` | window |
| `--max-nodes` | file 模式最大节点数（0=不限制） | 10000 |
| `--dataset-preset` | 数据预设名（见上方表格） | — |
| `--classifier` | `logreg` 或 `centroid` | logreg |
| `--logreg-epochs` | 逻辑回归训练轮次 | 260 |
| `--logreg-lr` | 逻辑回归学习率 | 0.35 |
| `--logreg-weight-decay` | 逻辑回归权重衰减 | 1e-4 |
| `--quantize` | 开启 int8 量化 | 关 |
| `--seed` | 随机种子 | 42 |

## 8. 补充说明

- 当前实现已切换为 CSR 稀疏邻接后端（依赖 SciPy），可显著降低大图内存占用。安装：`pip install scipy`。
- 该代码是"从数据预处理到实验评估"的系统化研究原型，适合毕业设计实验与论文撰写。
- `MAG-` 原始数据体量极大，`mag_sample` 使用了节点/边抽样，并用结构统计构造轻量特征。
- `twitter` 原始数据缺少标准属性与标签，`twitter_sample` 采用结构统计特征和度分桶伪标签。
- `Amazon-3M.raw` 是极端多标签分类数据集（约 280 万商品），`amazon3m_sample` 通过共享标签（co-label）关系构建商品图，特征采用文本统计量。

若你已有历史结果目录，只想补画曲线图：

```bash
python src/plot_metrics_svg.py --metrics-csv outputs/xxx/metrics_per_snapshot.csv
```
