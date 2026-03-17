# EDANE 快速上手

## 1. 环境准备

建议使用 **Python 3.9+**。安装依赖：

```bash
pip install -r requirements.txt
```

或手动安装（当前代码依赖 NumPy + SciPy）：

```bash
pip install numpy scipy
```

## 2. 准备数据

如果你有 `D:\毕设资料\dataset` 原始数据，可直接执行：

```bash
python src/prepare_datasets.py --prepare-reddit --prepare-amazon --prepare-amazon3m --prepare-mag --prepare-twitter
```

执行后会在项目下生成：

- `data/reddit_sample/`
- `data/amazon2m_sample/`
- `data/amazon3m_sample/`
- `data/mag_sample/`
- `data/twitter_sample/`

## 3. 快速跑通实验

### 3.1 合成数据（最快，无需准备数据）

```bash
python src/edane_full_pipeline.py --mode synthetic --quantize
```

### 3.2 Reddit 样本

```bash
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 6 --quantize
```

使用逻辑回归评估（推荐）：

```bash
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 6 --classifier logreg --quantize
```

### 3.3 Amazon2M 样本

```bash
python src/edane_full_pipeline.py --mode file --dataset-preset amazon2m_sample --snapshots 4 --quantize
```

如果机器内存有限（推荐），可加节点上限：

```bash
python src/edane_full_pipeline.py --mode file --dataset-preset amazon2m_sample --snapshots 4 --max-nodes 8000 --quantize
```

### 3.4 MAG 样本

```bash
python src/edane_full_pipeline.py --mode file --dataset-preset mag_sample --snapshots 6 --quantize
```

### 3.5 Twitter 样本

```bash
python src/edane_full_pipeline.py --mode file --dataset-preset twitter_sample --snapshots 6 --quantize
```

### 3.6 Amazon-3M 样本

```bash
python src/edane_full_pipeline.py --mode file --dataset-preset amazon3m_sample --snapshots 6 --quantize
```

### 3.7 一键运行全部（Windows）

```bash
run_all.bat
```

## 4. 查看输出结果

每次运行会在 `outputs/<数据集名>_<时间戳>/` 生成结果（如 `synthetic_20260316_165834/`），重点看：

| 文件 | 说明 |
|------|------|
| `summary.json` | 整体实验摘要 |
| `metrics_per_snapshot.csv` | 每个快照的指标与更新时间 |
| `final_embedding.npy` | 最终浮点嵌入 |
| `metrics_curves.svg` | 自动生成的指标曲线图（可直接放论文） |

## 5. 用你自己的数据运行

准备以下文件：

- `edges.csv`（必需）：`src,dst,time`
- `features.csv`（可选）：`node_id,f1,f2,...`
- `labels.csv`（可选）：`node_id,label`
- `attr_updates.csv`（可选）：`time,node_id,f1,f2,...`

运行命令：

```bash
python src/edane_full_pipeline.py ^
  --mode file ^
  --edges-path your_data/edges.csv ^
  --features-path your_data/features.csv ^
  --labels-path your_data/labels.csv ^
  --attr-updates-path your_data/attr_updates.csv ^
  --snapshots 8 ^
  --snapshot-mode window ^
  --quantize
```

## 6. 常用参数速查

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dim` | 嵌入维度 | 64 |
| `--order` | 结构传播阶数 | 2 |
| `--projection-density` | 随机投影密度 | 0.12 |
| `--learning-rate` | 增量更新率 | 0.55 |
| `--snapshots` | 快照数 | 8 |
| `--snapshot-mode` | `window` 或 `cumulative` | window |
| `--max-nodes` | file 模式最大节点数（0=不限制） | 10000 |
| `--classifier` | `logreg`（推荐）或 `centroid` | logreg |

## 7. 常见问题

**Q: 运行报错找不到 preset 目录？**
A: 先执行 `python src/prepare_datasets.py`，或检查 `data/<preset>` 是否存在。

**Q: 指标看起来偏低？**
A: 可尝试增加 `--dim`、调整 `--learning-rate`、提升数据质量和标签覆盖率。

**Q: MAG/Twitter 能直接用原始全量吗？**
A: 理论上可处理，但全量数据过大。当前的 `mag_sample` 和 `twitter_sample` 是更适合毕业设计实验机的可运行版本。

**Q: 运行慢/占内存高？**
A: 减少节点数、减小 `--dim`、减少快照数，先做小规模验证。
