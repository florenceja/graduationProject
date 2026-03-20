# EDANE 快速上手

## 1. 环境准备

建议使用 **Python 3.9+**。安装依赖：

```bash
pip install -r requirements.txt
```

如需启用 PyTorch dense 后端，再额外安装 PyTorch：

```bash
pip install torch
```

或手动安装默认依赖（当前基础运行依赖 NumPy + SciPy）：

```bash
pip install numpy scipy
```

## 2. 准备数据

当前推荐把原始数据放在**项目根目录下的 `dataset/`**。`prepare_datasets.py` 当前默认读取这个目录。

准备好原始数据后，执行：

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

如需启用 PyTorch 的 dense 计算路径：

```bash
python src/edane_full_pipeline.py --mode synthetic --quantize --backend torch
```

### 3.2 Reddit 样本

建议第一次先用较小规模命令验证整条链路是否跑通，例如：

```bash
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 3 --max-nodes 3000 --quantize
```

确认可运行后，再尝试更大节点数或更多快照。

常规运行示例：

```bash
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 6 --quantize
```

使用逻辑回归评估（推荐）：

```bash
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 6 --classifier logreg --quantize
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 6 --classifier logreg --quantize --binary-quantize
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 6 --classifier logreg --quantize --update-rate 100
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 6 --classifier logreg --quantize --no-attr
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 6 --classifier logreg --quantize --no-hyperbolic
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 6 --classifier logreg --quantize --no-inc
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

说明：`outputs/` 属于本地实验产物目录，当前仓库默认将其视为本地生成内容，不建议直接纳入远程版本管理。

| 文件 | 说明 |
|------|------|
| `summary.json` | 整体实验摘要 |
| `metrics_per_snapshot.csv` | 每个快照的指标与更新时间（含 `link_ap`、`reconstruction_auc`） |
| `final_embedding.npy` | 最终浮点嵌入 |
| `final_embedding_int8.npy` | int8 量化副本 |
| `final_embedding_scale.npy` | int8 缩放系数 |
| `final_embedding_binary.npy` | Binary 副本（若开启 `--binary-quantize`） |
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
| `--update-rate` | 目标变化速率（次/秒，0=不控制） | 0 |
| `--snapshots` | 快照数 | 8 |
| `--snapshot-mode` | `window` 或 `cumulative` | window |
| `--max-nodes` | file 模式最大节点数（0=不限制） | 10000 |
| `--classifier` | `logreg`（推荐）或 `centroid` | logreg |
| `--backend` | `numpy`（默认）或 `torch`（仅 dense 后端） | numpy |
| `--no-attr` | 消融：禁用属性融合（w/o-Attr） | 关 |
| `--no-hyperbolic` | 消融：禁用双曲融合（w/o-Hyperbolic） | 关 |
| `--no-inc` | 消融：禁用增量更新（w/o-Inc） | 关 |

### 6.1 阶段2/3矩阵实验（一键）

```bash
python src/run_stage23_experiments.py --mode file --dataset-preset reddit_sample --snapshots 3 --max-nodes 3000
```

输出目录示例：`outputs/stage23_matrix_reddit_sample_<timestamp>/`

- `stage2_rate_results.csv`
- `stage3_ablation_results.csv`
- `stage23_combined_results.csv`

其中：

- `avg_update_latency_ms` = 端到端时延（含节流等待）
- `avg_compute_update_latency_ms` = 纯计算时延
- `avg_pacing_wait_ms` = 速率控制等待时延

更细的模块级参数（如 `init_*`、`fusion_*`、`binary_quantize`）请参考 `docs/modules_1_4_integrated.md`；当前主流水线默认只暴露常用实验参数。

当前评估输出已包含：`Macro-F1`、`Micro-F1`、`link_auc`、`link_ap`、`reconstruction_auc`。

## 7. 常见问题

**Q: 运行报错找不到 preset 目录？**
A: 先确认项目根目录下 `dataset/` 是否已有原始数据，再执行 `python src/prepare_datasets.py`，并检查 `data/<preset>` 是否存在。

**Q: 为什么我把原始数据放进仓库后，Git 状态很乱？**
A: 当前约定是 `dataset/` 和 `outputs/` 按本地目录管理，建议只在本地保留数据与实验输出，不直接作为远程仓库正文提交。

**Q: 指标看起来偏低？**
A: 可尝试增加 `--dim`、调整 `--learning-rate`、提升数据质量和标签覆盖率。

**Q: MAG/Twitter 能直接用原始全量吗？**
A: 理论上可处理，但全量数据过大。当前的 `mag_sample` 和 `twitter_sample` 是更适合毕业设计实验机的可运行版本。

**Q: 运行慢/占内存高？**
A: 减少节点数、减小 `--dim`、减少快照数，先做小规模验证。

**Q: `--backend torch` 为什么报缺少依赖？**
A: `requirements.txt` 默认只安装 NumPy + SciPy。若要启用 PyTorch dense 后端，请先单独安装 `torch`。
