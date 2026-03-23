# EDANE 开发说明（OAG 固定数据集）

## 1. 当前数据约束

当前仓库的真实数据工作流已经收敛为：

- 原始/统一输入目录：`dataset/OAG/`
- synthetic 模式：仅用于快速调试
- 其他历史数据样本目录：不再作为当前开发目标

## 2. 关键脚本

| 文件 | 作用 |
|------|------|
| `src/edane_full_pipeline.py` | 主流水线 |
| `src/run_stage23_experiments.py` | 阶段2/3矩阵实验 |
| `src/prepare_datasets.py` | 检查 `dataset/OAG/` 是否完整 |
| `src/edane.py` | EDANE |
| `src/dane.py` | DANE baseline |
| `src/dtformer.py` | DTFormer-style baseline |

## 3. file 模式约定

- file 模式固定解析 `dataset/OAG/edges.csv`
- file 模式固定解析 `dataset/OAG/features.csv`
- file 模式固定解析 `dataset/OAG/labels.csv`
- 若存在 `dataset/OAG/attr_updates.csv`，则自动使用
- 不再通过 `--dataset-preset` 选择数据集

## 4. 新增模型时的要求

新模型应继续满足统一接口：

- `fit(adj, attrs)`
- `apply_updates(...)`
- `get_embedding()`

这样才能直接复用现有 OAG 评估流水线。

## 5. 调试建议

先检查 OAG 目录：

```bash
python src/prepare_datasets.py
```

再跑 synthetic：

```bash
python src/edane_full_pipeline.py --mode synthetic
```

最后跑 OAG：

```bash
python src/edane_full_pipeline.py --mode file --snapshots 3 --max-nodes 3000
```
