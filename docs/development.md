# EDANE 开发说明（OAG 固定数据集）

## 1. 当前数据约束

当前仓库的真实数据工作流已经收敛为：

- 原始 OAG 压缩包目录：`dataset/OAG/`
- 统一 CSV 输入目录：`data/OAG/`
- synthetic 模式：仅用于快速调试
- 其他历史数据样本目录：不再作为当前开发目标

## 2. 关键脚本

| 文件 | 作用 |
|------|------|
| `src/edane_full_pipeline.py` | 主流水线 |
| `src/run_stage23_experiments.py` | 阶段2/3矩阵实验 |
| `src/prepare_datasets.py` | 将 `dataset/OAG/` 原始 zip 转换并校验 `data/OAG/` |
| `src/edane.py` | EDANE |
| `src/dane.py` | DANE baseline |
| `src/dtformer.py` | DTFormer-style baseline |

说明：

- `dane.py` 与 `dtformer.py` 均以 `paper_approximation` 口径接入统一流水线（见 `summary.json/implementation_fidelity`）。
- `prepare_datasets.py` 内仍保留部分历史样本构建函数（Amazon/MAG/Twitter 等），但当前工作流不再依赖它们；若后续彻底不需要，可再清理。

## 3. file 模式约定

- file 模式固定解析 `data/OAG/edges.csv`
- file 模式固定解析 `data/OAG/features.csv`
- file 模式固定解析 `data/OAG/labels.csv`
- 若存在 `data/OAG/attr_updates.csv`，则自动使用
- 不再通过 `--dataset-preset` 选择数据集

## 4. 新增模型时的要求

新模型应继续满足统一接口：

- `fit(adj, attrs)`
- `apply_updates(...)`
- `get_embedding()`

这样才能直接复用现有 OAG 评估流水线。

## 5. 调试建议

先检查/转换 OAG CSV 目录：

```bash
python src/prepare_datasets.py --validate-only
```

再跑 synthetic：

```bash
python src/edane_full_pipeline.py --mode synthetic
```

最后跑 OAG：

```bash
python src/edane_full_pipeline.py --mode file --model edane --snapshots 3 --max-nodes 3000
```
