# EDANE 快速上手（OAG 固定数据集）

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

如需启用 PyTorch dense 后端：

```bash
pip install torch
```

## 2. 准备 OAG 数据目录

项目当前固定使用：

```text
dataset/OAG/
```

该目录至少需要：

- `edges.csv`
- `features.csv`
- `labels.csv`

可选：

- `attr_updates.csv`

来源页：

- https://open.aminer.cn/open/article?id=67aaf63af4cbd12984b6a5f0

如果当前目录里还是 `v5_oag_publication_*.zip` 原始包，先执行转换：

```bash
python src/prepare_datasets.py --convert-oag --overwrite
```

如果只是检查 CSV 是否已就绪：

```bash
python src/prepare_datasets.py --validate-only
```

## 3. 跑通实验

### 3.1 synthetic 自检

```bash
python src/edane_full_pipeline.py --mode synthetic --quantize
```

### 3.2 OAG 文件模式

```bash
python src/edane_full_pipeline.py --mode file --snapshots 3 --max-nodes 3000 --quantize
python src/edane_full_pipeline.py --mode file --snapshots 6 --classifier logreg --quantize
python src/edane_full_pipeline.py --mode file --snapshots 6 --classifier logreg --quantize --no-inc
```

### 3.3 阶段2/3矩阵实验

```bash
python src/run_stage23_experiments.py --mode file --snapshots 3 --max-nodes 3000
```

输出目录示例：

- `outputs/stage23_matrix_oag_<timestamp>/`

## 4. 一键运行

```bash
run_all.bat
```

## 5. 结果输出

单次运行默认输出到：

- `outputs/oag_<timestamp>/`
- `outputs/synthetic_<timestamp>/`

重点文件：

- `summary.json`
- `metrics_per_snapshot.csv`
- `metrics_curves.svg`
- `final_embedding.npy`

## 6. 当前范围

- 当前项目的真实数据实验只考虑 `dataset/OAG/`
- 其他历史样本数据集不再作为当前工作流的一部分
