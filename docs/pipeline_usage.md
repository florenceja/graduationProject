# EDANE 全流程代码使用说明（OAG 固定数据集）

## 1. 当前约束

- `--mode synthetic`：仍用于快速自检。
- `--mode file`：**固定且仅使用** `data/OAG/`。
- 其他历史样本数据集（Reddit、Amazon、MAG、Twitter）**不再作为当前实验范围**。

固定数据目录要求：

```text
data/
└── OAG/
    ├── edges.csv
    ├── features.csv
    ├── labels.csv
    └── attr_updates.csv   # 可选
```

数据来源页：

- https://open.aminer.cn/open/article?id=67aaf63af4cbd12984b6a5f0

转换口径：

- `references` → `edges.csv`
- `year` → `time`
- `title + abstract + keywords` → `features.csv`
- `venue` → `labels.csv`

注意：这是为当前实验管线构造的 **OAG-derived 代理任务**。

## 2. 主脚本

| 脚本 | 路径 | 职责 |
|------|------|------|
| 实验主脚本 | `src/edane_full_pipeline.py` | 端到端实验流水线 |
| 阶段2/3矩阵脚本 | `src/run_stage23_experiments.py` | 批量运行更新频率与消融实验 |
| 数据检查脚本 | `src/prepare_datasets.py` | 将 `dataset/OAG/` 原始 zip 转为 `data/OAG/` CSV，并检查输入是否完整 |

## 3. 运行方式

### 3.1 合成数据模式

```bash
python src/edane_full_pipeline.py --mode synthetic --quantize
```

### 3.2 OAG 文件模式

如果 `dataset/OAG/` 下有原始 `v5_oag_publication_*.zip`，先转换到 `data/OAG/`：

```bash
python src/prepare_datasets.py --convert-oag --subset-profile test --overwrite
```

常用 profile：

- `test`：本机测试算法，优先可跑性
- `small`：更大一些的对比实验子集
- `medium`：更接近正式实验的数据量
- `full`：尝试全量 OAG 转换

如果只检查固定 CSV 目录：

```bash
python src/prepare_datasets.py --validate-only
```

如遇到极少数超大 JSON 记录导致内存峰值，可调小：

```bash
python src/prepare_datasets.py --convert-oag --overwrite --max-record-bytes 2000000
```

如果换到更强硬件上尝试全量 OAG：

```bash
python src/prepare_datasets.py --convert-oag --subset-profile full --overwrite
```

建议环境：

- 128GB 及以上内存
- NVMe SSD
- 长时间稳定运行环境（服务器/工作站）

注意：全量转换 ≠ 当前主流水线就能直接全量训练。主流水线仍会整读 CSV，通常还需要结合 `--max-nodes` 或后续采样。

然后运行主流水线：

```bash
python src/edane_full_pipeline.py --mode file --snapshots 6 --quantize
python src/edane_full_pipeline.py --mode file --snapshots 6 --classifier logreg --quantize
python src/edane_full_pipeline.py --mode file --snapshots 6 --classifier logreg --eval-protocol repeated_stratified --eval-repeats 10 --quantize
```

消融实验：

```bash
python src/edane_full_pipeline.py --mode file --snapshots 6 --quantize --no-attr
python src/edane_full_pipeline.py --mode file --snapshots 6 --quantize --no-hyperbolic
python src/edane_full_pipeline.py --mode file --snapshots 6 --quantize --no-inc
```

## 4. 阶段2/3矩阵实验

```bash
python src/run_stage23_experiments.py --mode file --snapshots 3 --max-nodes 3000
python src/run_stage23_experiments.py --mode synthetic --snapshots 2 --stage2-rates 10,100 --synthetic-rounds 6
```

输出目录示例：

- `outputs/stage23_matrix_oag_<timestamp>/`

## 5. 数据文件格式

### 5.1 `edges.csv`

```csv
src,dst,time
u1,u2,1710000000
u2,u5,1710003600
```

### 5.2 `features.csv`

```csv
node_id,f1,f2,f3
u1,0.21,-0.05,1.77
u2,0.14,0.88,-0.33
```

### 5.3 `labels.csv`

```csv
node_id,label
u1,0
u2,1
```

### 5.4 `attr_updates.csv`（可选）

```csv
time,node_id,f1,f2,f3
1710007200,u1,0.23,-0.01,1.80
```

## 6. 当前说明

- file 模式不再接受 `--dataset-preset`。
- file 模式固定读取 `data/OAG/`。
- `dataset/OAG/` 仅作为原始 OAG zip 存放目录。
