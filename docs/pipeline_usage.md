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

构造更接近正式实验的子图（示例：节点>=15000，优先保边）：

```bash
python src/prepare_datasets.py --convert-oag --overwrite \
  --selection-strategy dense --max-papers 15000 --candidate-multiplier 3 \
  --min-venue-support 5 --keep-unlabeled --max-record-bytes 2000000
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
python src/edane_full_pipeline.py --mode file --model edane --snapshots 6 --quantize
python src/edane_full_pipeline.py --mode file --model edane --snapshots 6 --classifier logreg --quantize
python src/edane_full_pipeline.py --mode file --model edane --snapshots 6 --classifier logreg --eval-protocol repeated_stratified --eval-repeats 10 --quantize

# baseline 对比（同一数据、同一协议）
python src/edane_full_pipeline.py --mode file --model dane --snapshots 6 --max-nodes 15000
python src/edane_full_pipeline.py --mode file --model dtformer --snapshots 6 --max-nodes 15000
```

模型选择说明：

- `--model edane`：支持 `--quantize/--binary-quantize`、`--no-attr/--no-hyperbolic/--no-inc`
- `--model dane`：不支持 EDANE 专属消融参数；量化开关不会生效（summary 中会被强制记为 false）
- `--model dtformer`：不支持 EDANE 专属消融参数；量化开关不会生效（summary 中会被强制记为 false）

输出口径说明：

- `summary.json` 中包含 `implementation_fidelity` 字段：`edane` 为 `native`，基线为 `paper_approximation`

消融实验：

```bash
python src/edane_full_pipeline.py --mode file --snapshots 6 --quantize --no-attr
python src/edane_full_pipeline.py --mode file --snapshots 6 --quantize --no-hyperbolic
python src/edane_full_pipeline.py --mode file --snapshots 6 --quantize --no-inc
```

### 3.3 参数一览（`python src/edane_full_pipeline.py ...`）

> 参数以 `src/edane_full_pipeline.py::build_parser()` 为准。
> 说明中的“适用范围”用于提示哪些模型/模式会真正使用该参数（其余情况下可能被忽略或被强制无效）。

| 参数 | 默认值 | 可选值/类型 | 含义 | 适用范围 |
|---|---:|---|---|---|
| `--mode` | `synthetic` | `synthetic` / `file` | 数据来源：合成图或磁盘 CSV（file 模式固定 `data/OAG/`） | 全部 |
| `--output-dir` | `""` | 字符串路径 | 输出目录；为空则写入 `outputs/<tag>_<timestamp>/` | 全部 |
| `--model` | `edane` | `edane` / `dane` / `dtformer` | 选择算法/基线 | 全部 |
| `--snapshots` | `8` | int | 动态快照数（file 模式会按时间/顺序切分边） | 全部 |
| `--snapshot-mode` | `window` | `window` / `cumulative` | 快照边集口径：窗口式或累计式 | 全部 |
| `--max-nodes` | `10000` | int | file 模式最大节点数（0=不限制，可能内存溢出） | file |
| `--dim` | `64` | int | 嵌入维度 `d` | 全部（不同模型均使用） |
| `--order` | `2` | int | EDANE 结构传播阶数 `q` | EDANE |
| `--projection-density` | `0.12` | float | EDANE 稀疏随机投影非零密度 | EDANE |
| `--learning-rate` | `0.55` | float | EDANE 增量更新步长（`apply_updates` 内使用） | EDANE |
| `--dane-attr-topk` | `20` | int | DANE 属性相似图 top-k（稀疏近似） | DANE |
| `--dane-similarity-block-size` | `512` | int | DANE 构建属性相似图的分块大小 | DANE |
| `--dane-perturbation-rank` | `64` | int | DANE 扰动式在线更新的近似秩（越大越慢/更占用） | DANE |
| `--dtformer-patch-size` | `2` | int | DTFormer patch 大小（历史 token 聚合粒度） | DTFormer |
| `--dtformer-history-snapshots` | `8` | int | DTFormer 保留的历史快照数（越大越吃内存） | DTFormer |
| `--dtformer-hidden-dim` | `96` | int | DTFormer 隐层维度（越大越吃内存/更慢） | DTFormer |
| `--dtformer-attention-temperature` | `1.0` | float | DTFormer 注意力温度系数（数值稳定/平滑程度） | DTFormer |
| `--update-rate` | `0` | int | 目标变化速率（次/秒）；0=不做节流 | 全部 |
| `--quantize` | `False` | flag | EDANE int8 量化（压缩副本） | EDANE（DANE/DTFormer 会被强制无效） |
| `--binary-quantize` | `False` | flag | EDANE 二值化副本 | EDANE（DANE/DTFormer 会被强制无效） |
| `--no-attr` | `False` | flag | EDANE 消融：禁用属性融合（仅结构表征） | EDANE（对其他模型会报错） |
| `--no-hyperbolic` | `False` | flag | EDANE 消融：禁用双曲融合，改欧氏门控融合 | EDANE（对其他模型会报错） |
| `--no-inc` | `False` | flag | 消融：禁用增量更新，每个快照全量重训（w/o-Inc） | 全部 |
| `--seed` | `42` | int | 随机种子（切分、初始化等） | 全部 |
| `--classifier` | `logreg` | `centroid` / `logreg` | 节点分类评估分类器：最近类中心或 softmax 逻辑回归 | 仅 F1 评估 |
| `--eval-protocol` | `repeated_stratified` | `single_random` / `repeated_stratified` | F1 评估协议：单次随机或重复分层切分 | 仅 F1 评估 |
| `--eval-repeats` | `10` | int | 重复分层评估次数（single_random 时忽略） | 仅 F1 评估 |
| `--eval-train-ratio` | `0.7` | float | F1 评估训练集比例 | 仅 F1 评估 |
| `--label-cleanup-mode` | `off` | `off` / `eval_only` | 评估前标签清洗：`eval_only` 会过滤低支持类别并重映射标签 | 仅 F1 评估 |
| `--min-class-support` | `5` | int | `eval_only` 时保留类别的最小样本数（建议>=2） | 仅 F1 评估 |
| `--backend` | `numpy` | `numpy` / `torch` | dense 数值后端；torch 仅对 EDANE 的部分 dense 运算生效 | EDANE（其他模型通常忽略） |
| `--logreg-epochs` | `260` | int | `classifier=logreg` 时的训练轮数 | 仅 F1 评估 |
| `--logreg-lr` | `0.35` | float | `classifier=logreg` 时的学习率 | 仅 F1 评估 |
| `--logreg-weight-decay` | `1e-4` | float | `classifier=logreg` 时的 L2 权重衰减 | 仅 F1 评估 |
| `--logreg-class-weight` | `none` | `none` / `balanced` | `classifier=logreg` 的类别权重策略（长尾可尝试 balanced） | 仅 F1 评估 |
| `--synthetic-nodes` | `600` | int | 合成图节点数 | synthetic |
| `--synthetic-classes` | `6` | int | 合成图类别数 | synthetic |
| `--synthetic-feat-dim` | `24` | int | 合成图特征维度 | synthetic |
| `--synthetic-rounds` | `50` | int | 合成动态图变化轮数（快照更新次数） | synthetic |

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

注意：当前 `build_graph_from_files()` 会把边对规范化为无向图（引用边方向会被丢弃），并在有 `time` 时按分位点切分快照。

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

## 7. 常见问题：为什么 Macro/Micro-F1 很低？如何改进？

在 OAG-derived 口径下，节点分类标签来自 `venue`（会场/期刊）。当你看到 EDANE/DANE 的 `macro_f1`、`micro_f1` 都很低时，通常不是模型“完全不可用”，而是数据本身呈现 **类别数非常多 + 长尾极端（每类样本很少）**，导致多分类线性 probe 很难学、且 Macro-F1 会被大量小类拉低。

另外要注意：

- `--min-class-support` 只有在 `--label-cleanup-mode eval_only` 时才会生效；若为 `off`，评估会保留全部长尾类别，F1 很容易被拖到很低。
- 当存在大量“每类 1 个样本”的类别时，`repeated_stratified` 可能失败并在 `summary.json` 中记录降级（`f1_eval_protocol_used=single_random_fallback`）。

### 7.1 推荐做法（尽量只提升 F1，不影响 link/recon）

下面这些改动只影响 **F1 评估阶段**（以及 summary 中的 `labeled_nodes/eval_class_count` 统计），一般不改变 `link_auc/link_ap/reconstruction_auc`：

1) 评估阶段过滤低支持类别（推荐默认）：

```bash
python src/edane_full_pipeline.py --mode file --model edane --snapshots 6 \
  --classifier logreg --eval-protocol repeated_stratified --eval-repeats 10 \
  --label-cleanup-mode eval_only --min-class-support 5
```

2) 分类头更适配长尾（评估专用）：

```bash
python src/edane_full_pipeline.py --mode file --model dane --snapshots 6 \
  --classifier logreg --eval-protocol repeated_stratified --eval-repeats 10 \
  --logreg-class-weight balanced \
  --logreg-lr 0.1 --logreg-epochs 800 --logreg-weight-decay 1e-3
```

更详细的设置解释见：`docs/evaluation_setting_comparison.md`。
