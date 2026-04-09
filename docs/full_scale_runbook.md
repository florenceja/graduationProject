# 全量数据集跑通指南（OAG / file 模式）

本文档的目标很单一：**让算法在全量数据集上“能跑通、跑得稳、可复现”**。

> 重要区分：
> - **全量转换**：把 `dataset/OAG/*.zip` 转成 `data/OAG/*.csv`。
> - **全量训练/评估**：在不限制节点的情况下（`--max-nodes 0`）直接训练/评估。
>
> 本项目已对 file 模式做了“流式加载 + 快照分桶写盘”，因此“读 CSV 爆内存”不再是主要阻塞点；
> 真正限制全量训练的是 **CSR 邻接矩阵 + 多份 embedding/attrs 的常驻内存**。

---

## 1. 运行前硬件/磁盘检查（建议）

### 1.1 经验下限

- 内存：**128GB+**（更理想：**256GB+**）
- 磁盘：NVMe SSD（建议预留 **数百 GB**，用于：CSV 输出、SQLite 临时库、快照分桶临时文件）
- CPU：16 核以上更理想（转换过程以 IO + JSON 解码为主）

### 1.2 粗略内存估算（用于决定 max-nodes）

设选择的子图规模为 `N` 个节点，`E` 条无向边（写入 CSR 时会变成约 `2E` 条有向存储）。

- **CSR 邻接**（SciPy CSR）：
  - `data`：float64，约 `8 * 2E` bytes
  - `indices`：int32（通常），约 `4 * 2E` bytes
  - `indptr`：int32，约 `4 * (N+1)` bytes
  - 合计约 `~ 24E + 4N` bytes（忽略 Python/对象开销）

- **属性矩阵 attrs**：float64，约 `8 * N * F` bytes
- **embedding/struct/attr**：至少 3 份 float64（`embedding_ / struct_emb / attr_emb`），约 `24 * N * D` bytes

因此，想“全量训练”本质上是对 `N、E、F、D` 的硬约束；不满足就必须用 `--max-nodes` 控制规模。

---

## 2. 全量转换（生成 data/OAG/*.csv）

在目标机器上运行（会写入 `data/OAG/`）：

```bash
python src/prepare_datasets.py --convert-oag --subset-profile full --overwrite
```

说明：
- `prepare_datasets.py` 本身是流式扫描 zip（逐行 JSON），更偏 IO；
- 需要较大磁盘空间，尤其是 SQLite 临时库与输出 CSV。

---

## 3. file 模式跑通策略（推荐路线）

### 3.1 推荐：全量转换 + 子图训练（可控、最容易稳定跑通）

核心思路：**让“全量”发生在数据转换阶段；训练/评估阶段用可控规模子图跑通算法。**

```bash
python src/edane_full_pipeline.py --mode file --model edane \
  --snapshots 6 --snapshot-mode window \
  --max-nodes 200000 \
  --node-selection-mode bounded \
  --partition-dir "PATH_TO_PARTITIONS" \
  --time-quantile-sample 1000000 \
  --no-all-results
```

其中 `PATH_TO_PARTITIONS` 请替换为你机器上的快路径（建议 NVMe）：

- Windows 示例：`D:/edane_partitions`
- Linux 示例：`/mnt/nvme/edane_partitions`

参数解释：
- `--max-nodes`：训练规模的“硬闸”，决定能否跑通
- `--node-selection-mode bounded`：用近似 Top-K 度数统计先筛节点，避免扫描阶段维护超大字典
- `--partition-dir`：快照分桶临时文件目录（强烈建议放 NVMe）
- `--time-quantile-sample`：时间分位点切分快照的采样容量；越大越准，但越慢

### 3.2 尝试：真正全量训练（高风险，只有大内存机器才建议）

```bash
python src/edane_full_pipeline.py --mode file --model edane \
  --snapshots 6 --snapshot-mode window \
  --max-nodes 0 \
  --node-selection-mode exact \
  --partition-dir "PATH_TO_PARTITIONS_FULL" \
  --keep-partitions \
  --no-all-results
```

其中 `PATH_TO_PARTITIONS_FULL` 同理：

- Windows 示例：`D:/edane_partitions_full`
- Linux 示例：`/mnt/nvme/edane_partitions_full`

注意：
- `--max-nodes 0` 会把“是否能跑通”完全交给硬件；
- 如遇到 OOM，优先降低 `--max-nodes`，其次降低 `--dim`、减少快照数。

---

## 4. 运行输出与可观测性

每次运行会生成：

- `outputs/<tag>_<timestamp>/summary.json`
- `outputs/<tag>_<timestamp>/metrics_per_snapshot.csv`
- `outputs/<tag>_<timestamp>/metrics_curves.svg`
- `outputs/<tag>_<timestamp>/final_embedding.npy`

file 模式的加载阶段会打印：
- pass1：扫描边、节点筛选、time 分位点
- pass2：按快照分桶写盘（partition）

---

## 5. 常见故障排查

1) **分桶目录磁盘爆满 / 临时盘空间不足**
- 现象：运行中断、Windows temp 盘满
- 处理：显式指定 `--partition-dir` 到大盘/NVMe；必要时关闭 `--keep-partitions`

2) **评估阶段 repeated_stratified 经常降级为 single_random_fallback**
- 原因：`venue` 标签长尾极端（大量类别样本数 < 2）
- 建议：仅影响 F1 评估口径，可用
  - `--label-cleanup-mode eval_only --min-class-support 5`
  - 或 `--classifier logreg --logreg-class-weight balanced`

3) **运行很慢**
- 优先检查：`--stable-batches` 是否开启（全量大图不建议）
- 适当降低：`--time-quantile-sample`
- 降低训练规模：`--max-nodes`

---

## 6. 推荐的“阶梯放量”跑通流程

1) `--max-nodes 50000` 跑通（验证端到端）
2) `--max-nodes 200000` 压测（观察内存与耗时）
3) 在目标机上逐步放大到期望规模；最后再考虑 `--max-nodes 0`（如果目标是全量训练）
