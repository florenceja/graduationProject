# 按 RAM 档位的“全量 OAG 跑通”方案与执行建议

本文档给出一个明确结论：

- **流式加载 + 快照分桶写盘**（已在 `src/edane_full_pipeline.py` 实现）能显著降低“读 CSV/构图阶段”的内存压力；
- 但要实现 **`--max-nodes 0` 真全量训练/评估**，瓶颈仍主要来自：
  1) EDANE 的多份 `float64` 稠密矩阵常驻（attrs/embedding 等）；
  2) SciPy CSR 邻接（通常不止一份）；
  3) Python `dict[str,int]`、`set[(u,v)]` 的对象开销。

因此，“全量 OAG 跑通”建议拆成两件事（也是最可落地的工程路线）：

1) **全量转换**（zip → `data/OAG/*.csv`，证明数据来源全量）
2) **可控规模训练**（在全量数据生成后，用 `--max-nodes` 控制训练/评估规模，阶梯放量）

只有在 512GB~1TB RAM 级别，才建议尝试 `--max-nodes 0`。

---

## 0. 你需要先拿到的两个统计量（决定一切）

在数据机器上跑：

```bash
python src/prepare_datasets.py --convert-oag --subset-profile full --dry-run
```

你关心的是输出里的：

- `kept_papers`：记为 **N_full**
- `written_edges`：记为 **E_full**

> 说明：现在 `--dry-run` 已允许在 `data/OAG/*.csv` 存在时运行，不需要 `--overwrite`。

拿到 `N_full/E_full` 后，再决定“真全量训练是否可能”，以及每个 RAM 档位该选多大的 `--max-nodes`。

---

## 1. RAM 估算方法（为什么需要那么多内存）

令：

- `N`：训练节点数（≈`--max-nodes`；真全量则 N=N_full）
- `E`：训练无向边数（过滤到选中节点后的边）
- `F`：特征维度（OAG 默认 128）
- `D`：embedding 维度（默认 64）

### 1.1 EDANE 稠密矩阵（float64）常驻内存

当前实现里常驻的主要矩阵包含：

- `attrs`: `N×F`
- `random_projection`: `N×D`（注意：实现为 dense ndarray，即使值很稀疏也按满矩阵占内存）
- `struct_emb`: `N×D`
- `attr_emb`: `N×D`
- `embedding_`: `N×D`

仅这些 dense 常驻矩阵的近似内存为：

\[
\text{RAM}_{dense} \approx 8 \cdot N \cdot (F + 4D)
\]

默认 `F=128, D=64`：

- 每节点约 `8*(128+256)=3072 bytes ≈ 3KB/节点`
- `N=1,000,000` → ≈ 3GB
- `N=10,000,000` → ≈ 30GB

### 1.2 CSR 邻接矩阵（SciPy）

无向图常用双向存储，`nnz≈2E`。单份 CSR 近似：

\[
\text{RAM}_{csr,1} \approx (8+4)\cdot 2E + 4(N+1) \approx 24E + 4N
\]

实际运行时常有 `adj` 与 `norm_adj`（至少 2 份），流水线状态里可能出现第 3 份拷贝，建议按 **2~3 倍**预留。

### 1.3 Python 对象开销（真全量的“隐藏大头”）

当 `N/E` 很大时，以下对象会迅速吞噬内存：

- `node_to_idx: dict[str,int]`（字符串 key）
- 快照边集合：window 模式每快照读成 `set[(u,v)]`；cumulative 模式维护 `union_set`

这部分很难用公式精确估算，但经验上在“千万级节点/亿级边”后会显著主导 RAM 需求，也是为什么 256GB 仍不一定能真全量。

---

## 2. 各 RAM 档位的可行方案（推荐执行路线）

下面每个档位都给出：

- **目标定义**（推荐你把“跑通全量 OAG”落在什么层级）
- **建议磁盘空闲**（包含全量 CSV + 分桶 + 输出的安全余量）
- **建议命令**（可直接执行）

> 共通建议：
> - file 模式务必指定 `--partition-dir` 到 NVMe/大盘；
> - 全量长跑建议加 `--no-all-results`，避免反复改写仓库内 `all_results.csv`。
> - 下文命令中的 `PATH_TO_PARTITIONS` / `PATH_TO_PARTITIONS_FULL` 为占位：
>   - Windows 示例：`D:/edane_partitions`、`D:/edane_partitions_full`
>   - Linux 示例：`/mnt/nvme/edane_partitions`、`/mnt/nvme/edane_partitions_full`

### 2.1 RAM 32GB（开发机常见）

**可行目标**：

- 在大机器完成全量转换；本机只跑子图（`--max-nodes` 控制在 5万~20万 起步，逐步放量）

**建议磁盘空闲**：

- 若本机不存全量 CSV：≥ 50GB（outputs + partitions + 余量）
- 若本机要存全量 CSV：通常不建议（很可能磁盘先爆）

**推荐命令**：

```bash
python src/edane_full_pipeline.py --mode file --model edane \
  --snapshots 4 --snapshot-mode window \
  --max-nodes 100000 \
  --node-selection-mode bounded \
  --partition-dir "PATH_TO_PARTITIONS" \
  --time-quantile-sample 200000 \
  --classifier centroid \
  --no-all-results
```

### 2.2 RAM 64GB

**可行目标**：

- 可尝试在本机完成全量转换（视磁盘而定）；训练建议 `--max-nodes 20万~80万` 阶梯放量

**建议磁盘空闲**：

- 更稳：≥ 2TB（如果你要保留全量 CSV + 分桶 + 多次输出）
- 最低可尝试：≥ 1TB

**推荐命令**：

```bash
python src/edane_full_pipeline.py --mode file --model edane \
  --snapshots 6 --snapshot-mode window \
  --max-nodes 300000 \
  --node-selection-mode bounded \
  --partition-dir "PATH_TO_PARTITIONS" \
  --classifier centroid \
  --no-all-results
```

### 2.3 RAM 128GB

**可行目标**：

- 全量转换 + 更大子图训练（常见可跑 `--max-nodes 50万~200万`，取决于边密度）

**建议磁盘空闲**：

- ≥ 2TB（推荐）
- ≥ 1TB（可尝试）

**推荐命令**：

```bash
python src/edane_full_pipeline.py --mode file --model edane \
  --snapshots 6 --snapshot-mode window \
  --max-nodes 1000000 \
  --node-selection-mode bounded \
  --partition-dir "PATH_TO_PARTITIONS" \
  --time-quantile-sample 1000000 \
  --no-all-results
```

### 2.4 RAM 256GB

**可行目标**：

- 全量转换 + “接近全量”的大子图训练（常见可跑 `--max-nodes 100万~500万`，仍取决于 E 与快照数）
- 不建议默认追 `--max-nodes 0`，除非你已经拿到 `N_full/E_full` 且确认规模远小于典型 full OAG

**建议磁盘空闲**：

- ≥ 3TB（更稳）
- ≥ 2TB（推荐下限）

**推荐命令**：

```bash
python src/edane_full_pipeline.py --mode file --model edane \
  --snapshots 6 --snapshot-mode window \
  --max-nodes 3000000 \
  --node-selection-mode bounded \
  --partition-dir "PATH_TO_PARTITIONS" \
  --no-all-results
```

### 2.5 RAM 512GB ~ 1TB（才建议尝试真全量训练）

**可行目标**：

- 允许尝试 `--max-nodes 0`（真全量），但仍然有失败风险：
  - Python `dict/set` 的对象开销可能成为决定性瓶颈
  - 输出 embedding 可能巨大

**建议磁盘空闲**：

- ≥ 4TB（推荐），NVMe 优先

**推荐命令（真全量尝试）**：

```bash
python src/edane_full_pipeline.py --mode file --model edane \
  --snapshots 6 --snapshot-mode window \
  --max-nodes 0 \
  --node-selection-mode exact \
  --partition-dir "PATH_TO_PARTITIONS_FULL" \
  --keep-partitions \
  --classifier centroid \
  --no-all-results
```

---

## 3. 执行策略（强烈推荐的“阶梯放量”）

无论你是哪档 RAM，都建议按下面流程“稳稳跑通”：

1) 先跑 `--max-nodes 50000`（确保端到端通）
2) 再跑 `--max-nodes 200000`
3) 再跑 `--max-nodes 1000000`（如果 RAM ≥ 128GB）
4) 每一步记录：耗时、峰值内存、输出 embedding 大小、分桶目录占用

如果某一步 OOM：

- 第一优先级：降低 `--max-nodes`
- 第二优先级：降低 `--dim`、减少 `--snapshots`
- 第三优先级：将 `--partition-dir` 指向更快/更大的 NVMe

---

## 4. 与其他文档的关系

- 更详细的命令与故障排查：`docs/full_scale_runbook.md`
- OAG 全量与硬件建议背景：`docs/oag_full_scale.md`
- 参数表：`docs/pipeline_usage.md`
