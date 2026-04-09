# HPC 全量 OAG 运行命令手册（Python 3.9.2 版）

> 目标：在高性能平台上完成
> 1) 全量转换（zip → `data/OAG/*.csv`）
> 2) 阶梯放量训练
> 3) 最终尝试真全量训练（`--max-nodes 0`）

适配对象：当前 `algorithms` 项目（Linux，已切换到 Python 3.9.2）

---

## 0. 进入项目并确认解释器

```bash
cd /public/$USER/algorithms   # 改成你的实际路径
python --version               # 需要看到 Python 3.9.2

module avail python
module load python/3.9.2
```

如果此处仍是 Python 2.7，请先加载平台 Python3 模块或使用 python3.9 命令后再继续。

---

## 1. 创建并激活虚拟环境（固定依赖）

```bash
python -m venv .venv
source .venv/bin/activate
python --version
pip install -U pip setuptools wheel
pip install -r requirements.txt



source .venv/bin/activate
python --version


nohup python -u src/prepare_datasets.py --convert-oag --subset-profile full --resume --checkpoint-every 200000 2>&1 | tee -a convert_full_1zip.log

python src/edane_full_pipeline.py \
  --mode file --model edane \
  --snapshots 4 --snapshot-mode window \
  --max-nodes 0 \
  --node-selection-mode exact \
  --partition-dir /public/home/hpc234712187/dyn_net_exp/algorithms/tmp_partitions \
  --output-dir /public/home/hpc234712187/dyn_net_exp/algorithms/hpc_outputs/run_fullnodes_single_zip \
  --classifier centroid \
  --no-all-results
```

---

## 2. 准备运行目录（放大盘）

```bash
mkdir -p /store/$USER/edane_partitions
mkdir -p /store/$USER/edane_outputs
```

> 不建议把分桶目录和大量输出放在 `/home`。

---

## 3. 检查原始 OAG 压缩包

```bash
ls -lh dataset/OAG/*.zip
```

说明：你提到总共 16 个 zip。若当前只有 1 个，也可测试流程，但不是全量口径。

---

## 4. 先做 dry-run 统计（推荐）

```bash
python src/prepare_datasets.py --convert-oag --subset-profile full --dry-run
```

---

## 5. 执行全量转换（关键步骤）

```bash
python src/prepare_datasets.py --convert-oag --subset-profile full --overwrite
```

转换后校验：

```bash
python src/prepare_datasets.py --validate-only
ls -lh data/OAG
```

---

## 6. 冒烟测试（先确认 file 模式完整跑通）

```bash
python src/edane_full_pipeline.py \
  --mode file --model edane \
  --snapshots 3 --snapshot-mode window \
  --max-nodes 50000 \
  --node-selection-mode bounded \
  --partition-dir /store/$USER/edane_partitions \
  --output-dir /store/$USER/edane_outputs/smoke_50k \
  --classifier centroid \
  --no-all-results
  
  
  python -u src/prepare_datasets.py \
  --convert-oag --subset-profile full --overwrite \
  --resume --checkpoint-every 200000 \
  2>&1 | tee convert_full_1zip.log
  
  
  python -u src/prepare_datasets.py \
  --convert-oag --subset-profile full \
  --resume --checkpoint-every 200000 \
  2>&1 | tee convert_full_1zip_resume.log
```

---

## 7. 阶梯放量（建议顺序执行）

### 7.1 50 万节点

```bash
python src/edane_full_pipeline.py \
  --mode file --model edane \
  --snapshots 4 --snapshot-mode window \
  --max-nodes 500000 \
  --node-selection-mode bounded \
  --partition-dir /store/$USER/edane_partitions \
  --output-dir /store/$USER/edane_outputs/run_500k \
  --no-all-results
```

### 7.2 100 万节点

```bash
python src/edane_full_pipeline.py \
  --mode file --model edane \
  --snapshots 4 --snapshot-mode window \
  --max-nodes 1000000 \
  --node-selection-mode bounded \
  --partition-dir /store/$USER/edane_partitions \
  --output-dir /store/$USER/edane_outputs/run_1m \
  --no-all-results
```

### 7.3 200 万节点（视内存情况）

```bash
python src/edane_full_pipeline.py \
  --mode file --model edane \
  --snapshots 4 --snapshot-mode window \
  --max-nodes 2000000 \
  --node-selection-mode bounded \
  --partition-dir /store/$USER/edane_partitions \
  --output-dir /store/$USER/edane_outputs/run_2m \
  --no-all-results
```

---

## 8. 最终尝试真全量训练（`--max-nodes 0`）

> 建议在 50万/100万/200万都稳定后再执行。

```bash
python src/edane_full_pipeline.py \
  --mode file --model edane \
  --snapshots 4 --snapshot-mode window \
  --max-nodes 0 \
  --node-selection-mode exact \
  --partition-dir /store/$USER/edane_partitions \
  --output-dir /store/$USER/edane_outputs/run_full_all_nodes \
  --classifier centroid \
  --no-all-results
```

---

## 9. 失败兜底（OOM / 进程被杀）

按优先级降载：

1. `--max-nodes 0` → `2000000` / `1000000`
2. `--dim 64` → `--dim 32`
3. `--snapshots 4` → `--snapshots 3`

示例：

```bash
python src/edane_full_pipeline.py \
  --mode file --model edane \
  --snapshots 3 --snapshot-mode window \
  --max-nodes 1000000 \
  --dim 32 \
  --node-selection-mode bounded \
  --partition-dir /store/$USER/edane_partitions \
  --output-dir /store/$USER/edane_outputs/run_fallback_1m_d32 \
  --no-all-results
```

---

## 10. 完成后检查结果

每轮至少检查：

- `summary.json`
- `metrics_per_snapshot.csv`
- `metrics_curves.svg`

默认输出目录：

- `/store/$USER/edane_outputs/`







































平台后台UI设计优化：

1.租户管理：第一个卡片查询标题和输入框放在同一行，按钮右靠齐。整体只占一行
