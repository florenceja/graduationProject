# 评估设置说明（OAG 固定数据集）

当前仓库后续真实数据实验统一使用：

- `data/OAG/`

因此，旧版本中基于历史样本数据集的评估设置对比说明，已不再作为当前默认实验口径。

当前建议：

- 节点分类：`repeated_stratified`
- 训练比例：`--eval-train-ratio 0.7`
- 低支持类别处理：按需要使用 `--label-cleanup-mode eval_only`

示例：

```bash
python src/edane_full_pipeline.py --mode file --snapshots 6 --classifier logreg --eval-protocol repeated_stratified --eval-repeats 10
```

补充说明（重要）：

- 当类别数非常多/分布极不均衡时，分层拆分可能失败。流水线会在 `summary.json` 中记录：
  - `f1_eval_protocol`（你配置的协议）
  - `f1_eval_protocol_used`（实际使用的协议，可能降级为 `single_random_fallback`）
- 若希望更稳定：
  - 生成子图时提高 `--min-venue-support`（减少类别数）；或
  - 评估时提高 `--min-class-support`；或
  - 使用 `--label-cleanup-mode eval_only` 在评估阶段过滤低支持类别。

---

## 为什么 Macro/Micro-F1 经常很低（常见原因排查）

在当前仓库的 OAG-derived 口径下，节点分类标签来自 `venue`（会场/期刊）。如果你观察到 **EDANE / DANE 的 macro-f1、micro-f1 都很低**，通常不是“模型完全失效”，而是下面这些设置/数据特性叠加导致：

1) **类别数太多 + 长尾极端**

- `venue` 作为标签时，类别数往往非常大（上千级），且每个类别的样本数很少。
- 在这种设定下：
  - **Macro-F1** 会被大量小类（测试集中几乎预测不对）拉到很低；
  - **Micro-F1**（近似整体准确率）也会因为“多分类难度极高”而偏低。

2) `--min-class-support` 在 `--label-cleanup-mode off` 时**不会生效**

- 代码里只有在 `--label-cleanup-mode eval_only` 时才会按 `--min-class-support` 过滤低支持类别，并将剩余标签重映射为连续整数。
- 若保持 `off`，则会保留全部长尾类别参与评估，F1 很容易“被拖没”。

3) 分层切分（repeated_stratified）在长尾场景可能降级

- 分层切分要求每个类别在 train/test 中都至少出现一次；当存在大量“每类 1 个样本”的类别时，分层切分会失败。
- 流水线会降级为 `single_random_fallback`，从而使评估波动更大。

你可以用下面的小脚本快速检查 `labels.csv` 的类别数量与长尾情况（只看 label>=0 的带标签节点）：

```bash
python - <<'PY'
import csv
from collections import Counter

labels=[]
with open('data/OAG/labels.csv','r',encoding='utf-8-sig',newline='') as f:
    r=csv.DictReader(f)
    for row in r:
        labels.append(int(float(row['label'])))

labels=[x for x in labels if x>=0]
ctr=Counter(labels)
counts=sorted(ctr.values())
print('labeled_nodes:',sum(counts))
print('class_count:',len(counts))
print('min/median/p90/max:',counts[0],counts[len(counts)//2],counts[int(len(counts)*0.9)],counts[-1])
print('classes with count < 2:',sum(1 for c in counts if c<2))
print('classes with count < 5:',sum(1 for c in counts if c<5))
PY
```

---

## 改进建议（尽量只提升 F1，不影响 link/recon 等其他指标）

本仓库的 `link_auc/link_ap/reconstruction_auc` 是在同一次 `evaluate_snapshot()` 中基于 **embedding+adj** 计算的，与“评估阶段过滤标签/训练分类头”无关。

因此，下面这些改动只会影响 **F1（以及 summary 中的 labeled_nodes / eval_class_count 等统计）**，一般不改变 link/recon 指标：

### 方案 A（推荐默认）：评估阶段过滤长尾类别

让 `--min-class-support` 真正生效：

```bash
python src/edane_full_pipeline.py --mode file --model edane --snapshots 4 \
  --classifier logreg --eval-protocol repeated_stratified --eval-repeats 10 \
  --label-cleanup-mode eval_only --min-class-support 5
```

经验上：
- `min-class-support=2`：尽量保留评估规模，主要用于避免分层切分失败；
- `min-class-support=5/10`：更“像样”的分类结果，但评估覆盖会变小。

### 方案 B：分类头更适配长尾（评估专用）

在不改 embedding 的情况下，尝试让逻辑回归更稳定：

```bash
python src/edane_full_pipeline.py --mode file --model dane --snapshots 4 \
  --classifier logreg --eval-protocol repeated_stratified --eval-repeats 10 \
  --logreg-class-weight balanced \
  --logreg-lr 0.1 --logreg-epochs 800 --logreg-weight-decay 1e-3
```

### 方案 C（谨慎）：重新生成子图降低类别数（会影响其他指标）

如果你在 `prepare_datasets.py` 转换阶段提高 `--min-venue-support`，确实可以显著减少类别数并提升 F1；
但这通常会改变“保留哪些论文/节点/边”，进而影响 embedding 与 link/recon 指标。
如果你的目标是“只提升 F1 而不动其他指标”，优先使用方案 A/B。
