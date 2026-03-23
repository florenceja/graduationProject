# 评估设置说明（OAG 固定数据集）

当前仓库后续真实数据实验统一使用：

- `dataset/OAG/`

因此，旧版本中基于 `twitter_sample`、`amazon3m_sample` 等数据集的评估设置对比说明，已不再作为当前默认实验口径。

当前建议：

- 节点分类：`repeated_stratified`
- 训练比例：`--eval-train-ratio 0.7`
- 低支持类别处理：按需要使用 `--label-cleanup-mode eval_only`

示例：

```bash
python src/edane_full_pipeline.py --mode file --snapshots 6 --classifier logreg --eval-protocol repeated_stratified --eval-repeats 10
```
