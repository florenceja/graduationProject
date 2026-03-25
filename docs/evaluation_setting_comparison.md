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
