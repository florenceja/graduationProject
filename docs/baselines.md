# 对比算法（Baselines）说明

本文档说明本项目当前已接入的对比算法（以及实现口径/限制），并给出与 EDANE 统一评测的注意事项。

> 入口脚本：`src/edane_full_pipeline.py`，通过 `--model {edane,dane,dtformer}` 选择。

## 1. 总览

| 模型 | 选择参数 | 代码位置 | 复现口径 | 适用对比维度 |
|---|---|---|---|---|
| EDANE | `--model edane` | `src/edane.py` | `native` | 效果 + 更新效率 + 压缩存储 |
| DANE | `--model dane` | `src/dane.py` | `paper_approximation` | 同赛道经典强基线（但偏重谱分解） |
| DTFormer | `--model dtformer` | `src/dtformer.py` | `paper_approximation` | 2024+ 前沿对比（适配统一 embedding 输出） |

所有运行的 `summary.json` 都会记录：

- `model`
- `implementation_fidelity`（EDANE 为 `native`，基线为 `paper_approximation`）

## 2. 公平性与参数约束

### 2.1 EDANE 专属参数

以下参数仅对 `--model edane` 有意义：

- `--quantize` / `--binary-quantize`
- `--no-attr` / `--no-hyperbolic` / `--no-inc`

当 `--model dane/dtformer` 时：

- EDANE 专属消融参数会被拒绝（防止“假消融”）
- 量化开关不会生效，`summary.json` 中会被强制记录为 false

### 2.2 统一评测口径

本项目的对比默认遵循：

- 同一数据输入（`data/OAG`）
- 同一 `snapshots/snapshot_mode`
- 同一分类器与评估协议（`logreg` + `eval_protocol`）

但要注意：当类别数很多时，`repeated_stratified` 可能降级为 `single_random_fallback`。

## 3. DANE（CIKM 2017）

### 3.1 论文来源

- Jundong Li, Harsh Dani, Xia Hu, Jiliang Tang, Yi Chang, Huan Liu.
  **Attributed Network Embedding for Learning in a Dynamic Environment**. CIKM 2017.
- DOI: 10.1145/3132847.3132919
- arXiv: https://arxiv.org/abs/1706.01860

### 3.2 本项目复现口径

- 结构视图：广义谱嵌入近似
- 属性视图：top-k cosine 相似图 + 谱嵌入近似
- 融合：共识投影融合
- 在线更新：扰动式近似更新；数值不稳定或节点增删时回退 refit

### 3.3 特色与不足

特色：

- 同赛道经典对比，能覆盖“动态+属性”的核心定义

不足/风险：

- 谱分解代价较高，随节点数增长更明显
- 数值稳定性依赖求解器（ARPACK/Lobpcg）与参数

## 4. DTFormer（CIKM 2024）

### 4.1 论文来源

- Xi Chen et al. **DTFormer: A Transformer-Based Method for Discrete-Time Dynamic Graph Representation Learning**. CIKM 2024.
- DOI: 10.1145/3627673.3679568
- arXiv: https://arxiv.org/abs/2407.18523
- Official repo: https://github.com/chenxi1228/DTFormer

### 4.2 本项目复现口径

官方 DTFormer 更偏离散时间动态图的事件/链接预测训练。本项目为了统一为“输出 embedding → 统一评估”，实现了 DTFormer-style 的适配器：

- snapshot history
- patching
- 时序位置/强度编码
- Transformer-style temporal aggregation

### 4.3 特色与不足

特色：

- 满足“2024+”前沿对比要求
- 与 EDANE 的“效率/轻量”形成互补对比维度

不足/风险：

- 训练目标与官方实现不完全同构，论文写作需明确为适配版 baseline

## 5. 运行示例

### 5.1 在 OAG 子图上对比

```bash
python src/edane_full_pipeline.py --mode file --model edane --snapshots 4 --max-nodes 15000
python src/edane_full_pipeline.py --mode file --model dane --snapshots 4 --max-nodes 15000
python src/edane_full_pipeline.py --mode file --model dtformer --snapshots 4 --max-nodes 15000
```

如需更快的本机对比，可统一降低：

- `--dim`（例如 16）
- `--eval-repeats`（例如 3）
- `--snapshots`（例如 3）

## 6. MTSN（WWW 2021）— 已提及但当前未接入

### 6.1 论文来源

- Zhijun Liu, Chao Huang, Yanwei Yu, Junyu Dong.
  **Motif-Preserving Dynamic Attributed Network Embedding**. WWW 2021.
- DOI: https://doi.org/10.1145/3442381.3449821
- 代码（作者公开）：https://github.com/ZhijunLiu95/MTSN

### 6.2 方法特色

- 使用 motif-preserving encoder 显式建模高阶结构（motif/triad 等）
- 使用 temporal shift 操作建模快照序列的时间演化
- 同时考虑节点属性与结构，是动态属性网络嵌入的“同赛道强基线”之一

### 6.3 复现难点/不足

- 公开代码偏旧：TensorFlow 1.x + Python 2.7 生态，依赖版本老
- motif 计算会引入额外预处理与复杂依赖
- 对当前仓库的统一 embedding 管线需要做适配（输入/训练脚本/输出接口）

### 6.4 论文对比写法建议

如果论文中需要引用 MTSN 作为强基线：

- 建议将其定位为“同类任务强基线”，说明与 EDANE 的差异侧重点：
  - MTSN 更强调 motif 高阶结构 + temporal shift
  - EDANE 更强调增量更新效率 + 结构/属性融合 + 压缩存储
- 若未能在工程上复现，应在论文中说明原因（依赖生态/实现适配成本），避免被认为是“选择性对比”。
