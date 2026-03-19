# EDANE 检测 / 评估指标设计与使用文档

> 适用代码：`src/edane_full_pipeline.py`、`src/plot_metrics_svg.py`、`src/generate_thesis_conclusion.py`
>
> 适用函数：`evaluate_snapshot()`、`macro_micro_f1()`、`auc_from_scores()`、`average_precision_from_scores()`

---

## 1. 文档目的

这份文档专门说明当前项目中“检测指标 / 评估指标”是如何定义、计算、解释和输出的。

它回答 5 个问题：

1. 当前项目到底评估哪些指标？
2. 每个指标在数学上是什么意思？
3. 代码里是怎么实现的？
4. 每个指标应该怎么解释，适合说明什么问题？
5. 在论文和答辩中，如何正确表述这些指标？

---

## 2. 当前项目已输出的评估指标

当前 `evaluate_snapshot()` 输出以下指标：

### 节点分类指标
- `macro_f1`
- `micro_f1`

### 链路预测指标
- `link_auc`
- `link_ap`

### 网络重构指标
- `reconstruction_auc`

### 补充统计指标
- `labeled_nodes`
- `initialization_latency_ms`
- `avg_update_latency_ms`
- `p95_update_latency_ms`
- `quantization_compression_ratio`
- `quantization_error`
- `binary_compression_ratio`
- `binary_error`

---

## 3. 评估总览：指标分别回答什么问题

| 指标 | 任务类型 | 它回答的问题 |
|---|---|---|
| `macro_f1` | 节点分类 | 模型对各类别是否均衡有效？ |
| `micro_f1` | 节点分类 | 模型整体分类准确性如何？ |
| `link_auc` | 链路预测 | 模型是否能把真实边排在假边前面？ |
| `link_ap` | 链路预测 | 模型对真实边的精准捕捉能力如何？ |
| `reconstruction_auc` | 网络重构 | 嵌入向量能否重新还原原图结构？ |
| `update_latency_ms` | 系统效率 | 每次动态更新需要多久？ |
| `quantization_compression_ratio` | 存储效率 | 压缩后节省了多少存储？ |
| `quantization_error` | 压缩质量 | int8 量化带来了多大向量失真？ |
| `binary_error` | 极限压缩质量 | binary 副本损失是否过大？ |

---

## 4. 节点分类指标

节点分类的任务是：

> 把嵌入向量当作节点特征，训练一个轻量分类器，预测节点类别标签。

当前项目中：

- 默认分类器：`logreg`（NumPy 实现的 softmax 逻辑回归）
- 备选分类器：`centroid`（最近类中心）

### 4.1 Macro-F1

对每个类别分别计算 F1，再做简单平均。

F1 定义：

$$
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
$$

Macro-F1 定义：

$$
Macro\text{-}F1 = \frac{1}{C}\sum_{c=1}^{C} F1_c
$$

#### 含义
- 每个类别权重相同
- 对少数类 / 长尾类更敏感
- 能衡量“是否只会分大类，不会分小类”

#### 当前代码位置
- `src/edane_full_pipeline.py::macro_micro_f1()`

### 4.2 Micro-F1

将所有类别的 TP/FP/FN 汇总后再统一计算 F1。

#### 含义
- 多数类影响更大
- 更反映整体准确率/总体分类效果
- 通常会高于 Macro-F1

#### 解释建议
- **Macro-F1 低，Micro-F1 高**：说明模型偏向多数类
- **Macro-F1 和 Micro-F1 都高**：说明模型整体和各类都较稳

---

## 5. 链路预测指标

链路预测的任务是：

> 给定节点向量，计算两个节点之间的相似度，判断它们之间是否应该存在边。

当前项目的做法：

1. 从当前图中采样真实边（正样本）
2. 采样不存在的边（负样本）
3. 使用 `cosine_scores()` 计算节点对余弦相似度
4. 基于打分计算 `link_auc` 和 `link_ap`

### 5.1 Link prediction ROC-AUC (`link_auc`)

当前代码中的 AUC 含义是：

> 随机抽一个正样本边和一个负样本边，正样本得分高于负样本的概率。

代码实现：

$$
AUC = \mathbb{E}[\mathbb{1}(s^+ > s^-) + 0.5\cdot \mathbb{1}(s^+ = s^-)]
$$

#### 当前代码位置
- `src/edane_full_pipeline.py::auc_from_scores()`

#### 含义
- 关注整体排序能力
- 不太依赖阈值设置
- 适合图链接预测这种正负样本不平衡问题

### 5.2 Link prediction Average Precision (`link_ap`)

AP 的核心思想是：

> 当模型把边按得分从高到低排序时，真实边是否排得足够靠前。

当前项目实现方式：

1. 把正负边得分拼接在一起
2. 按得分从高到低排序
3. 每遇到一个正样本，记录当前 precision
4. 对这些 precision 求平均

公式上可以理解为：

$$
AP = \frac{1}{P} \sum_{k \in \text{positive ranks}} Precision@k
$$

其中 `P` 是正样本数。

#### 当前代码位置
- `src/edane_full_pipeline.py::average_precision_from_scores()`

#### 含义
- 比 AUC 更关注真实边是否排在前面
- 对正样本识别质量更敏感
- 在稀疏图场景下很重要

#### 术语建议
在论文和答辩里，建议明确写成：

- **Link prediction ROC-AUC**
- **Link prediction Average Precision (AP)**

而不是简单写“AUC/AP”不解释。

---

## 6. 网络重构指标

网络重构的任务是：

> 看嵌入向量是否还能把原图的边结构重新还原出来。

当前项目中：

- 仍然使用节点对余弦相似度作为重构打分
- 重新采样一组正边/负边
- 用同样的 AUC 计算逻辑评估

### 6.1 Edge reconstruction ROC-AUC (`reconstruction_auc`)

它和 `link_auc` 的形式相同，但**解释目标不同**：

- `link_auc`：强调“边存在预测能力”
- `reconstruction_auc`：强调“嵌入是否保留了原始结构信息”

#### 当前代码位置
- `src/edane_full_pipeline.py::evaluate_snapshot()`
  - `recon_pos_pairs, recon_neg_pairs = sample_link_pairs(..., seed + 999)`
  - `metrics["reconstruction_auc"] = auc_from_scores(...)`

#### 为什么要单独输出
虽然实现上和 link AUC 形式相近，但从论文结构上，单独列出 `reconstruction_auc` 更符合 `edane.md` 的指标设计，也更便于说明“嵌入是否编码了图结构”。

#### 术语建议
建议在文档中写成：

- **Edge reconstruction ROC-AUC**

或者简写：

- **Reconstruction ROC-AUC**

---

## 7. 辅助统计指标

这些不是“预测质量指标”，但在系统评估中非常重要。

### 7.1 `labeled_nodes`

表示当前快照中可参与分类评估的节点数。

#### 作用
- 判断当前数据集标签覆盖率
- 避免把“分类效果差”误解成“模型一定差”，因为可能只是标签太少

### 7.2 初始化时延

- `initialization_latency_ms`

表示从 `fit()` 开始到初始嵌入生成完成的耗时。

#### 作用
- 衡量模块一 + 模块三 + 模块四在初始快照上的总开销

### 7.3 动态更新时间

- `avg_update_latency_ms`
- `p95_update_latency_ms`

#### 含义
- `avg_update_latency_ms`：平均每个快照的更新耗时
- `p95_update_latency_ms`：95 分位更新时延，更适合看“尾部慢情况”

#### 用途
- 证明模块二是否具备实时性优势
- 比只看平均值更稳健

---

## 8. 压缩相关指标

这些指标来自模块四。

### 8.1 `quantization_compression_ratio`

表示原始浮点嵌入与 int8 副本相比节省了多少存储。

公式：

$$
compression\_ratio = \frac{float\ bytes}{int8\ values\ bytes + scale\ bytes}
$$

#### 含义
- 数值越大，说明压缩越明显
- 例如接近 `8x`，表示存储量约缩小到原来的 `1/8`

### 8.2 `quantization_error`

表示 int8 反量化后与原始浮点嵌入之间的相对误差：

$$
E_q = \frac{\|Z - \hat Z_q\|_F}{\|Z\|_F}
$$

#### 含义
- 越小越好
- 说明压缩是否对向量本身造成明显失真

### 8.3 `binary_compression_ratio`

表示 binary 副本的压缩比。

当前实现中 binary 仍是 `±1 float32` 副本，不是位级 bit-pack，
所以这个值还不是理论极限压缩比，而是当前工程实现下的压缩比。

### 8.4 `binary_error`

表示 binary 副本与原始嵌入之间的相对误差。

#### 含义
- 通常会显著高于 int8 误差
- 主要用于“极限压缩实验对照”，不适合当主结果

---

## 9. 当前代码中的指标计算路径

完整路径如下：

1. `run_pipeline()` 调用 `evaluate_snapshot()`
2. `evaluate_snapshot()` 内部：
   - 分类：`macro_micro_f1()`
   - 链路预测：`auc_from_scores()` + `average_precision_from_scores()`
   - 网络重构：再次采样后调用 `auc_from_scores()`
3. 结果写入：
   - `metrics_per_snapshot.csv`
   - `summary.json`
   - `all_results.csv`
4. 曲线图由：
   - `save_metrics_curves_svg()`
   - `src/plot_metrics_svg.py`
   负责绘制

---

## 10. 输出文件中的指标位置

### 10.1 `metrics_per_snapshot.csv`

每个快照都会包含：

- `snapshot`
- `update_latency_ms`
- `macro_f1`
- `micro_f1`
- `link_auc`
- `link_ap`
- `reconstruction_auc`
- `labeled_nodes`

### 10.2 `summary.json`

最终摘要包含：

- `final_macro_f1`
- `final_micro_f1`
- `final_link_auc`
- `final_link_ap`
- `final_reconstruction_auc`
- `initialization_latency_ms`
- `avg_update_latency_ms`
- `p95_update_latency_ms`
- `quantization_compression_ratio`
- `quantization_error`
- `binary_compression_ratio`
- `binary_error`

### 10.3 `all_results.csv`

汇总表会保留每个数据集最新一次运行的最终行，适合后续生成论文结论稿。

---

## 11. 如何正确解读这些指标

### 情况 1：`Micro-F1` 高，但 `Macro-F1` 低
说明模型总体分类还行，但对小类别不友好。

### 情况 2：`link_auc` 高，`link_ap` 低
说明模型有一定排序能力，但对真实边的“前排命中率”不够好。

### 情况 3：`link_auc` 高，但 `reconstruction_auc` 一般
说明当前采样下链路预测还可以，但嵌入对整体原图结构的保留不够强。

### 情况 4：压缩比高，`quantization_error` 小
说明模块四实现效果较好：既省空间又保留了向量质量。

### 情况 5：`binary_error` 很高
这是常见现象，说明 binary 更适合作为对照而非主压缩方案。

---

## 12. 论文 / 答辩表述建议

建议你统一使用下面这套叫法：

- **Macro-F1**
- **Micro-F1**
- **Link prediction ROC-AUC**
- **Link prediction Average Precision (AP)**
- **Edge reconstruction ROC-AUC**
- **Initialization latency**
- **Average update latency**
- **P95 update latency**
- **Quantization compression ratio**
- **Quantization reconstruction error**

这样最规范，也和当前项目输出完全一致。

---

## 13. 当前边界（实话实说）

虽然指标体系已经比较完整，但还有几个边界要清楚：

1. `reconstruction_auc` 当前仍是基于采样边/非边计算，不是全图穷举重构
2. `link_auc` 和 `reconstruction_auc` 形式接近，但解释目标不同
3. `binary_compression_ratio` 目前还不是 bit-pack 后的理论极限值
4. 分类评估仍依赖标签覆盖率，标签过少时 F1 解释应谨慎

---

## 14. 最小复现清单

运行：

```bash
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 3 --quantize --binary-quantize
```

然后查看：

1. `outputs/.../metrics_per_snapshot.csv`
2. `outputs/.../summary.json`
3. `all_results.csv`
4. `metrics_curves.svg`

如果这四处都包含上述字段，就说明评估指标链路已经完整工作。
