# 模块三（双曲几何协同融合）设计与使用文档

> 适用代码：`src/edane.py`
>
> 适用类/方法：`EDANE._fuse_embeddings()` 及其相关辅助函数

---

## 1. 目标与定位

模块三负责把模块一得到的**结构嵌入** `H_s` 与属性投影得到的**属性嵌入** `H_x` 融合为最终节点表示 `Z`。

本实现按“文档严格版”保留以下核心思想：

1. 先将欧氏空间表示映射到双曲空间
2. 使用门控注意力计算结构与属性的融合权重
3. 在双曲空间中做 Möbius 加权融合
4. 再映射回欧氏空间，作为最终输出嵌入

与最早的原型不同，当前模块三已经从“固定点积门控”升级为**可学习门控注意力**。

---

## 2. 输入输出定义

### 2.1 输入

- `struct_emb = H_s ∈ R^{N×d}`：结构嵌入
- `attr_emb = H_x ∈ R^{N×d}`：属性嵌入

### 2.2 输出

- `embedding = Z ∈ R^{N×d}`：融合后的最终嵌入

---

## 3. 文档版四步流程

模块三严格按照四步设计：

### 第一步：指数映射到双曲空间

将结构与属性表示映射到 Poincaré Ball：

$$
p_{struct} = \exp_0(H_s), \quad p_{attr} = \exp_0(H_x)
$$

代码里通过：

- `_exp_map_zero()`

完成。

### 第二步：门控注意力权重计算

文档要求：

$$
\alpha = \sigma(W[p_{struct} \| p_{attr}] + b)
$$

当前实现中：

- `W` 对应 `self.fusion_weight_`
- `b` 对应 `self.fusion_bias_`
- `[p_struct | p_attr]` 通过拼接双曲表示构造特征

因此这是一个**逐节点、可学习、标量门控**。

### 第三步：双曲加权融合

按文档形式：

$$
p_{final} = \alpha \otimes p_{struct} \oplus (1-\alpha) \otimes p_{attr}
$$

其中：

- `⊗`：Möbius 标量乘（`_mobius_scalar_mul`）
- `⊕`：Möbius 加法（`_mobius_add`）

### 第四步：映射回欧氏空间

$$
Z = \log_0(p_{final})
$$

代码中使用 `_log_map_zero()`，最后再做行归一化。

---

## 4. 当前实现相对文档的具体落地方式

模块三的关键函数链如下：

1. `_build_fusion_gate_features(struct_emb, attr_emb)`
   - 构造双曲空间拼接特征
2. `_compute_fusion_gate(features)`
   - 计算 `alpha = sigmoid(Wx+b)`
3. `_fuse_embeddings(struct_emb, attr_emb)`
   - 做 Möbius 加权融合并映射回欧氏空间

在 `fit()` 中的调用顺序为：

1. 模块一输出 `struct_emb`
2. 属性投影得到 `attr_emb`
3. 初始化门控参数
4. 轻量训练门控参数
5. 调用 `_fuse_embeddings()` 得到最终嵌入

在 `apply_updates()` 中：

- 只对受影响节点局部重算结构/属性后，再调用 `_fuse_embeddings()` 局部更新最终嵌入

---

## 5. 可学习门控的训练设计

### 5.1 为什么需要训练门控

如果直接用点积门控，虽然简单，但本质上仍是手工规则，不是文档里“注意力网络”的实现。

因此当前实现加入了轻量可学习参数：

- `fusion_weight_ ∈ R^{2d×1}`
- `fusion_bias_ ∈ R`

### 5.2 训练目标

当前代码采用一个**轻量伪目标训练**：

- 用结构与属性嵌入的相似性

$$
target = \sigma\left(\frac{\langle h_s, h_x \rangle}{\sqrt{d}}\right)
$$

作为门控目标值，训练 `W,b` 学会从双曲拼接特征中预测这一门控强度。

这不是完整监督学习，而是一个工程上可落地、又符合文档意图的轻量注意力训练方式。

### 5.3 更新方式

在 `_train_fusion_gate()` 中：

1. 构造 `features = [p_struct | p_attr]`
2. 计算 `pred = sigmoid(features @ W + b)`
3. 计算 `diff = pred - target`
4. 对 `W,b` 做梯度下降更新

并带有简单的 `weight_decay` 正则。

---

## 6. 参数说明

以下参数属于模块三：

| 参数 | 默认值 | 作用 | 调参建议 |
|---|---:|---|---|
| `fusion_train_steps` | 40 | 门控训练步数 | 20~80 常用 |
| `fusion_lr` | 0.1 | 门控训练学习率 | 门控不稳可降到 0.05 |
| `fusion_weight_decay` | 1e-4 | 门控参数正则 | 过拟合/抖动时适当增大 |

内部状态：

- `fusion_weight_`
- `fusion_bias_`

---

## 7. 数学流程汇总

设：

- `H_s`：结构嵌入
- `H_x`：属性嵌入

则模块三执行：

### 7.1 双曲映射

$$
p_s = \exp_0(0.5H_s), \quad p_x = \exp_0(0.5H_x)
$$

### 7.2 门控

$$
g = \sigma(W[p_s \| p_x] + b)
$$

### 7.3 双曲融合

$$
p = g \otimes p_s \oplus (1-g) \otimes p_x
$$

### 7.4 回欧氏空间

$$
Z = \log_0(p)
$$

### 7.5 归一化

$$
Z \leftarrow \frac{Z}{\|Z\|}
$$

---

## 8. 数值稳定性设计

双曲空间最容易出问题的是数值爆炸，因此当前实现加入了以下保护：

1. `_safe_norm()`：避免除零
2. `tanh / arctanh` 输入裁剪
3. Möbius 运算后半径裁剪，避免超出 Poincaré 球边界
4. 输出后统一做行归一化

这使模块三在本科毕设规模的数据上更稳。

---

## 9. 与模块二的衔接方式

模块三不是每次全量重算，而是与模块二联动：

1. 模块二找到 `affected_nodes`
2. 模块二更新这些节点的 `struct_emb` / `attr_emb`
3. 模块三只对这些节点调用 `_fuse_embeddings()`
4. 回写到 `embedding_`

这样保留了“增量更新”的整体思想。

---

## 10. 使用方式

### 10.1 默认使用（无需手动调用）

只要调用：

```python
model.fit(adj, attrs)
```

模块三就会自动执行：

1. 初始化门控参数
2. 训练门控参数
3. 生成融合嵌入

### 10.2 自定义参数

```python
model = EDANE(
    dim=64,
    fusion_train_steps=60,
    fusion_lr=0.05,
    fusion_weight_decay=1e-3,
)
```

---

## 11. 使用示例

```python
from edane import EDANE

model = EDANE(
    dim=64,
    order=2,
    fusion_train_steps=50,
    fusion_lr=0.08,
    fusion_weight_decay=1e-4,
)

model.fit(adj, attrs)
embedding = model.get_embedding(dequantize=False)
```

如果你想看融合参数是否已初始化，可以在调试时查看：

```python
print(model.fusion_weight_.shape)
print(model.fusion_bias_)
```

---

## 12. 调参与排错建议

### 12.1 融合后效果和原来差不多

- 增加 `fusion_train_steps`
- 检查结构和属性输入是否本身差异很小

### 12.2 融合后波动大

- 降低 `fusion_lr`
- 增大 `fusion_weight_decay`

### 12.3 门控学习太慢

- 适当增大学习率
- 减少嵌入维度或先检查模块一/属性投影质量

### 12.4 数值异常

- 检查输入嵌入是否包含 NaN/Inf
- 检查是否有人为删除了归一化步骤

---

## 13. 当前边界（实话实说）

虽然模块三已经明显更接近文档，但仍有两个边界：

1. 当前“注意力网络”是**单层线性门控**，不是更深的神经网络
2. 当前门控训练使用的是**轻量伪目标**，不是基于外部监督标签的联合任务训练

这意味着：

- 它已经是“可学习注意力融合”
- 但仍属于**工程化轻量实现**，不是重型深度双曲模型

---

## 14. 与论文/答辩表述建议

你可以这样描述模块三：

1. 首先将结构嵌入与属性嵌入映射到 Poincaré 球
2. 然后通过可学习门控注意力计算融合权重
3. 在双曲空间中使用 Möbius 运算进行加权融合
4. 最后通过对数映射回到欧氏空间作为最终节点表示

如果老师追问“是不是完整深度注意力网络”，你可以明确说：

> 当前实现采用轻量可学习门控，是对文档中注意力机制的工程化落地版本。

---

## 15. 最小复现清单

1. 运行：

```bash
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 3
```

2. 确认运行成功并生成 `summary.json`
3. 若需要调试融合模块，可打印 `fusion_weight_` 与 `fusion_bias_`

只要流程能正常输出最终 embedding，就说明模块三已被完整调用。
