# EDANE 模块一至模块四整合文档

> 本文档将原有四份模块文档整合为一份统一说明，便于连续阅读、论文整理与答辩准备。
>
> 对应代码：`src/edane.py`、`src/edane_full_pipeline.py`

---

## 目录

1. [整体说明](#整体说明)
2. [模块总览](#模块总览)
3. [模块一：高效初始化](#模块一高效初始化)
4. [模块二：局部增量更新](#模块二局部增量更新)
5. [模块三：双曲门控融合](#模块三双曲门控融合)
6. [模块四：量化压缩与轻量存储](#模块四量化压缩与轻量存储)
7. [模块之间的衔接关系](#模块之间的衔接关系)
8. [当前实现边界与统一表述口径](#当前实现边界与统一表述口径)

---

## 整体说明

EDANE 当前代码实现可理解为一个四模块流水线：

1. **模块一**：在初始快照上生成结构初始嵌入；
2. **模块二**：当图结构或属性变化时，局部更新已有嵌入；
3. **模块三**：融合结构表示与属性表示，得到最终节点嵌入；
4. **模块四**：将最终嵌入压缩为更轻量的存储形式。

整体设计思路可以概括为：

> **高效初始化 + 局部更新 + 自适应融合 + 压缩存储**

这四个模块并不是互相独立的算法，而是一个连续的实验原型系统：

- 模块一给模块二提供初始结构底座；
- 模块二更新局部结构与属性后，模块三只对受影响节点重融合；
- 模块三输出最终浮点嵌入，再由模块四量化压缩；
- 模块四在 `fit()` 和 `apply_updates()` 之后都会自动刷新。

---

## 模块总览

| 模块 | 目标 | 当前实现关键词 | 对应代码 |
|------|------|----------------|----------|
| 模块一 | 初始结构嵌入 | 稀疏随机投影锚点 + 显式目标优化 | `EDANE._compute_structure_embedding()` |
| 模块二 | 动态局部更新 | 一跳邻域、局部差分修正、soft deletion | `EDANE.apply_updates()` |
| 模块三 | 结构属性融合 | 双曲映射 + 轻量门控 + Möbius 融合 | `EDANE._fuse_embeddings()` |
| 模块四 | 轻量存储 | 逐维对称 int8 量化 + 可选 binary 副本 | `EDANE._refresh_quantized_embedding()` |

---

## 模块一：高效初始化

### 1. 目标与定位

模块一负责在**初始快照**上生成结构初始嵌入 `Y_struct`，是模块二、模块三、模块四的共同起点。

核心目标：

1. 保留稀疏随机投影的高效率；
2. 在传播基础上加入显式目标函数与迭代优化；
3. 让结构初始表示具备可解释性和后续可更新性。

一句话概括：

> 模块一不是“随机给一个初始向量”，而是用**锚点传播 + 显式优化**得到一个稳定的结构起点。

### 2. 输入与输出

#### 输入

- 邻接矩阵 `A`（稀疏或稠密，代码统一转为 CSR）
- 节点数 `N`
- 嵌入维度 `d`
- 传播阶数 `q`
- 稀疏随机投影密度 `p`
- 初始化迭代参数（步长、正则、最大轮数、早停阈值）

#### 输出

- 结构嵌入 `Y_struct ∈ R^{N×d}`（按行归一化）

### 3. 设计逻辑：两阶段初始化

模块一采用“两阶段初始化”：

1. **锚点构建（Anchor Construction）**
2. **显式目标优化（Objective Refinement）**

#### 阶段 A：锚点构建

先构造归一化结构算子：

$$
S = D^{-1/2}(A+I)D^{-1/2}
$$

再采样三元稀疏随机投影矩阵：

$$
R \in \mathbb{R}^{N\times d},\quad r_{ij}\in\{-1,0,1\}
$$

通过多阶传播构造锚点：

$$
Y_{anchor}=\sum_{k=0}^{q}\alpha_k S^kR
$$

其中：

- `α_k` 可由 `structure_weights` 指定；
- 若未指定，则默认按 `1/(k+1)` 归一化自动生成。

#### 阶段 B：显式目标优化

在锚点基础上最小化：

$$
J(Y)=\|Y-SY\|_F^2 + \lambda\|Y-Y_{anchor}\|_F^2
$$

含义：

- 第一项：结构平滑项（邻接一致性）
- 第二项：锚点约束项（防止优化偏移过大）

梯度下降更新：

$$
Y^{(t+1)}=Y^{(t)}-\eta\nabla J(Y^{(t)})
$$

代码中的梯度实现对应：

- `residual = Y - S@Y`
- `smooth_grad = residual - S^T@residual`
- `anchor_grad = Y - Y_anchor`
- `grad = 2 * (smooth_grad + λ * anchor_grad)`

每轮更新后会：

- 行归一化；
- 计算相对变化量；
- 小于阈值则早停。

### 4. 参数说明（模块一相关）

| 参数 | 默认值 | 作用 | 调参建议 |
|---|---:|---|---|
| `dim` | 32 | 嵌入维度 `d` | 32/64 常用 |
| `order` | 2 | 传播阶数 `q` | 1~3 常见，过大易过平滑 |
| `projection_density` | 0.1 | 随机投影非零密度 | 图很大时可降到 0.05 |
| `structure_weights` | `None` | 多阶传播权重 `α_k` | 未设时自动递减 |
| `init_iterations` | 24 | 最大迭代轮数 | 10~40 常见 |
| `init_step_size` | 0.35 | 梯度步长 | 不稳时降低 |
| `init_reg` | 0.2 | 锚点正则 `λ` | 噪声高时可增大 |
| `init_tol` | 1e-4 | 早停阈值 | 对速度敏感可放宽 |

### 5. 复杂度与内存

时间复杂度近似：

$$
O((q+T)\cdot |E|\cdot d)
$$

其中：

- 锚点阶段约为 `O(q·|E|·d)`
- 迭代优化约为 `O(T·|E|·d)`

空间复杂度：

- `S` 采用 CSR：`O(|E|)`
- `R`、`Y`、`Y_anchor`：`O(N·d)`

### 6. 代码映射

- `_prepare_adjacency()`：邻接矩阵预处理
- `_normalized_adjacency()`：构造 `S`
- `_sample_sparse_random_matrix()`：采样 `R`
- `EDANE._compute_structure_embedding()`：模块一主实现

### 7. 当前边界

模块一已经具备：

- 明确目标函数
- 明确迭代更新规则
- 明确早停策略

但它仍然是：

> **工程化近似优化实现**，不是带严格收敛证明的理论完备版本。

---

## 模块二：局部增量更新

### 1. 目标与定位

模块二负责在动态图变化到来时，避免全量重训练，仅对受影响节点进行局部更新。

核心思想是：

> 把动态图变化视作**局部扰动问题**，而不是每次都把整张图重新训练一遍。

### 2. 输入与输出

#### 输入（`apply_updates`）

```python
apply_updates(
    node_additions: Optional[Dict[int, np.ndarray]] = None,
    node_removals: Optional[Iterable[int]] = None,
    edge_additions: Optional[Iterable[Tuple[int, int]]] = None,
    edge_removals: Optional[Iterable[Tuple[int, int]]] = None,
    attr_updates: Optional[Dict[int, np.ndarray]] = None,
) -> List[int]
```

参数含义：

- `node_additions`：新增节点索引 → 原始属性向量
- `node_removals`：删除节点索引列表（当前实现为软删除）
- `edge_additions`：新增边列表
- `edge_removals`：删除边列表
- `attr_updates`：节点属性更新（原始值，不是标准化值）

#### 输出

- 返回 `affected_nodes`：本轮受影响节点索引列表

### 3. 依赖模块一与当前状态

模块二依赖 `fit()` 后保存的状态：

- `self.adj`
- `self.attrs`
- `self.struct_emb`
- `self.attr_emb`
- `self.embedding_`
- `self.random_projection`
- `self.attr_projection`
- `self.feature_mean_ / self.feature_std_`

如果这些状态不存在，`apply_updates()` 会报错并要求先调用 `fit()`。

### 4. 核心流程

当前代码中的模块二执行顺序为：

1. 变化检测（节点增删、边增删、属性变更）
2. 构建变化前一跳邻域
3. 落地图更新与属性更新
4. 构建变化后一跳邻域，形成受影响集合
5. 执行局部结构更新
6. 重算变化节点属性投影
7. 仅对受影响节点做融合更新
8. 刷新量化副本

### 5. 四类变化检测

#### 节点增加（`ΔV+`）

当前实现要求新增索引连续追加，例如当前节点数为 `N`，新增两个节点时索引必须是 `N, N+1`。

处理动作：

1. 扩展邻接矩阵维度
2. 对新增节点属性按原有统计量标准化
3. 扩展随机投影矩阵 `R`
4. 初始化新增节点结构、属性与融合向量

#### 节点删除（`ΔV-`）

当前采用 **soft deletion**：

1. 清空相关边（行列置零）
2. 属性与嵌入向量置零

这样可以保持索引稳定，不必重排整张图。

#### 边增加（`ΔE+`）

- 对称写入 `(u,v)` 与 `(v,u)`
- 忽略自环

#### 边删除（`ΔE-`）

- 对称清除 `(u,v)` 与 `(v,u)`
- 忽略自环

### 6. 一跳邻域分析（`k=1`）

当前文档与代码统一采用 `k=1`：

$$
U = T \cup N_{before}(T) \cup N_{after}(T)
$$

其中：

- `T`：直接变化节点集合
- `N_before(T)`：变化前一跳邻居
- `N_after(T)`：变化后一跳邻居

### 7. 增量更新公式与当前实现

文档层面，可将模块二抽象写成：

$$
y_i' = y_i + \alpha \cdot f(\Delta A_i y_{N(i)})
$$

其中 `f` 为线性加权聚合（mean）。

代码中的工程化落地更具体：

1. 求变化邻居集合：
   - `add_neighbors = N_after(i) - N_before(i)`
   - `rem_neighbors = N_before(i) - N_after(i)`
2. 构造扰动项：

$$
\Delta A_i y_{N(i)} \approx \sum_{j\in add} y_j - \sum_{j\in rem} y_j
$$

3. 用当前邻域规模做 mean 聚合：

$$
f(\cdot)=\frac{1}{\max(|N_{after}(i)|,1)}(\cdot)
$$

4. 用 `learning_rate` 作为 `\alpha`：

$$
y_i' = y_i + \text{learning\_rate} \cdot f(\cdot)
$$

5. 最后对结构向量做行归一化。

### 8. 属性更新规则

`attr_updates` 与 `node_additions` 的属性向量会按 `fit()` 保存的 `mean/std` 标准化后写入。

属性更新后只做局部重算：

1. 重算变化节点对应的属性投影 `attr_emb[idx]`
2. 行归一化

### 9. 与模块三、模块四的联动

结构与属性更新后：

1. 仅对受影响节点调用模块三融合
2. 回写全局 `embedding_`
3. 刷新模块四量化副本

### 10. 复杂度与边界

设：

- `|T|`：直接变化节点数
- `|U|`：受影响节点数
- `d`：嵌入维度

主要开销近似：

- 邻域差分与扰动计算：`O((|ΔE| + |U|) · d)`
- 受影响节点融合：`O(|U| · d)`

当前边界：

1. `ΔV-` 为 soft deletion
2. `ΔV+` 要求新增索引连续
3. 只实现 `k=1`
4. **不是严格谱分解闭式更新**，而是局部差分修正的工程化近似实现

---

## 模块三：双曲门控融合

### 1. 目标与定位

模块三负责把模块一得到的结构嵌入 `H_s` 与属性投影得到的属性嵌入 `H_x` 融合为最终节点表示 `Z`。

核心目标：

1. 让结构信息与属性信息自适应平衡；
2. 在双曲空间中建模潜在层级结构；
3. 与模块二局部更新兼容，只对受影响节点重融合。

### 2. 输入与输出

#### 输入

- `struct_emb = H_s ∈ R^{N×d}`
- `attr_emb = H_x ∈ R^{N×d}`

#### 输出

- `embedding = Z ∈ R^{N×d}`：最终融合嵌入

### 3. 四步主流程

#### 第一步：映射到双曲空间

$$
p_s = \exp_0(0.5H_s), \quad p_x = \exp_0(0.5H_x)
$$

这里用 `0.5` 缩放是为了数值稳定。

#### 第二步：计算门控系数

$$
g = \sigma(W[p_s \| p_x] + b)
$$

当前实现中：

- `W` 对应 `self.fusion_weight_`
- `b` 对应 `self.fusion_bias_`
- 这是一个**逐节点、可学习、标量门控**

#### 第三步：双曲加权融合

$$
p = g \otimes p_s \oplus (1-g) \otimes p_x
$$

其中：

- `⊗`：Möbius 标量乘
- `⊕`：Möbius 加法

#### 第四步：映回欧氏空间

$$
Z = \log_0(p)
$$

并做行归一化。

### 4. 训练方式

模块三不是完整的下游监督训练网络，而是一个**轻量门控融合器**。

当前训练方式：

1. 构造门控特征 `[p_s | p_x]`
2. 计算预测门控 `pred = sigmoid(features @ W + b)`
3. 用结构与属性嵌入相似性构造伪目标：

$$
target = \sigma\left(\frac{\langle h_s, h_x \rangle}{\sqrt{d}}\right)
$$

4. 对 `W, b` 做轻量梯度下降更新

因此更准确的说法是：

> 当前实现是**轻量双曲门控融合**，而不是重型多头注意力网络。

### 5. 关键函数链

1. `_build_fusion_gate_features(struct_emb, attr_emb)`
2. `_compute_fusion_gate(features)`
3. `_fuse_embeddings(struct_emb, attr_emb)`

在 `fit()` 中：

1. 模块一输出 `struct_emb`
2. 属性投影得到 `attr_emb`
3. 初始化门控参数
4. 轻量训练门控参数
5. 调用 `_fuse_embeddings()` 得到最终嵌入

在 `apply_updates()` 中：

- 只对受影响节点局部重算并重融合。

### 6. 数值稳定性设计

为避免双曲空间数值爆炸，当前实现加入了：

1. 安全范数函数
2. `tanh / arctanh` 输入裁剪
3. Möbius 运算后半径裁剪
4. 输出后统一行归一化

### 7. 参数说明

| 参数 | 默认值 | 作用 | 调参建议 |
|---|---:|---|---|
| `fusion_train_steps` | 40 | 门控训练步数 | 20~80 常见 |
| `fusion_lr` | 0.1 | 门控学习率 | 不稳可降到 0.05 |
| `fusion_weight_decay` | 1e-4 | 正则项 | 抖动明显时可适度增大 |

### 8. 当前边界

模块三已经做到：

- 双曲映射
- 门控参数学习
- Möbius 融合
- 局部重融合

但它仍然是：

1. 单层门控，而非复杂注意力网络
2. 无监督伪目标训练，而非完整任务监督训练
3. 更偏工程原型，而不是极致表达能力版本

---

## 模块四：量化压缩与轻量存储

### 1. 目标与定位

模块四负责将最终嵌入从高精度浮点表示压缩为更轻量的存储形式，同时保留可恢复、可统计、可评估的能力。

当前支持：

- `int8` 量化（主实现）
- `binary` 副本（可选）
- 压缩比统计
- 重构误差统计

### 2. 输入与输出

#### 输入

- 最终浮点嵌入 `Z ∈ R^{N×d}`

#### 输出

- `QuantizedEmbedding`
- `BinaryEmbedding`
- `quantization_error_`
- `binary_error_`
- `quantization_compression_ratio_`
- `binary_compression_ratio_`

### 3. Int8 量化设计

当前实现采用**逐维对称量化**。

对第 `j` 维，先计算缩放系数：

$$
s_j = \frac{\max_i |z_{ij}|}{127}
$$

然后量化：

$$
q_{ij} = \text{clip}\left(\text{round}\left(\frac{z_{ij}}{s_j}\right), -127, 127\right)
$$

恢复：

$$
\hat z_{ij} = q_{ij} \cdot s_j
$$

对应类：`QuantizedEmbedding`

- `values`：`int8` 量化矩阵
- `scale`：每维缩放系数
- `dequantize()`：恢复为近似浮点矩阵

### 4. Binary 二值副本

当前采用符号二值化：

$$
b_{ij} =
\begin{cases}
1, & z_{ij} \ge 0 \\
-1, & z_{ij} < 0
\end{cases}
$$

对应类：`BinaryEmbedding`

- `values`：当前实现用 `float32` 的 `±1` 表示
- `dequantize()`：返回 `±1` 浮点形式

需要说明：

> 当前 binary 还不是 bit-pack 位级压缩版本，而是更稳定、便于调试的实验性二值副本。

### 5. 压缩误差与压缩比统计

#### Int8 重构误差

$$
E_{int8} = \frac{\|Z - \hat Z_{int8}\|_F}{\|Z\|_F}
$$

对应：`quantization_error_`

#### Binary 重构误差

$$
E_{bin} = \frac{\|Z - \hat Z_{bin}\|_F}{\|Z\|_F}
$$

对应：`binary_error_`

#### Int8 压缩比

$$
R_{int8} = \frac{\text{float bytes}}{\text{int8 values bytes} + \text{scale bytes}}
$$

对应：`quantization_compression_ratio_`

#### Binary 压缩比

$$
R_{bin} = \frac{\text{float bytes}}{\text{binary values bytes}}
$$

对应：`binary_compression_ratio_`

### 6. 执行时机

模块四不是独立手动执行，而是在嵌入更新后自动刷新：

#### 在 `fit()` 后

1. 模块一、三完成初始嵌入
2. 调用 `_refresh_quantized_embedding()`
3. 刷新 int8 副本
4. 根据配置刷新 binary 副本
5. 计算误差与压缩比

#### 在 `apply_updates()` 后

1. 局部更新 `embedding_`
2. 再次调用 `_refresh_quantized_embedding()`
3. 让压缩副本与最新嵌入保持一致

### 7. 对应导出文件

在 `edane_full_pipeline.py` 中，模块四相关输出包括：

- `final_embedding.npy`
- `final_embedding_int8.npy`
- `final_embedding_scale.npy`
- `final_embedding_binary.npy`（若开启 binary）

同时写入 `summary.json`：

- `quantization_compression_ratio`
- `quantization_error`
- `binary_compression_ratio`
- `binary_error`

### 8. 当前边界

模块四当前更准确的定位是：

1. 主量化方案是逐维对称 int8
2. binary 用于实验对比
3. 重点是轻量、可恢复、可统计
4. 不是极致压缩或 learned compression 系统

---

## 模块之间的衔接关系

四个模块之间的调用关系如下：

### 在 `fit()` 中

1. 预处理图与属性
2. 模块一生成 `struct_emb`
3. 属性投影得到 `attr_emb`
4. 模块三融合得到 `embedding_`
5. 模块四刷新量化副本

### 在 `apply_updates()` 中

1. 模块二处理节点/边/属性变化
2. 更新局部 `struct_emb` 与 `attr_emb`
3. 模块三只对受影响节点重融合
4. 模块四再次刷新量化副本

因此，四模块不是简单并列关系，而是：

> **模块一提供初始结构底座，模块二维持动态图连续更新，模块三输出最终融合表示，模块四负责轻量存储。**

---

## 当前实现边界与统一表述口径

为了避免论文、文档和代码再次出现口径漂移，建议统一使用以下说法：

### 模块一

> 当前实现是**稀疏随机投影锚点 + 显式目标优化初始化**。

### 模块二

> 当前实现是**局部一跳增量更新的工程化近似版本**，不是严格矩阵谱扰动闭式解。

### 模块三

> 当前实现是**轻量双曲门控融合**，不是重型多头注意力网络。

### 模块四

> 当前实现以**逐维对称 int8 量化**为主，binary 副本偏实验性质。

### 整体项目定位

> 当前项目是**可运行的研究原型 / 实验系统**，不是工业级分布式生产实现。

---

## 最后总结

如果把 EDANE 当前实现压缩成一句最准确的话，可以表述为：

> **它是一套以显式优化初始化为起点、以局部增量更新为核心、以双曲门控融合作为统一表示层、并以量化压缩作为存储层的动态图属性网络嵌入研究原型。**

这份整合文档适合你后续：

- 统一阅读四个模块；
- 写毕业论文的方法章节；
- 准备答辩时的模块化讲解；
- 检查文档表述是否与当前代码一致。
