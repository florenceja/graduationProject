# 模块一（高效初始化）设计与使用文档

> 适用代码：`src/edane.py`
>
> 适用类/方法：`EDANE.fit()` → `_compute_structure_embedding()`

---

## 1. 目标与定位

模块一负责在**初始快照**上生成结构初始嵌入 `Y_struct`，是后续模块二（增量更新）、模块三（融合）、模块四（量化）的共同起点。

本实现的核心目标：

1. 保留稀疏随机投影的高效率（适合大图）
2. 引入显式目标函数与迭代优化（避免“只有传播、没有优化目标”）
3. 保持对稀疏图的线性复杂度近似：`O(T · |E| · d)`

---

## 2. 输入输出定义

### 2.1 输入

- 邻接矩阵 `A`（稀疏或稠密，代码中统一转 CSR）
- 节点数 `N`
- 嵌入维度 `d`
- 传播阶数 `q`
- 稀疏随机投影密度 `p`
- 初始化迭代参数（步长、正则、迭代轮次、早停阈值）

### 2.2 输出

- 结构嵌入 `Y_struct ∈ R^{N×d}`（按行归一化）

---

## 3. 设计逻辑（两阶段）

模块一采用“两阶段初始化”：

1. **锚点构建（Anchor Construction）**
2. **显式目标优化（Objective Refinement）**

### 3.1 阶段 A：锚点构建

先构造归一化结构算子：

$$
S = D^{-1/2}(A+I)D^{-1/2}
$$

再采样三元稀疏随机投影矩阵：

$$
R \in \mathbb{R}^{N\times d},\quad r_{ij}\in\{-1,0,1\}
$$

并通过多阶传播得到锚点：

$$
Y_{anchor}=\sum_{k=0}^{q}\alpha_k S^kR
$$

其中 `α_k` 可由 `structure_weights` 指定；若未指定则按 `1/(k+1)` 归一化自动生成。

### 3.2 阶段 B：显式目标优化

在锚点基础上，最小化：

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

代码中的梯度实现为：

- `residual = Y - S@Y`
- `smooth_grad = residual - S^T@residual`
- `anchor_grad = Y - Y_anchor`
- `grad = 2 * (smooth_grad + λ * anchor_grad)`

每轮更新后做行归一化，并按相对变化量早停。

---

## 4. 代码映射（你可以直接对照）

### 4.1 关键函数

- `src/edane.py::_prepare_adjacency`：邻接矩阵预处理（对称化、去自环、二值化）
- `src/edane.py::_normalized_adjacency`：构造 `S`
- `src/edane.py::_sample_sparse_random_matrix`：采样 `R`
- `src/edane.py::EDANE._compute_structure_embedding`：模块一主实现（锚点 + 显式优化）

### 4.2 调用链

`EDANE.fit()` 内部顺序：

1. 图与属性预处理
2. 生成 `R` 与属性投影矩阵
3. `self.struct_emb = _compute_structure_embedding()`  ← 模块一
4. 模块三融合
5. 模块四量化

---

## 5. 参数说明（模块一相关）

以下参数在 `EDANE.__init__` 中可设：

| 参数 | 默认值 | 作用 | 调参建议 |
|---|---:|---|---|
| `dim` | 32 | 嵌入维度 `d` | 32/64 常用；维度越大质量潜力更高但更慢 |
| `order` | 2 | 传播阶数 `q` | 稀疏图常用 1~3，过大易过平滑 |
| `projection_density` | 0.1 | 随机投影非零密度 | 图很大时可降到 0.05 提速 |
| `structure_weights` | `None` | 多阶传播权重 `α_k` | 未设时默认递减权重 |
| `init_iterations` | 24 | 目标优化最大迭代轮数 `T` | 10~40 常见 |
| `init_step_size` | 0.35 | 梯度步长 `η` | 发散就降（如 0.2），太慢就升（如 0.5） |
| `init_reg` | 0.2 | 锚点正则 `λ` | 大图噪声高时可升高（0.3~0.5） |
| `init_tol` | 1e-4 | 早停阈值 | 对速度敏感可放宽到 1e-3 |

---

## 6. 复杂度与内存

### 6.1 时间复杂度

- 锚点阶段：`O(q · |E| · d)`
- 迭代优化：`O(T · |E| · d)`（每轮核心是 `S @ Y` 与 `S^T @ residual`）

总体近似：

$$
O((q+T)\cdot |E|\cdot d)
$$

### 6.2 空间复杂度

- `S` 采用 CSR：`O(|E|)`
- `R`、`Y`、`Y_anchor`：`O(N·d)`

不构造 `N×N` 稠密矩阵。

---

## 7. 使用方式

### 7.1 直接 API（推荐给算法实验）

```python
import numpy as np
from scipy import sparse
from edane import EDANE

adj = sparse.random(1000, 1000, density=0.002, format="csr")
adj = adj.maximum(adj.T)
adj.setdiag(0)
adj.eliminate_zeros()

attrs = np.random.randn(1000, 64)

model = EDANE(
    dim=64,
    order=2,
    projection_density=0.08,
    init_iterations=30,
    init_step_size=0.30,
    init_reg=0.25,
    init_tol=1e-4,
)
model.fit(adj, attrs)
emb = model.get_embedding(dequantize=False)
print(emb.shape)  # (1000, 64)
```

### 7.2 流水线模式（当前默认参数）

`src/edane_full_pipeline.py` 会通过命令行参数构造 `EDANE(...)`。若未显式透传模块一专属参数，则它们仍使用 `EDANE.__init__` 的默认值；但像 `dim`、`order`、`projection_density`、`learning_rate` 这类通用参数会由流水线命令行默认值覆盖。

```bash
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 3
```

> 若后续希望在命令行暴露模块一参数，可在 `build_parser()` 增加 `--init-iterations` 等参数并透传到 `EDANE(...)`。

---

## 8. 诊断与调参指南

### 8.1 初始化变慢明显

优先调整：

1. 减小 `init_iterations`
2. 减小 `dim`
3. 减小 `projection_density`

### 8.2 指标波动大 / 嵌入不稳定

优先调整：

1. 降低 `init_step_size`
2. 提高 `init_reg`
3. 保留 `order` 在 1~2，避免过高

### 8.3 过平滑（节点向量太相似）

优先调整：

1. 降低 `order`
2. 降低 `init_iterations`
3. 自定义 `structure_weights`，减少高阶权重

---

## 9. 与论文/答辩表述的建议对齐

建议你在论文里这样描述模块一（可直接复用）：

1. 先用稀疏随机投影构造低维结构锚点
2. 再通过显式目标函数优化提升结构一致性
3. 目标函数包含“结构平滑项 + 锚点保持项”
4. 全过程仅依赖稀疏矩阵乘法，复杂度近似线性

避免再写成“两条路线混合但未说明主路线”的表达。

---

## 10. 当前边界（实话实说）

模块一已经具备：

- 明确目标函数
- 明确迭代更新规则
- 明确收敛准则（早停）

但它仍是“工程化近似优化”，不是严格理论收敛证明版本。对于本科毕设与工程原型，这个定位是合理的。

---

## 11. 最小复现清单

如果你想快速确认模块一工作正常，跑这三步即可：

1. `python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 3`
2. 查看输出目录 `summary.json` 中 `initialization_latency_ms`
3. 查看 `metrics_per_snapshot.csv` 中 `snapshot=0` 的初始指标

这能验证模块一初始化既完成计算，也能被后续模块成功消费。
