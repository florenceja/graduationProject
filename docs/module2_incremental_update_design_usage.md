# 模块二（动态增量更新）设计与使用文档

> 适用代码：`src/edane.py`
>
> 适用类/方法：`EDANE.apply_updates()`

---

## 1. 目标与定位

模块二负责在动态图变化到来时，避免全量重训练，仅对受影响节点进行局部更新。

本实现按“文档严格版”对齐以下要求：

1. 变化检测支持四类事件：`ΔV+ / ΔV- / ΔE+ / ΔE-`
2. 局部拓扑分析固定为 `k=1`（一跳邻域）
3. 增量更新公式采用：

$$
y_i' = y_i + \alpha \cdot f(\Delta A_i y_{N(i)})
$$

其中 `f` 采用线性加权聚合（mean）。

---

## 2. 输入输出定义

## 2.1 输入（`apply_updates`）

```python
apply_updates(
    node_additions: Optional[Dict[int, np.ndarray]] = None,
    node_removals: Optional[Iterable[int]] = None,
    edge_additions: Optional[Iterable[Tuple[int, int]]] = None,
    edge_removals: Optional[Iterable[Tuple[int, int]]] = None,
    attr_updates: Optional[Dict[int, np.ndarray]] = None,
) -> List[int]
```

参数说明：

- `node_additions`：新增节点索引 -> 原始属性向量
- `node_removals`：删除节点索引列表（当前实现为软删除）
- `edge_additions`：新增边列表
- `edge_removals`：删除边列表
- `attr_updates`：节点属性更新（原始值，不是标准化值）

## 2.2 输出

- 返回 `affected_nodes`：本轮受影响节点（变化节点 + 一跳邻居）索引列表

---

## 3. 与模块一的依赖关系

模块二依赖模块一在 `fit()` 后保存的状态：

- `self.adj`：当前图
- `self.attrs`：标准化后的属性矩阵
- `self.struct_emb`：结构嵌入
- `self.attr_emb`：属性嵌入
- `self.embedding_`：最终融合嵌入
- `self.random_projection`：随机投影基向量
- `self.attr_projection`：属性映射矩阵
- `self.feature_mean_ / self.feature_std_`：属性标准化统计量

如果这些状态不存在，`apply_updates()` 会抛错，要求先调用 `fit()`。

---

## 4. 核心流程（严格版）

模块二按以下顺序执行：

1. **变化检测（四类）**
2. **构建变化前一跳邻域**
3. **落地图更新与属性更新**
4. **构建变化后一跳邻域，形成受影响集合**
5. **按文档公式执行局部结构更新**
6. **重算变化节点属性投影**
7. **仅对受影响节点做融合更新**
8. **刷新量化副本**

---

## 5. 四类变化检测（`ΔV+ / ΔV- / ΔE+ / ΔE-`）

## 5.1 节点增加（`ΔV+`）

当前实现约束：

- 新增索引必须是“从当前节点总数开始的连续索引”
  - 例如当前有 `N` 个节点，新增 2 个节点时索引必须是 `N, N+1`

处理动作：

1. 扩展邻接矩阵维度
2. 对新增节点属性按 `fit()` 统计量标准化
3. 扩展随机投影矩阵 `R`
4. 初始化新增节点的结构向量、属性向量与融合向量

## 5.2 节点删除（`ΔV-`）

当前实现采用**软删除**（索引不回收）：

1. 清空该节点相关边（行列置零）
2. 将该节点属性与嵌入向量置零

这样可以保持索引稳定，避免全图重排。

## 5.3 边增加（`ΔE+`）

- 对称写入 `(u,v)` 和 `(v,u)`
- 忽略自环

## 5.4 边删除（`ΔE-`）

- 对称清除 `(u,v)` 和 `(v,u)`
- 忽略自环

---

## 6. 局部拓扑分析（`k=1`）

文档严格要求仅考虑一跳邻居。

实现中：

- `before_by_node[i]`：变化前一跳邻居
- `after_by_node[i]`：变化后一跳邻居
- 受影响节点：

\[
U = T \cup N_{before}(T) \cup N_{after}(T)
\]

其中 `T` 是直接发生变化的节点集合。

---

## 7. 增量更新公式（文档落地）

每个受影响节点 `i` 的结构更新：

$$
y_i' = y_i + \alpha \cdot f(\Delta A_i y_{N(i)})
$$

代码中的离散化实现：

1. 先求邻居变化集合：
   - `add_neighbors = N_after(i) - N_before(i)`
   - `rem_neighbors = N_before(i) - N_after(i)`
2. 构造扰动项：

$$
\Delta A_i y_{N(i)} \approx \sum_{j\in add} y_j - \sum_{j\in rem} y_j
$$

3. 线性聚合（mean）：

$$
f(\cdot)=\frac{1}{\max(|N_{after}(i)|,1)}(\cdot)
$$

4. 用 `learning_rate` 作为 `\alpha`：

$$
y_i' = y_i + \text{learning\_rate} \cdot f(\cdot)
$$

5. 对全体结构向量做行归一化。

---

## 8. 属性更新规则

`attr_updates` 与 `node_additions` 的属性向量都按 `fit()` 保存的 `mean/std` 标准化后写入。

原因：避免“初始特征与更新特征不在同一尺度”的问题。

属性更新后：

1. 只重算变化节点对应的属性投影 `attr_emb[idx]`
2. 对属性嵌入做行归一化

---

## 9. 融合与输出更新

结构与属性更新后：

1. 仅对受影响节点 `U` 调用模块三融合
2. 回写全局 `embedding_`
3. 行归一化
4. 刷新模块四量化副本

返回 `affected_nodes`。

---

## 10. 边界与工程约束

为保持“文档严格版 + 可运行”，当前边界如下：

1. `ΔV-` 为软删除，不做索引压缩重排
2. `ΔV+` 要求新增索引连续（便于线性扩容）
3. 只实现 `k=1`，不支持更高阶邻域扩散
4. 未直接实现谱分解形式的矩阵扰动闭式公式，采用文档主公式的局部离散近似

---

## 11. 复杂度分析

设：

- `|T|`：直接变化节点数
- `|U|`：受影响节点数（含一跳扩展）
- `d`：嵌入维度

主要开销：

1. 邻域差分与扰动计算：`O((|ΔE| + |U|) · d)`
2. 受影响节点融合：`O(|U| · d)`

在稀疏变化场景下，一般 `|U| << N`，可显著优于全量重训练。

---

## 12. 使用示例

## 12.1 仅边变化 + 属性变化

```python
affected = model.apply_updates(
    edge_additions=[(10, 25), (31, 77)],
    edge_removals=[(8, 19)],
    attr_updates={
        25: new_attr_vec_25,
        77: new_attr_vec_77,
    },
)
```

## 12.2 包含节点新增/删除

```python
N = model.get_embedding(dequantize=False).shape[0]

affected = model.apply_updates(
    node_additions={
        N: attr_for_new_node_0,
        N + 1: attr_for_new_node_1,
    },
    node_removals=[12],
    edge_additions=[(N, 5), (N + 1, 20)],
)
```

注意：新增索引必须连续（`N, N+1, ...`）。

---

## 13. 调参与排错建议

### 13.1 更新幅度过小（“几乎没变化”）

- 增大 `learning_rate`
- 检查是否确实有邻域变化（只改属性时结构项变化可能小）

### 13.2 更新抖动大（指标不稳）

- 降低 `learning_rate`
- 检查输入事件是否过于密集（一次性变化过大）

### 13.3 `node_additions` 报错

- 检查新增索引是否连续且从当前节点数开始

### 13.4 属性更新报维度错误

- 向量维度必须与 `features.csv` 一致
- 输入必须是有限值（不能有 NaN/Inf）

---

## 14. 与论文/答辩表述建议

可用下面这段描述模块二：

1. 本文按 `ΔV+ / ΔV- / ΔE+ / ΔE-` 统一检测动态图事件
2. 采用 `k=1` 局部拓扑分析界定更新范围
3. 按 `y_i' = y_i + α·f(ΔA_i y_{N(i)})` 对受影响节点做增量更新
4. 仅重算局部节点，避免每个时间步全量重训练

这样能和你的文档保持一致，同时代码层也能自证可运行。

---

## 15. 最小复现清单

1. 先跑初始化：

```bash
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 3
```

2. 在自定义脚本中构造事件并调用 `apply_updates()`
3. 检查返回 `affected_nodes` 与更新后指标变化

若流程能跑通并有局部节点更新，即可确认模块二正常工作。
