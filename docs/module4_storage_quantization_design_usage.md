# 模块四（轻量化存储与量化压缩）设计与使用文档

> 适用代码：`src/edane.py`、`src/edane_full_pipeline.py`
>
> 适用类/方法：`QuantizedEmbedding`、`BinaryEmbedding`、`EDANE._refresh_quantized_embedding()`、`EDANE.get_embedding()`、`EDANE.get_binary_embedding()`

---

## 1. 目标与定位

模块四负责将最终嵌入从高精度浮点表示压缩为更轻量的存储形式，同时保留可恢复、可统计、可评估的能力。

本实现对应文档中的两条主线：

1. **量化存储**：将浮点向量压缩为低精度表示
2. **轻量化计算**：为后续相似度计算、存储统计与对比实验提供压缩副本

在当前版本中，模块四支持：

- `int8` 量化（主实现）
- `binary` 二值副本（可选）
- 压缩比统计
- 重构误差统计

---

## 2. 输入输出定义

### 2.1 输入

- 最终浮点嵌入 `Z ∈ R^{N×d}`

### 2.2 输出

- `QuantizedEmbedding`：逐维 `int8` 量化副本
- `BinaryEmbedding`：可选二值化副本
- `quantization_error_`
- `binary_error_`
- `quantization_compression_ratio_`
- `binary_compression_ratio_`

---

## 3. 模块四的整体结构

模块四在当前实现中由三个部分组成：

1. **Int8 量化子模块**（正式主实现）
2. **Binary 二值子模块**（可选扩展）
3. **压缩效果统计子模块**（压缩比 + 重构误差）

---

## 4. Int8 量化设计

### 4.1 量化目标

将浮点向量逐维压缩到 `[-127, 127]` 的 `int8` 空间，尽量减少信息损失。

### 4.2 量化公式

当前实现采用**逐维对称量化**。

对第 `j` 维，先计算缩放系数：

\[
s_j = \frac{\max_i |z_{ij}|}{127}
\]

然后量化：

\[
q_{ij} = \text{clip}\left(\text{round}\left(\frac{z_{ij}}{s_j}\right), -127, 127\right)
\]

恢复时：

\[
\hat z_{ij} = q_{ij} \cdot s_j
\]

### 4.3 数据结构

对应类：`QuantizedEmbedding`

包含：

- `values`：`int8` 量化矩阵
- `scale`：每维缩放系数

并提供：

- `dequantize()`：恢复为近似浮点矩阵

---

## 5. Binary 二值化设计

### 5.1 设计目的

Binary 副本用于极限压缩的实验对比。

### 5.2 二值化规则

当前实现采用符号二值化：

\[
b_{ij} =
\begin{cases}
1, & z_{ij} \ge 0 \\
-1, & z_{ij} < 0
\end{cases}
\]

### 5.3 数据结构

对应类：`BinaryEmbedding`

包含：

- `values`：当前实现用 `float32` 的 `±1` 表示

并提供：

- `dequantize()`：返回 `±1` 的浮点形式

> 注意：这还不是位级 bit-pack 版本，而是工程上更稳定、便于调试的二值副本实现。

---

## 6. 压缩误差与压缩比统计

模块四不仅生成压缩副本，还会自动统计压缩效果。

### 6.1 Int8 重构误差

\[
E_{int8} = \frac{\|Z - \hat Z_{int8}\|_F}{\|Z\|_F}
\]

在代码中记为：

- `quantization_error_`

### 6.2 Binary 重构误差

\[
E_{bin} = \frac{\|Z - \hat Z_{bin}\|_F}{\|Z\|_F}
\]

在代码中记为：

- `binary_error_`

### 6.3 Int8 压缩比

\[
R_{int8} = \frac{\text{float bytes}}{\text{int8 values bytes} + \text{scale bytes}}
\]

在代码中记为：

- `quantization_compression_ratio_`

### 6.4 Binary 压缩比

\[
R_{bin} = \frac{\text{float bytes}}{\text{binary values bytes}}
\]

在代码中记为：

- `binary_compression_ratio_`

---

## 7. 在代码中的执行时机

模块四不是单独手动调用，而是在嵌入更新后自动刷新。

### 7.1 在 `fit()` 后

当模块一、二、三完成初始嵌入后：

1. 调用 `_refresh_quantized_embedding()`
2. 刷新 int8 副本
3. 根据配置刷新 binary 副本
4. 计算误差与压缩比

### 7.2 在 `apply_updates()` 后

当局部节点增量更新完成后：

1. 更新 `embedding_`
2. 再次调用 `_refresh_quantized_embedding()`
3. 让压缩副本与最新嵌入保持一致

---

## 8. 代码映射

### 8.1 `src/edane.py`

关键位置：

- `QuantizedEmbedding`
- `BinaryEmbedding`
- `EDANE._refresh_quantized_embedding()`
- `EDANE.get_embedding()`
- `EDANE.get_binary_embedding()`

### 8.2 `src/edane_full_pipeline.py`

关键位置：

- 导出 `final_embedding.npy`
- 导出 `final_embedding_int8.npy`
- 导出 `final_embedding_scale.npy`
- 若开启 `--binary-quantize`，则导出 `final_embedding_binary.npy`
- 在 `summary.json` 中写入：
  - `quantization_compression_ratio`
  - `quantization_error`
  - `binary_compression_ratio`
  - `binary_error`

---

## 9. 使用方式

### 9.1 默认 Int8 量化

```python
model = EDANE(quantize=True)
model.fit(adj, attrs)
emb = model.get_embedding(dequantize=True)
```

说明：

- `quantize=True` 时，会自动维护 `int8` 副本
- `get_embedding(dequantize=True)` 返回反量化后的近似浮点向量

### 9.2 获取原始浮点嵌入

```python
float_emb = model.get_embedding(dequantize=False)
```

### 9.3 开启 Binary 副本

```python
model = EDANE(quantize=True, binary_quantize=True)
model.fit(adj, attrs)
binary_emb = model.get_binary_embedding(dequantize=False)
```

### 9.4 获取 Binary 恢复结果

```python
binary_float = model.get_binary_embedding(dequantize=True)
```

---

## 10. 输出文件说明

在 `edane_full_pipeline.py` 跑完后，输出目录中可能包含：

| 文件 | 说明 |
|---|---|
| `final_embedding.npy` | 最终浮点嵌入 |
| `final_embedding_int8.npy` | Int8 量化值 |
| `final_embedding_scale.npy` | Int8 缩放系数 |
| `final_embedding_binary.npy` | Binary 副本（若开启 `--binary-quantize`） |
| `summary.json` | 包含压缩比和误差统计 |

---

## 11. 当前已验证的效果示例

当前在 `reddit_sample` 上的流水线验证中，已经观测到：

- `quantization_compression_ratio ≈ 7.99`
- `quantization_error ≈ 0.0098`

说明当前 Int8 量化能明显降低存储，同时保持较小的重构误差。

---

## 12. 调参与排错建议

### 12.1 压缩比低于预期

可能原因：

- 维度较低时，`scale` 的额外存储占比相对更明显
- 未开启量化，结果会退回 1.0

### 12.2 量化误差偏大

可能原因：

- 嵌入分布跨度大
- 某些维度极端值过大，压缩了有效动态范围

可考虑：

- 在模块三后增强归一化稳定性
- 后续扩展为分块量化或更细粒度缩放

### 12.3 Binary 误差过大

这是预期现象，因为 Binary 压缩损失远大于 Int8。

Binary 更适合作为“极限压缩实验对照”，而非默认主方案。

---

## 13. 当前边界（实话实说）

模块四现在已经比早期原型完整得多，但仍有边界：

1. Binary 当前不是 bit-pack 到位级压缩，而是 `±1 float32` 副本
2. 当前压缩误差只统计向量重构误差，还没有单独统计“压缩后再评估任务性能”的系统实验
3. 压缩策略目前是逐维对称量化，尚未扩展到更高级量化方案

因此它已经满足本科毕设与工程原型需要，但如果你以后要写“更强存储模块”，还能继续扩展。

---

## 14. 与论文/答辩表述建议

你可以这样描述模块四：

1. 对最终嵌入进行逐维 Int8 对称量化
2. 可选构造 Binary 副本作为极限压缩对照
3. 在系统中同时记录压缩比与重构误差
4. 图结构仍采用 CSR 稀疏格式存储，整体降低存储开销

如果老师追问“是不是已经做到位级最优压缩”，可以明确回答：

> 当前实现是面向实验验证的轻量化存储模块，重点在于可运行、可恢复、可统计，而不是工业级 bit-pack 极限实现。

---

## 15. 最小复现清单

运行：

```bash
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 3 --quantize
```

然后检查：

1. 输出目录是否有 `final_embedding_int8.npy`
2. `summary.json` 是否包含：
   - `quantization_compression_ratio`
   - `quantization_error`
   - `binary_compression_ratio`
   - `binary_error`

若这些都存在，说明模块四已被完整调用。
