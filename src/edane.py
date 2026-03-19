"""EDANE 原型算法实现（含双曲融合与增量更新）。

本文件聚焦“算法核心”，用于给上层实验流水线调用：
1) fit(): 初始建模
2) apply_updates(): 动态增量更新
3) get_embedding(): 读取当前嵌入（浮点或反量化）

实现目标：
- 代码尽量轻依赖，仅使用 NumPy + SciPy（稀疏矩阵），方便课程/毕设环境快速复现。
- 逻辑与报告中的四大模块保持一致：随机投影初始化、增量更新、
  结构-属性融合、量化存储。
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse


ArrayLike = np.ndarray
SparseMatrix = sparse.csr_matrix
AdjLike = Union[np.ndarray, SparseMatrix]


def _sigmoid(x: ArrayLike) -> ArrayLike:
    """数值稳定版 Sigmoid。"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _row_normalize(x: ArrayLike, eps: float = 1e-12) -> ArrayLike:
    """按行做 L2 归一化，使每个节点向量长度接近 1。"""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _standardize_features(x: ArrayLike, eps: float = 1e-12) -> ArrayLike:
    """特征标准化（零均值、单位方差），减少量纲差异带来的不稳定。"""
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, eps)


def _compute_feature_stats(x: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """计算特征标准化统计量。"""
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return mean, std


def _apply_feature_standardization(
    x: ArrayLike,
    mean: ArrayLike,
    std: ArrayLike,
    eps: float = 1e-12,
) -> ArrayLike:
    """使用固定统计量标准化特征。"""
    return (x - mean) / np.maximum(std, eps)


def _prepare_adjacency(adj: AdjLike) -> SparseMatrix:
    """邻接矩阵预处理：对称化、去自环、二值化（返回 CSR 稀疏矩阵）。"""
    work = sparse.csr_matrix(adj, dtype=float)
    work = work.maximum(work.T).tocsr()
    work.setdiag(0.0)
    work.eliminate_zeros()
    if work.nnz > 0:
        work.data[:] = 1.0
    return work


def _normalized_adjacency(adj: SparseMatrix, add_self_loops: bool = True) -> SparseMatrix:
    """构造对称归一化邻接矩阵 D^-1/2 (A+I) D^-1/2（CSR）。"""
    if add_self_loops:
        work = adj + sparse.eye(adj.shape[0], dtype=float, format="csr")
    else:
        work = adj.copy()
    degree = np.asarray(work.sum(axis=1)).reshape(-1)
    inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1.0))
    d_inv = sparse.diags(inv_sqrt_degree, format="csr")
    return (d_inv @ work @ d_inv).tocsr()


def _sample_sparse_random_matrix(
    rows: int,
    cols: int,
    density: float,
    rng: np.random.Generator,
) -> ArrayLike:
    """采样稀疏随机投影矩阵（-1/0/1 三元分布）。

    density 控制非零比例；scale 用于保持不同密度下方差可比。
    """
    density = float(np.clip(density, 1e-3, 1.0))
    probs = [density / 2.0, 1.0 - density, density / 2.0]
    values = rng.choice([-1.0, 0.0, 1.0], size=(rows, cols), p=probs)
    scale = 1.0 / math.sqrt(density * cols)
    return values * scale


def _safe_norm(x: ArrayLike, axis: int = 1, keepdims: bool = True) -> ArrayLike:
    """安全范数，避免除零。"""
    return np.maximum(np.linalg.norm(x, axis=axis, keepdims=keepdims), 1e-12)


def _exp_map_zero(x: ArrayLike) -> ArrayLike:
    """欧氏 -> 双曲（Poincare 球）指数映射（原点处）。"""
    norm = _safe_norm(x)
    return np.tanh(norm) * x / norm


def _log_map_zero(y: ArrayLike) -> ArrayLike:
    """双曲 -> 欧氏 对数映射（原点处）。"""
    norm = np.clip(_safe_norm(y), 1e-12, 1.0 - 1e-6)
    return np.arctanh(norm) * y / norm


def _mobius_add(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """Poincare 球内的莫比乌斯加法。"""
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True)
    xy = np.sum(x * y, axis=1, keepdims=True)
    numerator = (1.0 + 2.0 * xy + y2) * x + (1.0 - x2) * y
    denominator = 1.0 + 2.0 * xy + x2 * y2
    result = numerator / np.maximum(denominator, 1e-12)
    norm = _safe_norm(result)
    clipped = np.minimum(norm, 1.0 - 1e-5)
    return result * (clipped / norm)


def _mobius_scalar_mul(weight: ArrayLike, x: ArrayLike) -> ArrayLike:
    """Poincare 球内的标量乘。"""
    norm = _safe_norm(x)
    scaled = np.tanh(weight * np.arctanh(np.clip(norm, 1e-12, 1.0 - 1e-6))) * x / norm
    new_norm = _safe_norm(scaled)
    clipped = np.minimum(new_norm, 1.0 - 1e-5)
    return scaled * (clipped / new_norm)


@dataclass
class QuantizedEmbedding:
    """量化后嵌入容器。

    values: int8 张量
    scale: 每一维的缩放系数（per-dimension scale）
    """

    values: ArrayLike
    scale: ArrayLike

    def dequantize(self) -> ArrayLike:
        """将 int8 量化值乘以缩放系数，恢复为近似浮点嵌入。"""
        return self.values.astype(np.float64) * self.scale


@dataclass
class BinaryEmbedding:
    """二值化嵌入容器。"""

    values: ArrayLike

    def dequantize(self) -> ArrayLike:
        """将布尔/符号表示恢复为 ±1 浮点形式。"""
        return self.values.astype(np.float64)


class EDANE:
    """EDANE 算法主类。

    关键参数：
    - dim: 嵌入维度
    - order: 结构传播阶数 q
    - projection_density: 随机投影矩阵非零密度
    - learning_rate: 增量更新率 eta
    - quantize: 是否维护 int8 量化副本
    """

    def __init__(
        self,
        dim: int = 32,
        order: int = 2,
        projection_density: float = 0.1,
        learning_rate: float = 0.5,
        quantize: bool = True,
        binary_quantize: bool = False,
        random_state: int = 42,
        structure_weights: Optional[Sequence[float]] = None,
        init_iterations: int = 24,
        init_step_size: float = 0.35,
        init_reg: float = 0.2,
        init_tol: float = 1e-4,
        fusion_train_steps: int = 40,
        fusion_lr: float = 0.1,
        fusion_weight_decay: float = 1e-4,
        use_attr_fusion: bool = True,
        use_hyperbolic_fusion: bool = True,
    ) -> None:
        self.dim = dim
        self.order = order
        self.projection_density = projection_density
        self.learning_rate = learning_rate
        self.quantize = quantize
        self.binary_quantize = binary_quantize
        self.random_state = random_state
        self.structure_weights = structure_weights
        self.init_iterations = max(1, int(init_iterations))
        self.init_step_size = float(init_step_size)
        self.init_reg = float(init_reg)
        self.init_tol = float(init_tol)
        self.fusion_train_steps = max(0, int(fusion_train_steps))
        self.fusion_lr = float(fusion_lr)
        self.fusion_weight_decay = float(fusion_weight_decay)
        self.use_attr_fusion = bool(use_attr_fusion)
        self.use_hyperbolic_fusion = bool(use_hyperbolic_fusion)
        self.rng = np.random.default_rng(random_state)

        self.adj: Optional[SparseMatrix] = None
        self.attrs: Optional[ArrayLike] = None
        self.norm_adj: Optional[SparseMatrix] = None
        self.random_projection: Optional[ArrayLike] = None
        self.attr_projection: Optional[ArrayLike] = None
        self.struct_emb: Optional[ArrayLike] = None
        self.attr_emb: Optional[ArrayLike] = None
        self.embedding_: Optional[ArrayLike] = None
        self.quantized_embedding_: Optional[QuantizedEmbedding] = None
        self.binary_embedding_: Optional[BinaryEmbedding] = None
        self.feature_mean_: Optional[ArrayLike] = None
        self.feature_std_: Optional[ArrayLike] = None
        self.fusion_weight_: Optional[ArrayLike] = None
        self.fusion_bias_: float = 0.0
        self.quantization_error_: float = 0.0
        self.binary_error_: float = 0.0
        self.quantization_compression_ratio_: float = 1.0
        self.binary_compression_ratio_: float = 1.0

    def fit(self, adj: AdjLike, attrs: ArrayLike) -> "EDANE":
        """在初始快照上训练并生成初始嵌入。

        步骤：
        1) 预处理图与特征
        2) 构造随机投影矩阵
        3) 计算结构嵌入与属性嵌入
        4) 双曲空间融合
        5) 可选量化
        """
        self.adj = _prepare_adjacency(adj)
        raw_attrs = np.asarray(attrs, dtype=np.float64)
        if raw_attrs.ndim != 2:
            raise ValueError("attrs must be a 2D array.")
        if not np.all(np.isfinite(raw_attrs)):
            raise ValueError("attrs must contain only finite values.")
        if self.adj.shape[0] != self.adj.shape[1]:
            raise ValueError("adjacency matrix must be square.")
        if self.adj.shape[0] != raw_attrs.shape[0]:
            raise ValueError("adjacency rows must match attrs rows.")

        self.feature_mean_, self.feature_std_ = _compute_feature_stats(raw_attrs)
        self.attrs = _apply_feature_standardization(raw_attrs, self.feature_mean_, self.feature_std_)
        self.norm_adj = _normalized_adjacency(self.adj)

        num_nodes, num_features = self.attrs.shape
        self.random_projection = _sample_sparse_random_matrix(
            num_nodes,
            self.dim,
            self.projection_density,
            self.rng,
        )
        self.attr_projection = self.rng.normal(
            loc=0.0,
            scale=1.0 / math.sqrt(max(num_features, 1)),
            size=(num_features, self.dim),
        )

        self.struct_emb = self._compute_structure_embedding()
        self.attr_emb = self._compute_attribute_embedding(self.attrs)
        self._init_fusion_gate_parameters()
        if self.use_attr_fusion:
            self._train_fusion_gate(self.struct_emb, self.attr_emb)
            self.embedding_ = self._fuse_embeddings(self.struct_emb, self.attr_emb)
        else:
            # 消融：w/o-Attr，仅保留结构表征。
            self.embedding_ = _row_normalize(self.struct_emb.copy())
        self._refresh_quantized_embedding()
        return self

    def _get_structure_weights(self) -> List[float]:
        """读取或自动构造结构传播权重 alpha_k。"""
        if self.structure_weights is not None:
            if len(self.structure_weights) != self.order + 1:
                raise ValueError("structure_weights length must equal order + 1.")
            weights = np.asarray(self.structure_weights, dtype=np.float64)
            weights = weights / weights.sum()
            return weights.tolist()
        weights = np.array([1.0 / (k + 1) for k in range(self.order + 1)], dtype=np.float64)
        weights /= weights.sum()
        return weights.tolist()

    def _compute_structure_embedding(self) -> ArrayLike:
        """计算结构嵌入（模块一：稀疏投影 + 显式迭代优化）。

        1) 先用 S^k R 构造锚点（稀疏随机投影初始化）；
        2) 再最小化目标函数：
           J(Y) = ||Y - S Y||_F^2 + lambda * ||Y - Y_anchor||_F^2
           使用梯度下降迭代更新。
        """
        assert self.norm_adj is not None
        assert self.random_projection is not None

        s_mat = self.norm_adj
        s_t_mat = s_mat.transpose().tocsr()
        weights = self._get_structure_weights()
        propagated = self.random_projection.copy()
        anchor = weights[0] * propagated
        for k in range(1, self.order + 1):
            propagated = s_mat @ propagated
            anchor += weights[k] * propagated

        y = _row_normalize(anchor)
        anchor = _row_normalize(anchor)

        for _ in range(self.init_iterations):
            sy = s_mat @ y
            residual = y - sy

            smooth_grad = residual - (s_t_mat @ residual)
            anchor_grad = y - anchor
            grad = 2.0 * (smooth_grad + self.init_reg * anchor_grad)

            y_next = y - self.init_step_size * grad
            y_next = _row_normalize(y_next)

            delta = np.linalg.norm(y_next - y) / max(np.linalg.norm(y), 1e-12)
            y = y_next
            if delta < self.init_tol:
                break

        return y

    def _compute_attribute_embedding(self, attrs: ArrayLike) -> ArrayLike:
        """计算属性嵌入 H_x = X W_x。"""
        assert self.attr_projection is not None
        attr_emb = attrs @ self.attr_projection
        return _row_normalize(attr_emb)

    def _init_fusion_gate_parameters(self) -> None:
        """初始化融合门控参数：alpha = sigmoid(W[p_s||p_a] + b)。"""
        feature_dim = 2 * self.dim
        self.fusion_weight_ = self.rng.normal(loc=0.0, scale=1.0 / math.sqrt(max(feature_dim, 1)), size=(feature_dim, 1))
        self.fusion_bias_ = 0.0

    def _build_fusion_gate_features(self, struct_emb: ArrayLike, attr_emb: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """构造双曲融合门控特征。"""
        struct_h = _exp_map_zero(0.5 * struct_emb)
        attr_h = _exp_map_zero(0.5 * attr_emb)
        features = np.concatenate([struct_h, attr_h], axis=1)
        return features, struct_h, attr_h

    def _compute_fusion_gate(self, features: ArrayLike) -> ArrayLike:
        """计算门控系数 alpha。"""
        if self.fusion_weight_ is None:
            self._init_fusion_gate_parameters()
        assert self.fusion_weight_ is not None
        logits = features @ self.fusion_weight_ + self.fusion_bias_
        return _sigmoid(logits)

    def _train_fusion_gate(self, struct_emb: ArrayLike, attr_emb: ArrayLike) -> None:
        """轻量训练融合门控参数（无监督伪目标）。"""
        if self.fusion_train_steps <= 0:
            return
        if self.fusion_weight_ is None:
            self._init_fusion_gate_parameters()
        assert self.fusion_weight_ is not None

        features, _, _ = self._build_fusion_gate_features(struct_emb, attr_emb)
        target = _sigmoid(np.sum(struct_emb * attr_emb, axis=1, keepdims=True) / math.sqrt(max(self.dim, 1)))
        n = max(features.shape[0], 1)

        for _ in range(self.fusion_train_steps):
            logits = features @ self.fusion_weight_ + self.fusion_bias_
            pred = _sigmoid(logits)
            diff = pred - target
            grad_w = (features.T @ diff) / n + self.fusion_weight_decay * self.fusion_weight_
            grad_b = float(np.mean(diff))

            self.fusion_weight_ = self.fusion_weight_ - self.fusion_lr * grad_w
            self.fusion_bias_ = self.fusion_bias_ - self.fusion_lr * grad_b

    def _fuse_embeddings(self, struct_emb: ArrayLike, attr_emb: ArrayLike) -> ArrayLike:
        """双曲门控融合结构与属性嵌入。

        严格版：alpha = sigmoid(W[p_struct | p_attr] + b)，
        再在双曲空间做 Möbius 加权融合并映射回欧氏空间。
        """
        if not self.use_attr_fusion:
            return _row_normalize(struct_emb)

        features, struct_h, attr_h = self._build_fusion_gate_features(struct_emb, attr_emb)
        gate = self._compute_fusion_gate(features)
        if self.use_hyperbolic_fusion:
            left = _mobius_scalar_mul(gate, struct_h)
            right = _mobius_scalar_mul(1.0 - gate, attr_h)
            fused_h = _mobius_add(left, right)
            fused = _log_map_zero(fused_h)
            return _row_normalize(fused)

        # 消融：w/o-Hyperbolic，保持门控但改为欧氏空间融合。
        fused_euclidean = gate * struct_emb + (1.0 - gate) * attr_emb
        return _row_normalize(fused_euclidean)

    def _refresh_quantized_embedding(self) -> None:
        """更新模块四压缩副本：int8 量化 + 可选二值化。"""
        if self.embedding_ is None:
            return
        float_bytes = max(self.embedding_.nbytes, 1)

        if not self.quantize:
            self.quantized_embedding_ = None
            self.quantization_error_ = 0.0
            self.quantization_compression_ratio_ = 1.0
        else:
            max_abs = np.max(np.abs(self.embedding_), axis=0, keepdims=True)
            scale = np.maximum(max_abs / 127.0, 1e-8)
            values = np.clip(np.round(self.embedding_ / scale), -127, 127).astype(np.int8)
            self.quantized_embedding_ = QuantizedEmbedding(values=values, scale=scale)
            restored = self.quantized_embedding_.dequantize()
            self.quantization_error_ = float(
                np.linalg.norm(self.embedding_ - restored) / max(np.linalg.norm(self.embedding_), 1e-12)
            )
            quantized_bytes = self.quantized_embedding_.values.nbytes + self.quantized_embedding_.scale.nbytes
            self.quantization_compression_ratio_ = float(float_bytes / max(quantized_bytes, 1))

        if not self.binary_quantize:
            self.binary_embedding_ = None
            self.binary_error_ = 0.0
            self.binary_compression_ratio_ = 1.0
        else:
            binary_values = np.where(self.embedding_ >= 0.0, 1.0, -1.0).astype(np.float32)
            self.binary_embedding_ = BinaryEmbedding(values=binary_values)
            restored_binary = self.binary_embedding_.dequantize()
            self.binary_error_ = float(
                np.linalg.norm(self.embedding_ - restored_binary) / max(np.linalg.norm(self.embedding_), 1e-12)
            )
            binary_bytes = self.binary_embedding_.values.nbytes
            self.binary_compression_ratio_ = float(float_bytes / max(binary_bytes, 1))

    def get_embedding(self, dequantize: bool = True) -> ArrayLike:
        """获取当前嵌入。

        dequantize=True 且存在量化副本时，返回反量化后的浮点向量。
        """
        if self.embedding_ is None:
            raise ValueError("Call fit() before get_embedding().")
        if dequantize and self.quantized_embedding_ is not None:
            return self.quantized_embedding_.dequantize()
        return self.embedding_.copy()

    def get_binary_embedding(self, dequantize: bool = True) -> ArrayLike:
        """获取二值化嵌入。"""
        if self.embedding_ is None:
            raise ValueError("Call fit() before get_binary_embedding().")
        if self.binary_embedding_ is None:
            raise ValueError("Binary quantization is not enabled.")
        if dequantize:
            return self.binary_embedding_.dequantize()
        return self.binary_embedding_.values.copy()

    def apply_updates(
        self,
        node_additions: Optional[Dict[int, ArrayLike]] = None,
        node_removals: Optional[Iterable[int]] = None,
        edge_additions: Optional[Iterable[Tuple[int, int]]] = None,
        edge_removals: Optional[Iterable[Tuple[int, int]]] = None,
        attr_updates: Optional[Dict[int, ArrayLike]] = None,
    ) -> List[int]:
        """文档严格版动态增量更新。

        变化检测支持四类操作：
        - 节点增加（ΔV+）
        - 节点删除（ΔV-，软删除）
        - 边增加（ΔE+）
        - 边删除（ΔE-）

        局部更新范围固定为 k=1（受影响节点及其一跳邻居）。
        结构更新公式采用：
            y'_i = y_i + alpha * f(DeltaA_i y_{N(i)})
        其中 f 为线性加权聚合（mean）。
        """
        if self.adj is None or self.attrs is None or self.embedding_ is None:
            raise ValueError("Call fit() before apply_updates().")
        if self.feature_mean_ is None or self.feature_std_ is None:
            raise ValueError("Call fit() before apply_updates().")

        current_adj = self.adj
        current_attrs = self.attrs
        current_struct = self.struct_emb
        current_attr = self.attr_emb
        current_emb = self.embedding_

        assert current_struct is not None
        assert current_attr is not None
        assert current_emb is not None
        assert self.random_projection is not None
        assert self.attr_projection is not None

        def _neighbors(mat: SparseMatrix, node: int) -> np.ndarray:
            start = mat.indptr[node]
            end = mat.indptr[node + 1]
            return mat.indices[start:end]

        old_adj = sparse.csr_matrix(current_adj, dtype=float)
        touched_nodes = set()

        # ΔV+：节点增加（新增索引必须连续接在末尾）
        if node_additions:
            sorted_items = sorted((int(idx), np.asarray(vec, dtype=np.float64)) for idx, vec in node_additions.items())
            base_n = current_attrs.shape[0]
            expected = list(range(base_n, base_n + len(sorted_items)))
            actual = [idx for idx, _ in sorted_items]
            if actual != expected:
                raise ValueError("node_additions must be contiguous new indices starting at current node count.")

            attr_dim = current_attrs.shape[1]
            standardized_rows: List[np.ndarray] = []
            for idx, raw_vec in sorted_items:
                if raw_vec.ndim != 1 or raw_vec.shape[0] != attr_dim:
                    raise ValueError(f"new node {idx} attr must be 1D with dim={attr_dim}.")
                if not np.all(np.isfinite(raw_vec)):
                    raise ValueError(f"new node {idx} attr must contain only finite values.")
                standardized = _apply_feature_standardization(
                    raw_vec.reshape(1, -1),
                    self.feature_mean_,
                    self.feature_std_,
                ).reshape(-1)
                standardized_rows.append(standardized)

            m = len(sorted_items)
            expanded_adj = sparse.lil_matrix((base_n + m, base_n + m), dtype=np.float64)
            old_coo = old_adj.tocoo()
            expanded_adj[old_coo.row, old_coo.col] = old_coo.data
            current_adj = sparse.csr_matrix(expanded_adj, dtype=float)

            new_attrs = np.vstack(standardized_rows)
            current_attrs = np.vstack([current_attrs, new_attrs])

            new_rp = _sample_sparse_random_matrix(m, self.dim, self.projection_density, self.rng)
            self.random_projection = np.vstack([self.random_projection, new_rp])

            init_struct = _row_normalize(new_rp)
            init_attr = _row_normalize(new_attrs @ self.attr_projection)
            init_fused = self._fuse_embeddings(init_struct, init_attr)

            current_struct = np.vstack([current_struct, init_struct])
            current_attr = np.vstack([current_attr, init_attr])
            current_emb = np.vstack([current_emb, init_fused])

            old_adj = sparse.csr_matrix(current_adj, dtype=float)
            touched_nodes.update(actual)

        # ΔV-：节点删除（软删除，不改变索引）
        for idx in node_removals or []:
            idx = int(idx)
            if idx < 0 or idx >= current_attrs.shape[0]:
                raise ValueError(f"node removal index out of range: {idx}")
            touched_nodes.add(idx)

        # ΔE+ / ΔE-
        for u, v in edge_additions or []:
            if u == v:
                continue
            if u < 0 or v < 0 or u >= current_attrs.shape[0] or v >= current_attrs.shape[0]:
                raise ValueError(f"edge addition index out of range: ({u}, {v})")
            touched_nodes.update([u, v])
        for u, v in edge_removals or []:
            if u == v:
                continue
            if u < 0 or v < 0 or u >= current_attrs.shape[0] or v >= current_attrs.shape[0]:
                raise ValueError(f"edge removal index out of range: ({u}, {v})")
            touched_nodes.update([u, v])

        # 属性变化
        if attr_updates:
            for idx, value in attr_updates.items():
                if idx < 0 or idx >= current_attrs.shape[0]:
                    raise ValueError(f"attr update node index out of range: {idx}")
                raw_value = np.asarray(value, dtype=np.float64)
                if raw_value.ndim != 1 or raw_value.shape[0] != current_attrs.shape[1]:
                    raise ValueError(
                        f"attr update for node {idx} must be 1D with dim={current_attrs.shape[1]}."
                    )
                if not np.all(np.isfinite(raw_value)):
                    raise ValueError(f"attr update for node {idx} must contain only finite values.")
                standardized = _apply_feature_standardization(
                    raw_value.reshape(1, -1),
                    self.feature_mean_,
                    self.feature_std_,
                ).reshape(-1)
                current_attrs[idx] = standardized
                touched_nodes.add(idx)

        if not touched_nodes:
            return []

        before_by_node: Dict[int, set] = {}
        for idx in touched_nodes:
            before_by_node[idx] = set(_neighbors(old_adj, idx).tolist())

        work_adj = old_adj.tolil(copy=True)

        # 节点软删除：清空该节点行列及其向量
        for idx in node_removals or []:
            idx = int(idx)
            nbrs = _neighbors(old_adj, idx)
            touched_nodes.update(nbrs.tolist())
            work_adj[idx, :] = 0.0
            work_adj[:, idx] = 0.0
            current_attrs[idx] = 0.0
            current_struct[idx] = 0.0
            current_attr[idx] = 0.0
            current_emb[idx] = 0.0

        for u, v in edge_additions or []:
            if u == v:
                continue
            work_adj[u, v] = 1.0
            work_adj[v, u] = 1.0
        for u, v in edge_removals or []:
            if u == v:
                continue
            work_adj[u, v] = 0.0
            work_adj[v, u] = 0.0

        current_adj = sparse.csr_matrix(work_adj, dtype=float)
        current_adj.eliminate_zeros()
        if current_adj.nnz > 0:
            current_adj.data[:] = 1.0

        after_by_node: Dict[int, set] = {}
        affected = set(touched_nodes)
        for idx in touched_nodes:
            after_by_node[idx] = set(_neighbors(current_adj, idx).tolist())
            affected.update(before_by_node[idx])
            affected.update(after_by_node[idx])

        self.norm_adj = _normalized_adjacency(current_adj)
        affected_list = sorted(affected)
        old_struct = current_struct.copy()

        for idx in affected_list:
            before_neighbors = before_by_node.get(idx, set(_neighbors(old_adj, idx).tolist()))
            after_neighbors = after_by_node.get(idx, set(_neighbors(current_adj, idx).tolist()))
            add_neighbors = after_neighbors - before_neighbors
            rem_neighbors = before_neighbors - after_neighbors

            delta_adj_y = np.zeros(self.dim, dtype=np.float64)
            if add_neighbors:
                add_idx = np.fromiter(add_neighbors, dtype=np.int64)
                delta_adj_y += old_struct[add_idx].sum(axis=0)
            if rem_neighbors:
                rem_idx = np.fromiter(rem_neighbors, dtype=np.int64)
                delta_adj_y -= old_struct[rem_idx].sum(axis=0)

            # y'_i = y_i + alpha * f(DeltaA_i y_{N(i)}), f 为线性加权聚合
            denom = max(len(after_neighbors), 1)
            f_delta = delta_adj_y / float(denom)
            current_struct[idx] = current_struct[idx] + self.learning_rate * f_delta

        current_struct = _row_normalize(current_struct)

        if attr_updates or node_additions or node_removals:
            changed_attr_nodes = set(attr_updates.keys()) if attr_updates else set()
            changed_attr_nodes.update(node_additions.keys() if node_additions else [])
            changed_attr_nodes.update(int(i) for i in (node_removals or []))
            for idx in changed_attr_nodes:
                if idx < 0 or idx >= current_attrs.shape[0]:
                    continue
                current_attr[idx] = (current_attrs[idx] @ self.attr_projection).reshape(-1)
            current_attr = _row_normalize(current_attr)

        if self.use_attr_fusion:
            fused = self._fuse_embeddings(current_struct[affected_list], current_attr[affected_list])
        else:
            fused = _row_normalize(current_struct[affected_list])
        current_emb[affected_list] = fused
        current_emb = _row_normalize(current_emb)

        self.adj = current_adj
        self.attrs = current_attrs
        self.struct_emb = current_struct
        self.attr_emb = current_attr
        self.embedding_ = current_emb
        self._refresh_quantized_embedding()
        return affected_list


def cosine_scores(embedding: ArrayLike, pairs: Sequence[Tuple[int, int]]) -> ArrayLike:
    """计算节点对余弦相似度，用于链路预测打分。"""
    emb = _row_normalize(embedding)
    return np.array([float(np.dot(emb[u], emb[v])) for u, v in pairs], dtype=np.float64)
