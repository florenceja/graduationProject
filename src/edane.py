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


def _prepare_adjacency(adj: AdjLike) -> SparseMatrix:
    """邻接矩阵预处理：对称化、去自环、二值化（返回 CSR 稀疏矩阵）。"""
    if sparse.issparse(adj):
        work = adj.tocsr().astype(np.float64)
    else:
        dense = np.asarray(adj, dtype=np.float64)
        work = sparse.csr_matrix(dense)
    work = work.maximum(work.T).tocsr()
    work.setdiag(0.0)
    work.eliminate_zeros()
    if work.nnz > 0:
        work.data[:] = 1.0
    return work


def _normalized_adjacency(adj: SparseMatrix, add_self_loops: bool = True) -> SparseMatrix:
    """构造对称归一化邻接矩阵 D^-1/2 (A+I) D^-1/2（CSR）。"""
    if add_self_loops:
        work = adj + sparse.eye(adj.shape[0], dtype=np.float64, format="csr")
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
        random_state: int = 42,
        structure_weights: Optional[Sequence[float]] = None,
    ) -> None:
        self.dim = dim
        self.order = order
        self.projection_density = projection_density
        self.learning_rate = learning_rate
        self.quantize = quantize
        self.random_state = random_state
        self.structure_weights = structure_weights
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
        self.attrs = _standardize_features(np.asarray(attrs, dtype=np.float64))
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
        self.embedding_ = self._fuse_embeddings(self.struct_emb, self.attr_emb)
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
        """计算结构嵌入 H_s = sum_k alpha_k S^k R。"""
        assert self.norm_adj is not None
        assert self.random_projection is not None
        weights = self._get_structure_weights()
        propagated = self.random_projection.copy()
        result = weights[0] * propagated
        for k in range(1, self.order + 1):
            propagated = self.norm_adj @ propagated
            result += weights[k] * propagated
        return _row_normalize(result)

    def _compute_attribute_embedding(self, attrs: ArrayLike) -> ArrayLike:
        """计算属性嵌入 H_x = X W_x。"""
        assert self.attr_projection is not None
        attr_emb = attrs @ self.attr_projection
        return _row_normalize(attr_emb)

    def _fuse_embeddings(self, struct_emb: ArrayLike, attr_emb: ArrayLike) -> ArrayLike:
        """双曲门控融合结构与属性嵌入。

        gate 由结构-属性相似度驱动，再在双曲空间做加权组合。
        """
        gate_logits = np.sum(struct_emb * attr_emb, axis=1, keepdims=True) / math.sqrt(self.dim)
        gate = _sigmoid(gate_logits)
        struct_h = _exp_map_zero(0.5 * struct_emb)
        attr_h = _exp_map_zero(0.5 * attr_emb)
        left = _mobius_scalar_mul(1.0 - gate, struct_h)
        right = _mobius_scalar_mul(gate, attr_h)
        fused_h = _mobius_add(left, right)
        fused = _log_map_zero(fused_h)
        return _row_normalize(fused)

    def _refresh_quantized_embedding(self) -> None:
        """更新 int8 量化副本，便于存储压缩对比。"""
        if self.embedding_ is None:
            return
        if not self.quantize:
            self.quantized_embedding_ = None
            return
        max_abs = np.max(np.abs(self.embedding_), axis=0, keepdims=True)
        scale = np.maximum(max_abs / 127.0, 1e-8)
        values = np.clip(np.round(self.embedding_ / scale), -127, 127).astype(np.int8)
        self.quantized_embedding_ = QuantizedEmbedding(values=values, scale=scale)

    def get_embedding(self, dequantize: bool = True) -> ArrayLike:
        """获取当前嵌入。

        dequantize=True 且存在量化副本时，返回反量化后的浮点向量。
        """
        if self.embedding_ is None:
            raise ValueError("Call fit() before get_embedding().")
        if dequantize and self.quantized_embedding_ is not None:
            return self.quantized_embedding_.dequantize()
        return self.embedding_.copy()

    def apply_updates(
        self,
        edge_additions: Optional[Iterable[Tuple[int, int]]] = None,
        edge_removals: Optional[Iterable[Tuple[int, int]]] = None,
        attr_updates: Optional[Dict[int, ArrayLike]] = None,
    ) -> List[int]:
        """对动态事件做增量更新，仅刷新局部受影响节点。

        返回值为本轮受影响节点索引列表，可用于外部统计/调试。
        """
        if self.adj is None or self.attrs is None or self.embedding_ is None:
            raise ValueError("Call fit() before apply_updates().")

        def _neighbors(mat: SparseMatrix, node: int) -> np.ndarray:
            start = mat.indptr[node]
            end = mat.indptr[node + 1]
            return mat.indices[start:end]

        old_adj = self.adj.tocsr(copy=True)
        touched_nodes = set()

        for u, v in edge_additions or []:
            if u == v:
                continue
            touched_nodes.update([u, v])
        for u, v in edge_removals or []:
            if u == v:
                continue
            touched_nodes.update([u, v])

        if attr_updates:
            for idx, value in attr_updates.items():
                self.attrs[idx] = value
                touched_nodes.add(idx)

        if not touched_nodes:
            return []

        before_by_node: Dict[int, set] = {}
        for idx in touched_nodes:
            before_by_node[idx] = set(_neighbors(old_adj, idx).tolist())

        work_adj = old_adj.tolil(copy=True)
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
        self.adj = work_adj.tocsr()
        self.adj.eliminate_zeros()
        if self.adj.nnz > 0:
            self.adj.data[:] = 1.0

        after_by_node: Dict[int, set] = {}
        affected = set(touched_nodes)
        for idx in touched_nodes:
            after_by_node[idx] = set(_neighbors(self.adj, idx).tolist())
            affected.update(before_by_node[idx])
            affected.update(after_by_node[idx])

        self.norm_adj = _normalized_adjacency(self.adj)
        affected_list = sorted(affected)

        assert self.random_projection is not None
        assert self.struct_emb is not None
        assert self.attr_projection is not None
        assert self.attr_emb is not None

        for idx in affected_list:
            # 局部更新近似：
            # h'_i = (1-eta) h_i + eta (base_i + DeltaA_i R + neighbor_agg)
            before_neighbors = before_by_node.get(idx, set(_neighbors(old_adj, idx).tolist()))
            after_neighbors = after_by_node.get(idx, set(_neighbors(self.adj, idx).tolist()))
            add_neighbors = after_neighbors - before_neighbors
            rem_neighbors = before_neighbors - after_neighbors

            base = self.random_projection[idx]
            delta_proj = np.zeros(self.dim, dtype=np.float64)
            if add_neighbors:
                add_idx = np.fromiter(add_neighbors, dtype=np.int64)
                delta_proj += self.random_projection[add_idx].sum(axis=0)
            if rem_neighbors:
                rem_idx = np.fromiter(rem_neighbors, dtype=np.int64)
                delta_proj -= self.random_projection[rem_idx].sum(axis=0)

            if after_neighbors:
                neigh_idx = np.fromiter(after_neighbors, dtype=np.int64)
                neighbor_agg = self.struct_emb[neigh_idx].mean(axis=0)
            else:
                neighbor_agg = np.zeros(self.dim, dtype=np.float64)

            updated = (1.0 - self.learning_rate) * self.struct_emb[idx]
            updated += self.learning_rate * (base + delta_proj + neighbor_agg)
            self.struct_emb[idx] = updated

        self.struct_emb = _row_normalize(self.struct_emb)

        if attr_updates:
            # 属性变化只重算变动节点对应的属性投影，避免全量重算。
            for idx in attr_updates:
                self.attr_emb[idx] = (self.attrs[idx] @ self.attr_projection).reshape(1, -1)
            self.attr_emb = _row_normalize(self.attr_emb)

        fused = self._fuse_embeddings(self.struct_emb[affected_list], self.attr_emb[affected_list])
        self.embedding_[affected_list] = fused
        self.embedding_ = _row_normalize(self.embedding_)
        self._refresh_quantized_embedding()
        return affected_list


def cosine_scores(embedding: ArrayLike, pairs: Sequence[Tuple[int, int]]) -> ArrayLike:
    """计算节点对余弦相似度，用于链路预测打分。"""
    emb = _row_normalize(embedding)
    return np.array([float(np.dot(emb[u], emb[v])) for u, v in pairs], dtype=np.float64)
