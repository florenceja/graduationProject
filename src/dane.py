"""DANE baseline for dynamic attributed network embedding.

Paper source:
Jundong Li, Harsh Dani, Xia Hu, Jiliang Tang, Yi Chang, and Huan Liu.
"Attributed Network Embedding for Learning in a Dynamic Environment."
Proceedings of CIKM 2017, pp. 387-396.

Implementation note:
- This module follows the paper's two-view design: structure embedding,
  attribute embedding, then consensus fusion.
- The online stage uses a guarded first-order perturbation update when the
  node set is unchanged. If the update is outside that regime (e.g. node
  addition/removal) or perturbation becomes numerically unstable, it falls
  back to full refit.
- Attribute similarity is approximated with a sparse top-k cosine graph so the
  baseline remains runnable in the local NumPy/SciPy pipeline.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as sparse_linalg


ArrayLike = np.ndarray
SparseMatrix = sparse.csr_matrix
AdjLike = Union[np.ndarray, SparseMatrix]


def _row_normalize(x: ArrayLike, eps: float = 1e-12) -> ArrayLike:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _prepare_adjacency(adj: AdjLike) -> SparseMatrix:
    work = sparse.csr_matrix(adj, dtype=np.float64)
    work = work.maximum(work.T).tocsr()
    work.setdiag(0.0)
    work.eliminate_zeros()
    if work.nnz > 0:
        work.data[:] = 1.0
    return work


def _ensure_finite_matrix(x: ArrayLike, name: str) -> ArrayLike:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _standardize_features(x: ArrayLike, eps: float = 1e-12) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    mean = x.mean(axis=0, keepdims=True)
    std = np.maximum(x.std(axis=0, keepdims=True), eps)
    return (x - mean) / std, mean, std


def _apply_standardization(x: ArrayLike, mean: ArrayLike, std: ArrayLike, eps: float = 1e-12) -> ArrayLike:
    return (x - mean) / np.maximum(std, eps)


def _generalized_spectral_embedding(adj_like: SparseMatrix, dim: int, seed: int) -> Tuple[ArrayLike, ArrayLike]:
    n = adj_like.shape[0]
    if n == 0:
        return np.zeros((0, dim), dtype=np.float64), np.zeros(dim, dtype=np.float64)

    degree = np.asarray(adj_like.sum(axis=1)).reshape(-1)
    laplacian = sparse.csr_matrix(sparse.diags(degree, format="csr") - adj_like, dtype=np.float64)
    degree_safe = sparse.csr_matrix(sparse.diags(np.maximum(degree, 1.0), format="csr"), dtype=np.float64)

    take = min(max(dim + 1, 2), max(n - 1, 1))
    if n <= dim + 2:
        lap_dense = laplacian.toarray()
        deg_dense = degree_safe.toarray()
        evals, evecs = linalg.eigh(lap_dense, deg_dense)
    else:
        rng = np.random.default_rng(seed)
        v0 = rng.normal(size=n)
        try:
            evals, evecs = sparse_linalg.eigsh(
                laplacian,
                k=take,
                M=degree_safe,
                which="SM",
                maxiter=max(4000, 10 * n),
                v0=v0,
            )
        except Exception:
            inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1.0))
            norm_adj = sparse.diags(inv_sqrt, format="csr") @ adj_like @ sparse.diags(inv_sqrt, format="csr")
            norm_lap = sparse.eye(n, format="csr") - norm_adj
            evals, evecs = sparse_linalg.eigsh(norm_lap, k=take, which="SM", maxiter=max(4000, 10 * n), v0=v0)

    order = np.argsort(evals)
    evals = np.asarray(evals[order], dtype=np.float64)
    evecs = np.asarray(evecs[:, order], dtype=np.float64)

    positive = np.where(evals > 1e-10)[0]
    use_idx = positive[:dim]
    if len(use_idx) < dim:
        extra = [idx for idx in range(len(evals)) if idx not in use_idx][: dim - len(use_idx)]
        use_idx = np.concatenate([use_idx, np.asarray(extra, dtype=np.int64)]) if extra else use_idx

    emb = evecs[:, use_idx] if len(use_idx) > 0 else np.zeros((n, 0), dtype=np.float64)
    if emb.shape[1] < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - emb.shape[1])), mode="constant")
    vals = evals[use_idx] if len(use_idx) > 0 else np.zeros(0, dtype=np.float64)
    if len(vals) < dim:
        vals = np.pad(vals, (0, dim - len(vals)), mode="edge" if len(vals) > 0 else "constant")
    return np.asarray(emb, dtype=np.float64), vals[:dim]


def _topk_cosine_graph(attrs: ArrayLike, topk: int, block_size: int) -> SparseMatrix:
    x = np.asarray(attrs, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float64)

    normalized = _row_normalize(x)
    row_idx: List[int] = []
    col_idx: List[int] = []
    data: List[float] = []
    topk = max(1, min(int(topk), max(n - 1, 1)))
    block_size = max(32, int(block_size))

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        block_scores = normalized[start:stop] @ normalized.T
        for local_row, scores in enumerate(block_scores):
            node = start + local_row
            scores[node] = -np.inf
            positive = np.where(scores > 0.0)[0]
            if len(positive) == 0:
                continue
            if len(positive) > topk:
                candidate_scores = scores[positive]
                top_local = np.argpartition(candidate_scores, -topk)[-topk:]
                chosen = positive[top_local]
                chosen = chosen[np.argsort(scores[chosen])[::-1]]
            else:
                chosen = positive[np.argsort(scores[positive])[::-1]]
            for idx in chosen.tolist():
                row_idx.append(node)
                col_idx.append(int(idx))
                data.append(float(scores[idx]))

    graph = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, n), dtype=np.float64)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def _consensus_embedding(struct_emb: ArrayLike, attr_emb: ArrayLike, out_dim: int) -> ArrayLike:
    ya = np.asarray(struct_emb, dtype=np.float64)
    yx = np.asarray(attr_emb, dtype=np.float64)
    a = ya.T @ ya
    b = yx.T @ yx
    cross = ya.T @ yx
    objective = np.block([[a, cross], [cross.T, b]])
    constraint = np.block(
        [
            [a + 1e-6 * np.eye(a.shape[0]), np.zeros_like(cross)],
            [np.zeros_like(cross.T), b + 1e-6 * np.eye(b.shape[0])],
        ]
    )
    evals, evecs = linalg.eigh(objective, constraint)
    order = np.argsort(evals)[::-1]
    proj = np.asarray(evecs[:, order[:out_dim]], dtype=np.float64)
    fused = np.concatenate([ya, yx], axis=1) @ proj
    return _row_normalize(fused)


class _SpectralState:
    def __init__(
        self,
        matrix: SparseMatrix,
        degree: ArrayLike,
        laplacian: SparseMatrix,
        eigvecs: ArrayLike,
        eigvals: ArrayLike,
    ) -> None:
        self.matrix = matrix
        self.degree = degree
        self.laplacian = laplacian
        self.eigvecs = eigvecs
        self.eigvals = eigvals


class DANE:
    """Dynamic Attributed Network Embedding baseline.

    This baseline exposes an EDANE-compatible interface so it can run through
    the existing evaluation pipeline.
    """

    def __init__(
        self,
        dim: int = 32,
        attr_topk: int = 20,
        similarity_block_size: int = 512,
        perturbation_rank: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        self.dim = int(dim)
        self.attr_topk = int(attr_topk)
        self.similarity_block_size = int(similarity_block_size)
        self.perturbation_rank = int(perturbation_rank) if perturbation_rank is not None else int(dim)
        self.random_state = int(random_state)

        self.adj: Optional[SparseMatrix] = None
        self.attrs_raw_: Optional[ArrayLike] = None
        self.attrs_: Optional[ArrayLike] = None
        self.embedding_: Optional[ArrayLike] = None
        self.feature_mean_: Optional[ArrayLike] = None
        self.feature_std_: Optional[ArrayLike] = None
        self.structure_state_: Optional[_SpectralState] = None
        self.attribute_state_: Optional[_SpectralState] = None
        self.quantized_embedding_ = None
        self.binary_embedding_ = None
        self.quantization_compression_ratio_ = 1.0
        self.quantization_error_ = 0.0
        self.binary_compression_ratio_ = 1.0
        self.binary_error_ = 0.0
        self.supports_incremental_updates_ = True
        self.online_update_mode_ = "perturbation"

    def _build_attribute_graph(self, attrs: ArrayLike) -> SparseMatrix:
        return _topk_cosine_graph(attrs, topk=self.attr_topk, block_size=self.similarity_block_size)

    def _compute_state(self, matrix: SparseMatrix, seed_offset: int) -> _SpectralState:
        eigvecs, eigvals = _generalized_spectral_embedding(matrix, self.dim, self.random_state + seed_offset)
        degree = np.asarray(matrix.sum(axis=1)).reshape(-1)
        lap = sparse.diags(degree, format="csr") - matrix
        return _SpectralState(matrix=matrix, degree=degree, laplacian=lap, eigvecs=eigvecs, eigvals=eigvals)

    def _fit_from_processed(self, adj: SparseMatrix, attrs_standardized: ArrayLike, attrs_raw: ArrayLike) -> "DANE":
        attr_graph = self._build_attribute_graph(attrs_standardized)
        structure_state = self._compute_state(adj, seed_offset=0)
        attribute_state = self._compute_state(attr_graph, seed_offset=1)
        embedding = _consensus_embedding(
            _row_normalize(structure_state.eigvecs),
            _row_normalize(attribute_state.eigvecs),
            self.dim,
        )

        self.adj = adj
        self.attrs_ = attrs_standardized
        self.attrs_raw_ = attrs_raw.copy()
        self.structure_state_ = structure_state
        self.attribute_state_ = attribute_state
        self.embedding_ = embedding
        return self

    def fit(self, adj: AdjLike, attrs: ArrayLike) -> "DANE":
        prepared_adj = _prepare_adjacency(adj)
        raw_attrs = _ensure_finite_matrix(attrs, "attrs")
        if prepared_adj.shape[0] != raw_attrs.shape[0]:
            raise ValueError("adjacency rows must match attrs rows.")
        attrs_standardized, mean, std = _standardize_features(raw_attrs)
        self.feature_mean_ = mean
        self.feature_std_ = std
        return self._fit_from_processed(prepared_adj, attrs_standardized, raw_attrs)

    def _perturb_eigenpairs(
        self,
        old_state: _SpectralState,
        new_matrix: SparseMatrix,
    ) -> Tuple[ArrayLike, ArrayLike]:
        old_vecs = np.asarray(old_state.eigvecs, dtype=np.float64)
        old_vals = np.asarray(old_state.eigvals, dtype=np.float64)
        if old_vecs.shape[1] == 0:
            return old_vecs, old_vals

        new_degree = np.asarray(new_matrix.sum(axis=1)).reshape(-1)
        new_lap = sparse.csr_matrix(sparse.diags(new_degree, format="csr") - new_matrix, dtype=np.float64)
        delta_l = sparse.csr_matrix(new_lap - old_state.laplacian, dtype=np.float64).toarray()
        delta_d = np.diag(new_degree - old_state.degree)

        updates = np.zeros_like(old_vecs)
        new_vals = old_vals.copy()
        rank = min(old_vecs.shape[1], self.perturbation_rank)
        for i in range(rank):
            a_i = old_vecs[:, i]
            lam_i = float(old_vals[i])
            delta_lambda = float(a_i.T @ delta_l @ a_i - lam_i * (a_i.T @ delta_d @ a_i))
            new_vals[i] = lam_i + delta_lambda

            delta_a = -0.5 * float(a_i.T @ delta_d @ a_i) * a_i
            for j in range(old_vecs.shape[1]):
                if i == j:
                    continue
                denom = lam_i - float(old_vals[j])
                if abs(denom) < 1e-8:
                    continue
                a_j = old_vecs[:, j]
                num = float(a_j.T @ delta_l @ a_i - lam_i * (a_j.T @ delta_d @ a_i))
                delta_a += (num / denom) * a_j
            updates[:, i] = a_i + delta_a

        if rank < old_vecs.shape[1]:
            updates[:, rank:] = old_vecs[:, rank:]
        updates = _row_normalize(updates)
        if not np.all(np.isfinite(updates)):
            raise ValueError("Perturbation update produced non-finite eigenvectors.")
        return updates, new_vals

    def get_embedding(self, dequantize: bool = True) -> ArrayLike:
        if self.embedding_ is None:
            raise ValueError("Call fit() before get_embedding().")
        return self.embedding_.copy()

    def apply_updates(
        self,
        node_additions: Optional[Dict[int, ArrayLike]] = None,
        node_removals: Optional[Iterable[int]] = None,
        edge_additions: Optional[Iterable[Tuple[int, int]]] = None,
        edge_removals: Optional[Iterable[Tuple[int, int]]] = None,
        attr_updates: Optional[Dict[int, ArrayLike]] = None,
    ) -> List[int]:
        if self.adj is None or self.attrs_raw_ is None or self.attrs_ is None:
            raise ValueError("Call fit() before apply_updates().")
        if self.feature_mean_ is None or self.feature_std_ is None:
            raise ValueError("Call fit() before apply_updates().")
        if node_additions or node_removals:
            self.online_update_mode_ = "refit"
            adj = self.adj
            attrs_raw = self.attrs_raw_.copy()
            if node_additions:
                sorted_items = sorted((int(idx), np.asarray(value, dtype=np.float64)) for idx, value in node_additions.items())
                expected = list(range(attrs_raw.shape[0], attrs_raw.shape[0] + len(sorted_items)))
                actual = [idx for idx, _ in sorted_items]
                if actual != expected:
                    raise ValueError("node_additions must be contiguous new indices starting at current node count.")
                attrs_raw = np.vstack([attrs_raw, np.vstack([value for _, value in sorted_items])])
                adj = sparse.csr_matrix(
                    sparse.vstack([adj, sparse.csr_matrix((len(sorted_items), adj.shape[1]))], format="csr"),
                    dtype=np.float64,
                )
                adj = sparse.csr_matrix(
                    sparse.hstack([adj, sparse.csr_matrix((adj.shape[0], len(sorted_items)))], format="csr"),
                    dtype=np.float64,
                )
            for idx in node_removals or []:
                idx = int(idx)
                adj = adj.tolil(copy=True)
                adj[idx, :] = 0.0
                adj[:, idx] = 0.0
                attrs_raw[idx] = 0.0
                adj = adj.tocsr()
            for u, v in edge_additions or []:
                if u != v:
                    work = adj.tolil(copy=True)
                    work[u, v] = 1.0
                    work[v, u] = 1.0
                    adj = work.tocsr()
            for u, v in edge_removals or []:
                if u != v:
                    work = adj.tolil(copy=True)
                    work[u, v] = 0.0
                    work[v, u] = 0.0
                    adj = work.tocsr()
            if attr_updates:
                for idx, value in attr_updates.items():
                    attrs_raw[int(idx)] = np.asarray(value, dtype=np.float64)
            self.fit(sparse.csr_matrix(adj, dtype=np.float64), attrs_raw)
            return list(range(self.embedding_.shape[0])) if self.embedding_ is not None else []

        new_adj = self.adj.tolil(copy=True)
        touched_nodes: set[int] = set()
        for u, v in edge_additions or []:
            u, v = int(u), int(v)
            if u == v:
                continue
            new_adj[u, v] = 1.0
            new_adj[v, u] = 1.0
            touched_nodes.update([u, v])
        for u, v in edge_removals or []:
            u, v = int(u), int(v)
            if u == v:
                continue
            new_adj[u, v] = 0.0
            new_adj[v, u] = 0.0
            touched_nodes.update([u, v])
        new_adj_csr = _prepare_adjacency(sparse.csr_matrix(new_adj.tocsr(), dtype=np.float64))

        new_raw_attrs = self.attrs_raw_.copy()
        for idx, value in (attr_updates or {}).items():
            idx = int(idx)
            raw_value = np.asarray(value, dtype=np.float64)
            if raw_value.ndim != 1 or raw_value.shape[0] != new_raw_attrs.shape[1]:
                raise ValueError(f"attr update for node {idx} must be 1D with dim={new_raw_attrs.shape[1]}.")
            if not np.all(np.isfinite(raw_value)):
                raise ValueError(f"attr update for node {idx} must contain only finite values.")
            new_raw_attrs[idx] = raw_value
            touched_nodes.add(idx)
        new_attrs = _apply_standardization(new_raw_attrs, self.feature_mean_, self.feature_std_)
        new_attr_graph = self._build_attribute_graph(new_attrs)

        try:
            assert self.structure_state_ is not None
            assert self.attribute_state_ is not None
            struct_vecs, struct_vals = self._perturb_eigenpairs(self.structure_state_, new_adj_csr)
            attr_vecs, attr_vals = self._perturb_eigenpairs(self.attribute_state_, new_attr_graph)
            self.structure_state_ = _SpectralState(
                matrix=new_adj_csr,
                degree=np.asarray(new_adj_csr.sum(axis=1)).reshape(-1),
                laplacian=sparse.diags(np.asarray(new_adj_csr.sum(axis=1)).reshape(-1), format="csr") - new_adj_csr,
                eigvecs=struct_vecs,
                eigvals=struct_vals,
            )
            self.attribute_state_ = _SpectralState(
                matrix=new_attr_graph,
                degree=np.asarray(new_attr_graph.sum(axis=1)).reshape(-1),
                laplacian=sparse.diags(np.asarray(new_attr_graph.sum(axis=1)).reshape(-1), format="csr") - new_attr_graph,
                eigvecs=attr_vecs,
                eigvals=attr_vals,
            )
            self.adj = new_adj_csr
            self.attrs_raw_ = new_raw_attrs
            self.attrs_ = new_attrs
            self.embedding_ = _consensus_embedding(
                _row_normalize(struct_vecs),
                _row_normalize(attr_vecs),
                self.dim,
            )
            self.online_update_mode_ = "perturbation"
        except Exception:
            self.online_update_mode_ = "refit"
            self._fit_from_processed(new_adj_csr, new_attrs, new_raw_attrs)

        if not touched_nodes and self.embedding_ is not None:
            return []
        assert self.embedding_ is not None
        return sorted(touched_nodes) if touched_nodes else list(range(self.embedding_.shape[0]))
