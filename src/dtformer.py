"""DTFormer-style baseline for discrete-time dynamic graph representation learning.

Paper source:
Xi Chen, Yun Xiong, Siwei Zhang, Jiawei Zhang, Yao Zhang, Shiyang Zhou,
Xixi Wu, Mingyang Zhang, Tengfei Liu, and Weiqiang Wang.
"DTFormer: A Transformer-Based Method for Discrete-Time Dynamic Graph
Representation Learning." CIKM 2024.

Implementation note:
- The official DTFormer release is a PyTorch link-prediction model over
  discrete-time edge-event histories.
- This module provides a runnable adapter for the local embedding pipeline by
  approximating DTFormer's core ideas: snapshot history, patching, positional
  temporal encoding, and Transformer-style temporal aggregation.
- It is a paper-inspired baseline adapter, not an exact reimplementation of the
  official training script.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse


ArrayLike = np.ndarray
SparseMatrix = sparse.csr_matrix
AdjLike = Union[np.ndarray, SparseMatrix]


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


def _row_normalize(x: ArrayLike, eps: float = 1e-12) -> ArrayLike:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _standardize_features(x: ArrayLike, eps: float = 1e-12) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    mean = x.mean(axis=0, keepdims=True)
    std = np.maximum(x.std(axis=0, keepdims=True), eps)
    return (x - mean) / std, mean, std


def _apply_standardization(x: ArrayLike, mean: ArrayLike, std: ArrayLike, eps: float = 1e-12) -> ArrayLike:
    return (x - mean) / np.maximum(std, eps)


class DTFormer:
    """Paper-inspired DTFormer adapter for the local embedding pipeline."""

    def __init__(
        self,
        dim: int = 64,
        patch_size: int = 2,
        history_snapshots: int = 8,
        transformer_hidden_dim: int = 96,
        attention_temperature: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.dim = int(dim)
        self.patch_size = max(1, int(patch_size))
        self.history_snapshots = max(1, int(history_snapshots))
        self.transformer_hidden_dim = max(self.dim, int(transformer_hidden_dim))
        self.attention_temperature = float(max(attention_temperature, 1e-6))
        self.random_state = int(random_state)

        rng = np.random.default_rng(self.random_state)
        self.W_struct_ = rng.normal(0.0, 1.0 / np.sqrt(self.transformer_hidden_dim), size=(self.transformer_hidden_dim, self.transformer_hidden_dim))
        self.W_attr_ = rng.normal(0.0, 1.0 / np.sqrt(self.transformer_hidden_dim), size=(self.transformer_hidden_dim, self.transformer_hidden_dim))
        self.W_q_ = rng.normal(0.0, 1.0 / np.sqrt(self.transformer_hidden_dim), size=(self.transformer_hidden_dim, self.transformer_hidden_dim))
        self.W_k_ = rng.normal(0.0, 1.0 / np.sqrt(self.transformer_hidden_dim), size=(self.transformer_hidden_dim, self.transformer_hidden_dim))
        self.W_v_ = rng.normal(0.0, 1.0 / np.sqrt(self.transformer_hidden_dim), size=(self.transformer_hidden_dim, self.transformer_hidden_dim))
        self.W_out_ = rng.normal(0.0, 1.0 / np.sqrt(self.transformer_hidden_dim), size=(self.transformer_hidden_dim, self.dim))

        self.adj: Optional[SparseMatrix] = None
        self.attrs_raw_: Optional[ArrayLike] = None
        self.attrs_: Optional[ArrayLike] = None
        self.feature_mean_: Optional[ArrayLike] = None
        self.feature_std_: Optional[ArrayLike] = None
        self.embedding_: Optional[ArrayLike] = None
        self.snapshot_history_: List[ArrayLike] = []
        self.quantized_embedding_ = None
        self.binary_embedding_ = None
        self.quantization_compression_ratio_ = 1.0
        self.quantization_error_ = 0.0
        self.binary_compression_ratio_ = 1.0
        self.binary_error_ = 0.0
        self.supports_incremental_updates_ = False
        self.online_update_mode_ = "refit"

    def _project_to_hidden(self, x: ArrayLike, target_dim: int) -> ArrayLike:
        if x.shape[1] == target_dim:
            return np.asarray(x, dtype=np.float64)
        if x.shape[1] > target_dim:
            return np.asarray(x[:, :target_dim], dtype=np.float64)
        return np.pad(np.asarray(x, dtype=np.float64), ((0, 0), (0, target_dim - x.shape[1])), mode="constant")

    def _compute_snapshot_token(self, adj: SparseMatrix, attrs_standardized: ArrayLike, snapshot_index: int) -> ArrayLike:
        degrees = np.asarray(adj.sum(axis=1)).reshape(-1, 1)
        inv_degree = 1.0 / np.maximum(degrees, 1.0)
        smoothed_attrs = adj @ attrs_standardized
        smoothed_attrs = smoothed_attrs * inv_degree
        struct_input = np.concatenate([smoothed_attrs, degrees, np.log1p(degrees)], axis=1)
        attr_input = np.concatenate([attrs_standardized, attrs_standardized ** 2], axis=1)

        struct_hidden = np.tanh(self._project_to_hidden(struct_input, self.transformer_hidden_dim) @ self.W_struct_)
        attr_hidden = np.tanh(self._project_to_hidden(attr_input, self.transformer_hidden_dim) @ self.W_attr_)

        position_scale = float(snapshot_index + 1) / float(max(self.history_snapshots, 1))
        count_channel = np.log1p(degrees)
        count_hidden = self._project_to_hidden(np.repeat(count_channel, self.transformer_hidden_dim, axis=1), self.transformer_hidden_dim)
        token = struct_hidden + attr_hidden + position_scale * count_hidden
        return np.asarray(token, dtype=np.float64)

    def _patch_tokens(self, tokens: ArrayLike) -> ArrayLike:
        # tokens: [n, t, h]
        n, t, h = tokens.shape
        remainder = t % self.patch_size
        if remainder != 0:
            pad = self.patch_size - remainder
            pad_block = np.repeat(tokens[:, -1:, :], pad, axis=1)
            tokens = np.concatenate([tokens, pad_block], axis=1)
            t = tokens.shape[1]
        num_patches = t // self.patch_size
        reshaped = tokens.reshape(n, num_patches, self.patch_size, h)
        return reshaped.mean(axis=2)

    def _transform_history(self) -> ArrayLike:
        if not self.snapshot_history_:
            return np.zeros((0, self.dim), dtype=np.float64)
        history = self.snapshot_history_[-self.history_snapshots :]
        tokens = np.stack(history, axis=1)  # [n, t, h]
        patched = self._patch_tokens(tokens)
        q = patched @ self.W_q_
        k = patched @ self.W_k_
        v = patched @ self.W_v_
        scores = np.einsum("nth,nsh->nts", q, k) / np.sqrt(float(self.transformer_hidden_dim) * self.attention_temperature)
        scores = scores - scores.max(axis=2, keepdims=True)
        weights = np.exp(scores)
        weights = weights / np.maximum(weights.sum(axis=2, keepdims=True), 1e-12)
        attended = np.einsum("nts,nsh->nth", weights, v)
        latest_token = tokens[:, -1, :]
        temporal_summary = attended.mean(axis=1) + latest_token
        embedding = np.tanh(temporal_summary @ self.W_out_)
        return _row_normalize(embedding)

    def _fit_from_processed(self, adj: SparseMatrix, attrs_standardized: ArrayLike, attrs_raw: ArrayLike) -> "DTFormer":
        snapshot_index = len(self.snapshot_history_)
        token = self._compute_snapshot_token(adj, attrs_standardized, snapshot_index)
        self.snapshot_history_.append(token)
        if len(self.snapshot_history_) > self.history_snapshots:
            self.snapshot_history_ = self.snapshot_history_[-self.history_snapshots :]
        self.adj = adj
        self.attrs_ = attrs_standardized
        self.attrs_raw_ = attrs_raw.copy()
        self.embedding_ = self._transform_history()
        return self

    def fit(self, adj: AdjLike, attrs: ArrayLike) -> "DTFormer":
        prepared_adj = _prepare_adjacency(adj)
        raw_attrs = _ensure_finite_matrix(attrs, "attrs")
        if prepared_adj.shape[0] != raw_attrs.shape[0]:
            raise ValueError("adjacency rows must match attrs rows.")
        attrs_standardized, mean, std = _standardize_features(raw_attrs)
        self.feature_mean_ = mean
        self.feature_std_ = std
        self.snapshot_history_ = []
        return self._fit_from_processed(prepared_adj, attrs_standardized, raw_attrs)

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
        if self.adj is None or self.attrs_raw_ is None or self.feature_mean_ is None or self.feature_std_ is None:
            raise ValueError("Call fit() before apply_updates().")

        adj = self.adj
        attrs_raw = self.attrs_raw_.copy()
        touched_nodes: set[int] = set()

        if node_additions:
            sorted_items = sorted((int(idx), np.asarray(value, dtype=np.float64)) for idx, value in node_additions.items())
            expected = list(range(attrs_raw.shape[0], attrs_raw.shape[0] + len(sorted_items)))
            actual = [idx for idx, _ in sorted_items]
            if actual != expected:
                raise ValueError("node_additions must be contiguous new indices starting at current node count.")
            attrs_raw = np.vstack([attrs_raw, np.vstack([value for _, value in sorted_items])])
            adj = sparse.csr_matrix(sparse.vstack([adj, sparse.csr_matrix((len(sorted_items), adj.shape[1]))], format="csr"), dtype=np.float64)
            adj = sparse.csr_matrix(sparse.hstack([adj, sparse.csr_matrix((adj.shape[0], len(sorted_items)))], format="csr"), dtype=np.float64)
            touched_nodes.update(actual)

        work = adj.tolil(copy=True)
        for idx in node_removals or []:
            idx = int(idx)
            work[idx, :] = 0.0
            work[:, idx] = 0.0
            attrs_raw[idx] = 0.0
            touched_nodes.add(idx)
        for u, v in edge_additions or []:
            u, v = int(u), int(v)
            if u == v:
                continue
            work[u, v] = 1.0
            work[v, u] = 1.0
            touched_nodes.update([u, v])
        for u, v in edge_removals or []:
            u, v = int(u), int(v)
            if u == v:
                continue
            work[u, v] = 0.0
            work[v, u] = 0.0
            touched_nodes.update([u, v])
        for idx, value in (attr_updates or {}).items():
            idx = int(idx)
            raw_value = np.asarray(value, dtype=np.float64)
            if raw_value.ndim != 1 or raw_value.shape[0] != attrs_raw.shape[1]:
                raise ValueError(f"attr update for node {idx} must be 1D with dim={attrs_raw.shape[1]}.")
            if not np.all(np.isfinite(raw_value)):
                raise ValueError(f"attr update for node {idx} must contain only finite values.")
            attrs_raw[idx] = raw_value
            touched_nodes.add(idx)

        next_adj = _prepare_adjacency(sparse.csr_matrix(work.tocsr(), dtype=np.float64))
        next_attrs = _apply_standardization(attrs_raw, self.feature_mean_, self.feature_std_)
        self._fit_from_processed(next_adj, next_attrs, attrs_raw)
        self.online_update_mode_ = "refit"
        assert self.embedding_ is not None
        return sorted(touched_nodes) if touched_nodes else list(range(self.embedding_.shape[0]))
