"""EDANE 端到端实验流水线。

覆盖流程：
1) 数据读取与清洗（CSV -> 图/特征/标签）
2) 动态快照构建（window / cumulative）
3) EDANE 初始化与增量更新
4) 评估（分类 F1 + 链路预测 AUC/AP + 网络重构 AUC）
5) 结果落盘（summary / 指标明细 / 嵌入 / 节点映射）
"""

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse

from dane import DANE
from dtformer import DTFormer
from edane import EDANE, cosine_scores

AdjLike = Union[np.ndarray, sparse.csr_matrix]


@dataclass
class DynamicBatch:
    """单个快照增量事件包。"""

    edge_additions: List[Tuple[int, int]]
    edge_removals: List[Tuple[int, int]]
    attr_updates: Dict[int, np.ndarray]


def ensure_dir(path: str) -> None:
    """若目录不存在则创建。"""
    os.makedirs(path, exist_ok=True)


def train_test_split(indices: np.ndarray, train_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """按给定比例随机切分索引。"""
    rng = np.random.default_rng(seed)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return shuffled[:split], shuffled[split:]


def stratified_train_test_split(
    labels: np.ndarray,
    indices: np.ndarray,
    train_ratio: float,
    seed: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """按类别分层切分索引，保证每个类别在 train/test 中都至少出现一次。

    若任一类别样本数不足 2，则返回 None，由上层决定是否回退到随机切分。
    """
    if len(indices) == 0:
        return None
    subset_labels = labels[indices]
    classes, counts = np.unique(subset_labels, return_counts=True)
    if len(classes) < 2 or np.any(counts < 2):
        return None

    rng = np.random.default_rng(seed)
    train_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    for cls in classes:
        cls_indices = indices[subset_labels == cls].copy()
        rng.shuffle(cls_indices)
        train_count = int(round(len(cls_indices) * train_ratio))
        train_count = min(max(train_count, 1), len(cls_indices) - 1)
        train_parts.append(cls_indices[:train_count])
        test_parts.append(cls_indices[train_count:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def prepare_labels_for_evaluation(
    labels: np.ndarray,
    cleanup_mode: str,
    min_class_support: int,
) -> Tuple[np.ndarray, Dict[str, Union[float, str]]]:
    """按评估规则过滤低支持类别，并将保留标签重映射为连续整数。"""
    filtered = np.asarray(labels, dtype=np.int64).copy()
    raw_valid_nodes = np.where(filtered >= 0)[0]
    raw_label_count = len(raw_valid_nodes)
    raw_class_count = len(np.unique(filtered[raw_valid_nodes])) if raw_label_count > 0 else 0

    metadata: Dict[str, Union[float, str]] = {
        "label_cleanup_mode": cleanup_mode,
        "min_class_support": float(max(min_class_support, 2)),
        "labeled_nodes_raw": float(raw_label_count),
        "eval_dropped_labeled_nodes": 0.0,
        "eval_class_count_raw": float(raw_class_count),
        "eval_class_count": float(raw_class_count),
        "eval_dropped_class_count": 0.0,
    }
    if cleanup_mode != "eval_only" or raw_label_count == 0:
        return filtered, metadata

    effective_min_support = max(min_class_support, 2)
    raw_labels = filtered[raw_valid_nodes]
    unique_labels, counts = np.unique(raw_labels, return_counts=True)
    kept_labels = [int(label) for label, count in zip(unique_labels.tolist(), counts.tolist()) if count >= effective_min_support]
    kept_label_set = set(kept_labels)
    if len(kept_label_set) < 2:
        metadata["eval_dropped_labeled_nodes"] = float(raw_label_count)
        metadata["eval_class_count"] = 0.0
        metadata["eval_dropped_class_count"] = float(raw_class_count)
        filtered[raw_valid_nodes] = -1
        return filtered, metadata

    dense_map = {label: idx for idx, label in enumerate(sorted(kept_label_set))}
    dropped_nodes = 0
    for node_idx in raw_valid_nodes:
        label = int(filtered[node_idx])
        if label not in kept_label_set:
            filtered[node_idx] = -1
            dropped_nodes += 1
            continue
        filtered[node_idx] = dense_map[label]

    metadata["eval_dropped_labeled_nodes"] = float(dropped_nodes)
    metadata["eval_class_count"] = float(len(kept_label_set))
    metadata["eval_dropped_class_count"] = float(raw_class_count - len(kept_label_set))
    return filtered, metadata


def macro_micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """手工计算 Macro-F1 / Micro-F1，避免额外依赖 sklearn。"""
    labels = np.unique(y_true)
    macro_scores: List[float] = []
    tp_total = fp_total = fn_total = 0

    for label in labels:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        tp_total += tp
        fp_total += fp
        fn_total += fn
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            macro_scores.append(0.0)
        else:
            macro_scores.append(2.0 * precision * recall / (precision + recall))

    macro = float(np.mean(macro_scores)) if macro_scores else 0.0
    micro_precision = tp_total / max(tp_total + fp_total, 1)
    micro_recall = tp_total / max(tp_total + fn_total, 1)
    if micro_precision + micro_recall == 0:
        micro = 0.0
    else:
        micro = 2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
    return macro, micro


def nearest_centroid_predict(
    embedding: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    """最近类中心分类器（轻量评估基线）。"""
    train_labels = labels[train_idx]
    classes = np.unique(train_labels)
    centroids = []
    for cls in classes:
        cls_nodes = train_idx[train_labels == cls]
        centroids.append(embedding[cls_nodes].mean(axis=0))
    centroid_arr = np.vstack(centroids)
    centroid_arr /= np.maximum(np.linalg.norm(centroid_arr, axis=1, keepdims=True), 1e-12)

    test_emb = embedding[test_idx].copy()
    test_emb /= np.maximum(np.linalg.norm(test_emb, axis=1, keepdims=True), 1e-12)
    scores = test_emb @ centroid_arr.T
    return classes[np.argmax(scores, axis=1)]


def _softmax(logits: np.ndarray) -> np.ndarray:
    """按行计算 softmax，包含数值稳定处理。"""
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.maximum(np.sum(exp_z, axis=1, keepdims=True), 1e-12)


def softmax_logreg_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    epochs: int = 250,
    lr: float = 0.3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    class_weight_mode: str = "none",
) -> np.ndarray:
    """使用 NumPy 训练多分类 softmax 逻辑回归并预测。

    这是一个轻量实现，用于替换最近类中心分类器，通常能给出更有参考价值的 F1。
    """
    classes = np.unique(train_y)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[v] for v in train_y], dtype=np.int64)
    num_classes = len(classes)
    if num_classes < 2:
        return np.full(len(test_x), classes[0], dtype=train_y.dtype)

    x_mean = train_x.mean(axis=0, keepdims=True)
    x_std = np.maximum(train_x.std(axis=0, keepdims=True), 1e-8)
    x_train = (train_x - x_mean) / x_std
    x_test = (test_x - x_mean) / x_std

    x_train_b = np.hstack([x_train, np.ones((len(x_train), 1), dtype=np.float64)])
    x_test_b = np.hstack([x_test, np.ones((len(x_test), 1), dtype=np.float64)])
    d = x_train_b.shape[1]

    rng = np.random.default_rng(seed)
    w = 0.01 * rng.normal(size=(d, num_classes))

    y_onehot = np.zeros((len(y_idx), num_classes), dtype=np.float64)
    y_onehot[np.arange(len(y_idx)), y_idx] = 1.0

    sample_weights = np.ones(len(y_idx), dtype=np.float64)
    if class_weight_mode == "balanced":
        class_counts = np.bincount(y_idx, minlength=num_classes).astype(np.float64)
        class_weights = len(y_idx) / np.maximum(class_counts * num_classes, 1e-12)
        sample_weights = class_weights[y_idx]
    normalizer = float(np.sum(sample_weights))

    for _ in range(max(20, epochs)):
        probs = _softmax(x_train_b @ w)
        weighted_errors = (probs - y_onehot) * sample_weights[:, None]
        grad = (x_train_b.T @ weighted_errors) / max(normalizer, 1e-12)
        # 不对偏置项做正则。
        reg = weight_decay * w
        reg[-1, :] = 0.0
        w -= lr * (grad + reg)

    pred_idx = np.argmax(x_test_b @ w, axis=1)
    return classes[pred_idx]


def sample_link_pairs(adj: AdjLike, sample_size: int, seed: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """从当前图中采样正负边对，用于链路预测评估。"""
    rng = np.random.default_rng(seed)
    n = adj.shape[0]

    if sparse.issparse(adj):
        coo = sparse.triu(adj, k=1).tocoo()
        if coo.nnz == 0:
            return [], []
        pos_candidates = np.column_stack([coo.row, coo.col])
        pos_take = min(sample_size, len(pos_candidates))
        pos_idx = rng.choice(len(pos_candidates), size=pos_take, replace=False)
        pos_pairs = [(int(row[0]), int(row[1])) for row in pos_candidates[pos_idx]]

        edge_set = set((int(u), int(v)) for u, v in pos_candidates.tolist())
        neg_pairs: List[Tuple[int, int]] = []
        max_trials = max(sample_size * 30, 1000)
        trials = 0
        while len(neg_pairs) < pos_take and trials < max_trials:
            u = int(rng.integers(0, n))
            v = int(rng.integers(0, n))
            if u == v:
                trials += 1
                continue
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                trials += 1
                continue
            neg_pairs.append((u, v))
            trials += 1
        return pos_pairs, neg_pairs

    dense_adj = np.asarray(adj, dtype=np.float64)
    pos_candidates = np.argwhere(np.triu(dense_adj, 1) > 0.0)
    neg_candidates = np.argwhere(np.triu(1.0 - dense_adj - np.eye(n), 1) > 0.0)
    if len(pos_candidates) == 0 or len(neg_candidates) == 0:
        return [], []
    pos_take = min(sample_size, len(pos_candidates))
    neg_take = min(sample_size, len(neg_candidates))
    pos_idx = rng.choice(len(pos_candidates), size=pos_take, replace=False)
    neg_idx = rng.choice(len(neg_candidates), size=neg_take, replace=False)
    pos_pairs = [(int(row[0]), int(row[1])) for row in pos_candidates[pos_idx]]
    neg_pairs = [(int(row[0]), int(row[1])) for row in neg_candidates[neg_idx]]
    return pos_pairs, neg_pairs


def auc_from_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """根据正负样本得分计算 AUC（含 tie=0.5 处理）。"""
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0
    comparisons = (pos_scores[:, None] > neg_scores[None, :]).astype(np.float64)
    ties = (pos_scores[:, None] == neg_scores[None, :]).astype(np.float64) * 0.5
    return float(np.mean(comparisons + ties))


def average_precision_from_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """根据正负样本得分计算 Average Precision。"""
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0
    y_true = np.concatenate(
        [np.ones(len(pos_scores), dtype=np.int64), np.zeros(len(neg_scores), dtype=np.int64)]
    )
    y_score = np.concatenate([pos_scores, neg_scores])
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]

    pos_total = int(np.sum(y_true))
    if pos_total == 0:
        return 0.0
    tp = 0
    precisions: List[float] = []
    for rank, label in enumerate(y_true, start=1):
        if label == 1:
            tp += 1
            precisions.append(tp / rank)
    return float(np.sum(precisions) / pos_total)


def evaluate_snapshot(
    embedding: np.ndarray,
    labels: np.ndarray,
    adj: AdjLike,
    seed: int,
    classifier: str,
    logreg_epochs: int,
    logreg_lr: float,
    logreg_weight_decay: float,
    eval_protocol: str,
    eval_repeats: int,
    eval_train_ratio: float,
    label_cleanup_mode: str,
    min_class_support: int,
    logreg_class_weight: str,
) -> Dict[str, Union[float, str]]:
    """评估当前快照上的节点分类、链路预测与网络重构指标。"""
    eval_labels, label_meta = prepare_labels_for_evaluation(
        labels,
        cleanup_mode=label_cleanup_mode,
        min_class_support=min_class_support,
    )
    valid_nodes = np.where(eval_labels >= 0)[0]
    metrics: Dict[str, Union[float, str]] = {
        "macro_f1": 0.0,
        "micro_f1": 0.0,
        "macro_f1_std": 0.0,
        "micro_f1_std": 0.0,
        "link_auc": 0.0,
        "link_ap": 0.0,
        "reconstruction_auc": 0.0,
        "labeled_nodes": float(len(valid_nodes)),
        **label_meta,
        "f1_eval_repeats": float(max(eval_repeats, 1) if eval_protocol == "repeated_stratified" else 1),
        "f1_eval_successful_repeats": 0.0,
        "f1_eval_protocol": eval_protocol,
    }
    if len(valid_nodes) > 10:
        macro_scores: List[float] = []
        micro_scores: List[float] = []
        protocol_used = eval_protocol
        total_repeats = max(eval_repeats, 1) if eval_protocol == "repeated_stratified" else 1
        for repeat in range(total_repeats):
            split_seed = seed + repeat
            if eval_protocol == "repeated_stratified":
                split = stratified_train_test_split(eval_labels, valid_nodes, train_ratio=eval_train_ratio, seed=split_seed)
                if split is None:
                    protocol_used = "single_random_fallback"
                    split = train_test_split(valid_nodes, train_ratio=eval_train_ratio, seed=split_seed)
            else:
                split = train_test_split(valid_nodes, train_ratio=eval_train_ratio, seed=split_seed)

            train_idx, test_idx = split
            if len(train_idx) == 0 or len(test_idx) == 0 or len(np.unique(eval_labels[train_idx])) < 2:
                continue
            if classifier == "logreg":
                pred = softmax_logreg_predict(
                    train_x=embedding[train_idx],
                    train_y=eval_labels[train_idx],
                    test_x=embedding[test_idx],
                    epochs=logreg_epochs,
                    lr=logreg_lr,
                    weight_decay=logreg_weight_decay,
                    seed=split_seed,
                    class_weight_mode=logreg_class_weight,
                )
            else:
                pred = nearest_centroid_predict(embedding, eval_labels, train_idx, test_idx)
            macro, micro = macro_micro_f1(eval_labels[test_idx], pred)
            macro_scores.append(macro)
            micro_scores.append(micro)

        metrics["f1_eval_protocol"] = protocol_used
        metrics["f1_eval_successful_repeats"] = float(len(macro_scores))
        if macro_scores:
            metrics["macro_f1"] = float(np.mean(macro_scores))
            metrics["micro_f1"] = float(np.mean(micro_scores))
            metrics["macro_f1_std"] = float(np.std(macro_scores))
            metrics["micro_f1_std"] = float(np.std(micro_scores))

    pos_pairs, neg_pairs = sample_link_pairs(adj, sample_size=512, seed=seed)
    pos_scores = cosine_scores(embedding, pos_pairs) if pos_pairs else np.array([], dtype=np.float64)
    neg_scores = cosine_scores(embedding, neg_pairs) if neg_pairs else np.array([], dtype=np.float64)
    metrics["link_auc"] = auc_from_scores(pos_scores, neg_scores)
    metrics["link_ap"] = average_precision_from_scores(pos_scores, neg_scores)

    recon_pos_pairs, recon_neg_pairs = sample_link_pairs(adj, sample_size=1024, seed=seed + 999)
    recon_pos_scores = cosine_scores(embedding, recon_pos_pairs) if recon_pos_pairs else np.array([], dtype=np.float64)
    recon_neg_scores = cosine_scores(embedding, recon_neg_pairs) if recon_neg_pairs else np.array([], dtype=np.float64)
    metrics["reconstruction_auc"] = auc_from_scores(recon_pos_scores, recon_neg_scores)
    return metrics


def parse_time_value(value: str) -> float:
    """将 time 字段解析为可排序数值。

    支持：
    - 数字时间戳
    - ISO 时间字符串
    - 无法解析时退化为 hash 顺序（保证可分桶）
    """
    text = value.strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        # Try ISO datetime and fall back to hash ordering.
        try:
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return float(abs(hash(text)) % (10**9))


def read_csv_dict(path: str) -> List[Dict[str, str]]:
    """读取 CSV 为字典列表，自动兼容 UTF-8 BOM。"""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [row for row in reader]


def parse_numeric_vector(
    row: Dict[str, str],
    columns: Sequence[str],
    context: str,
) -> np.ndarray:
    """将指定列解析为有限浮点向量，空值补 0。"""
    values: List[float] = []
    for column in columns:
        raw = row.get(column)
        text = "" if raw is None else raw.strip()
        try:
            values.append(float(text) if text != "" else 0.0)
        except ValueError as exc:
            raise ValueError(f"{context} 中列 {column} 的值无法解析为数值: {raw!r}") from exc
    vec = np.asarray(values, dtype=np.float64)
    if vec.ndim != 1 or vec.shape[0] != len(columns):
        raise ValueError(f"{context} 维度非法，期望 {len(columns)} 维。")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{context} 包含 NaN 或 Inf。")
    return vec


def build_graph_from_files(
    edges_path: str,
    features_path: Optional[str],
    labels_path: Optional[str],
    attr_updates_path: Optional[str],
    snapshots: int,
    snapshot_mode: str,
    max_nodes: int = 10000,
) -> Tuple[AdjLike, np.ndarray, np.ndarray, List[DynamicBatch], Dict[str, int]]:
    """从用户提供的 CSV 构建实验输入。

    返回：
    - 初始邻接矩阵 adj0
    - 节点属性矩阵 attrs
    - 标签向量 labels（无标签时为 -1）
    - 动态批次列表 batches
    - 节点 ID 到索引映射 node_to_idx
    """
    edge_rows = read_csv_dict(edges_path)
    if len(edge_rows) == 0:
        raise ValueError("边文件为空，至少需要一条边。")

    required_cols = {"src", "dst"}
    if not required_cols.issubset(set(edge_rows[0].keys())):
        raise ValueError("边文件必须包含列: src,dst，且可选 time。")

    has_time = "time" in edge_rows[0]
    temporal_edges: List[Tuple[str, str, float]] = []
    node_ids = set()
    for row in edge_rows:
        src = row["src"].strip()
        dst = row["dst"].strip()
        if src == "" or dst == "" or src == dst:
            continue
        t = parse_time_value(row["time"]) if has_time else 0.0
        temporal_edges.append((src, dst, t))
        node_ids.add(src)
        node_ids.add(dst)

    feature_map: Dict[str, np.ndarray] = {}
    feature_dim = 0
    if features_path:
        feature_rows = read_csv_dict(features_path)
        if len(feature_rows) > 0:
            if "node_id" not in feature_rows[0]:
                raise ValueError("特征文件必须包含列: node_id,f1,f2,...")
            feature_cols = [c for c in feature_rows[0].keys() if c != "node_id"]
            if len(feature_cols) == 0:
                raise ValueError("特征文件至少需要一个数值特征列。")
            feature_dim = len(feature_cols)
            for feature_row_idx, row in enumerate(feature_rows, start=2):
                node = row["node_id"].strip()
                if node == "":
                    continue
                feature_map[node] = parse_numeric_vector(
                    row,
                    feature_cols,
                    context=f"features.csv 第 {feature_row_idx} 行 node_id={node}",
                )
                node_ids.add(node)

    label_map: Dict[str, int] = {}
    if labels_path:
        label_rows = read_csv_dict(labels_path)
        if len(label_rows) > 0:
            if "node_id" not in label_rows[0] or "label" not in label_rows[0]:
                raise ValueError("标签文件必须包含列: node_id,label")
            for row in label_rows:
                node = row["node_id"].strip()
                if node == "":
                    continue
                label_map[node] = int(float(row["label"]))
                node_ids.add(node)

    # 大图保护：默认在 file 模式下限制节点规模，避免稠密邻接导致内存溢出。
    # max_nodes <= 0 表示不限制。
    original_num_nodes = len(node_ids)
    if max_nodes > 0 and original_num_nodes > max_nodes:
        degree_counter: Dict[str, int] = {}
        for src, dst, _ in temporal_edges:
            degree_counter[src] = degree_counter.get(src, 0) + 1
            degree_counter[dst] = degree_counter.get(dst, 0) + 1
        ranked_nodes = sorted(node_ids, key=lambda nid: (-degree_counter.get(nid, 0), nid))
        selected_nodes = set(ranked_nodes[:max_nodes])
        temporal_edges = [e for e in temporal_edges if e[0] in selected_nodes and e[1] in selected_nodes]
        feature_map = {k: v for k, v in feature_map.items() if k in selected_nodes}
        label_map = {k: v for k, v in label_map.items() if k in selected_nodes}
        node_ids = selected_nodes
        print(
            f"[内存保护] 节点数从 {original_num_nodes} 下采样到 {len(node_ids)}，"
            f"请用 --max-nodes 调整（0 为不限制）。"
        )

    if len(temporal_edges) == 0:
        raise ValueError("过滤后边集为空，请增大 --max-nodes 或检查 edges.csv。")

    sorted_nodes = sorted(node_ids)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(sorted_nodes)}
    num_nodes = len(sorted_nodes)

    if feature_dim == 0:
        feature_dim = 16
    attrs = np.zeros((num_nodes, feature_dim), dtype=np.float64)
    rng = np.random.default_rng(42)
    attrs[:] = 0.01 * rng.normal(size=attrs.shape)
    for node, vec in feature_map.items():
        if len(vec) != feature_dim:
            raise ValueError(f"节点 {node} 的特征维度为 {len(vec)}，与预期 {feature_dim} 不一致。")
        attrs[node_to_idx[node]] = vec

    labels = np.full(num_nodes, -1, dtype=np.int64)
    for node, label in label_map.items():
        labels[node_to_idx[node]] = label

    # ---- 1) 根据时间构造快照边集 ----
    temporal_edges.sort(key=lambda x: x[2])
    if snapshots < 2 or not has_time:
        snapshots = max(2, snapshots)
        # Without explicit timestamps, split edges by order.
        bins = np.linspace(0, len(temporal_edges), num=snapshots + 1, dtype=int)
        snapshot_edge_sets: List[set] = []
        for i in range(1, len(bins)):
            part = temporal_edges[bins[i - 1] : bins[i]]
            step_set = {(node_to_idx[s], node_to_idx[d]) if node_to_idx[s] < node_to_idx[d] else (node_to_idx[d], node_to_idx[s]) for s, d, _ in part}
            snapshot_edge_sets.append(step_set)
    else:
        times = np.array([e[2] for e in temporal_edges], dtype=np.float64)
        cuts = np.quantile(times, np.linspace(0, 1, snapshots + 1))
        snapshot_edge_sets = []
        for i in range(1, len(cuts)):
            left = cuts[i - 1]
            right = cuts[i]
            if i == len(cuts) - 1:
                seg = [e for e in temporal_edges if left <= e[2] <= right]
            else:
                seg = [e for e in temporal_edges if left <= e[2] < right]
            step_set = {(node_to_idx[s], node_to_idx[d]) if node_to_idx[s] < node_to_idx[d] else (node_to_idx[d], node_to_idx[s]) for s, d, _ in seg}
            snapshot_edge_sets.append(step_set)

    # cumulative 模式下，第 i 个快照等于前 i 个窗口边集并集。
    if snapshot_mode == "cumulative":
        cum = set()
        cumulative_sets = []
        for st in snapshot_edge_sets:
            cum = set(cum)
            cum.update(st)
            cumulative_sets.append(cum)
        snapshot_edge_sets = cumulative_sets

    # ---- 2) 将属性更新按时间分桶到对应快照 ----
    attr_update_bins: List[Dict[int, np.ndarray]] = [dict() for _ in range(len(snapshot_edge_sets))]
    if attr_updates_path:
        attr_rows = read_csv_dict(attr_updates_path)
        if len(attr_rows) > 0:
            required = {"time", "node_id"}
            if not required.issubset(set(attr_rows[0].keys())):
                raise ValueError("属性更新文件必须包含列: time,node_id,f1,f2,...")
            feat_cols = [c for c in attr_rows[0].keys() if c not in ("time", "node_id")]
            if len(feat_cols) == 0:
                raise ValueError("属性更新文件至少需要一个数值特征列。")
            if len(feat_cols) != feature_dim:
                raise ValueError("属性更新维度必须与 features 文件一致。")
            if has_time:
                boundaries = np.quantile(
                    np.array([e[2] for e in temporal_edges], dtype=np.float64),
                    np.linspace(0, 1, len(snapshot_edge_sets) + 1),
                )
            else:
                boundaries = np.linspace(0, 1, len(snapshot_edge_sets) + 1)
            for attr_row_idx, row in enumerate(attr_rows, start=2):
                node = row["node_id"].strip()
                if node not in node_to_idx:
                    continue
                t = parse_time_value(row["time"])
                vec = parse_numeric_vector(
                    row,
                    feat_cols,
                    context=f"attr_updates.csv 第 {attr_row_idx} 行 node_id={node}",
                )
                bin_idx = len(snapshot_edge_sets) - 1
                for i in range(len(snapshot_edge_sets)):
                    if boundaries[i] <= t <= boundaries[i + 1]:
                        bin_idx = i
                        break
                attr_update_bins[bin_idx][node_to_idx[node]] = vec

    # ---- 3) 生成初始图 + 每个快照的增量事件 ----
    first_edges = snapshot_edge_sets[0]
    row_idx: List[int] = []
    col_idx: List[int] = []
    for u, v in first_edges:
        row_idx.extend([u, v])
        col_idx.extend([v, u])
    data = np.ones(len(row_idx), dtype=np.float64)
    adj0 = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(num_nodes, num_nodes), dtype=np.float64)

    batches: List[DynamicBatch] = []
    prev_set = set(first_edges)
    for i in range(1, len(snapshot_edge_sets)):
        curr_set = snapshot_edge_sets[i]
        additions = sorted(curr_set - prev_set)
        removals = sorted(prev_set - curr_set)
        batches.append(
            DynamicBatch(
                edge_additions=additions,
                edge_removals=removals,
                attr_updates=attr_update_bins[i],
            )
        )
        prev_set = set(curr_set)

    if attrs.shape != (num_nodes, feature_dim):
        raise ValueError("attrs 矩阵形状非法。")
    if not np.all(np.isfinite(attrs)):
        raise ValueError("attrs 矩阵包含 NaN 或 Inf。")
    for batch_idx, batch in enumerate(batches, start=1):
        for node_idx, vec in batch.attr_updates.items():
            if node_idx < 0 or node_idx >= num_nodes:
                raise ValueError(f"第 {batch_idx} 个快照中的属性更新节点索引越界: {node_idx}")
            if vec.ndim != 1 or vec.shape[0] != feature_dim:
                raise ValueError(f"第 {batch_idx} 个快照中的属性更新维度非法。")
            if not np.all(np.isfinite(vec)):
                raise ValueError(f"第 {batch_idx} 个快照中的属性更新包含 NaN 或 Inf。")

    return adj0, attrs, labels, batches, node_to_idx


def build_synthetic_graph(
    num_nodes: int,
    num_classes: int,
    feature_dim: int,
    rounds: int,
    seed: int,
) -> Tuple[AdjLike, np.ndarray, np.ndarray, List[DynamicBatch], Dict[str, int]]:
    """构造可控的合成动态属性图，用于快速自检。"""
    rng = np.random.default_rng(seed)
    labels = np.arange(num_nodes) % num_classes
    rng.shuffle(labels)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            p = 0.16 if labels[i] == labels[j] else 0.02
            if rng.random() < p:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    centers = rng.normal(0.0, 1.0, size=(num_classes, feature_dim))
    attrs = np.vstack([centers[labels[i]] + 0.28 * rng.normal(size=feature_dim) for i in range(num_nodes)])

    batches: List[DynamicBatch] = []
    for _ in range(rounds):
        additions: List[Tuple[int, int]] = []
        removals: List[Tuple[int, int]] = []
        for _ in range(3):
            u, v = rng.choice(num_nodes, size=2, replace=False)
            if u > v:
                u, v = v, u
            if adj[u, v] > 0:
                adj[u, v] = 0.0
                adj[v, u] = 0.0
                removals.append((u, v))
            else:
                adj[u, v] = 1.0
                adj[v, u] = 1.0
                additions.append((u, v))

        attr_updates: Dict[int, np.ndarray] = {}
        for _ in range(2):
            idx = int(rng.integers(0, num_nodes))
            cls = labels[idx]
            attr_updates[idx] = centers[cls] + 0.22 * rng.normal(size=feature_dim)
        batches.append(DynamicBatch(additions, removals, attr_updates))

    # Rebuild initial adjacency for training start.
    # We replay from a fresh graph with same seed to keep first snapshot stable.
    rng = np.random.default_rng(seed)
    labels = np.arange(num_nodes) % num_classes
    rng.shuffle(labels)
    init_adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            p = 0.16 if labels[i] == labels[j] else 0.02
            if rng.random() < p:
                init_adj[i, j] = 1.0
                init_adj[j, i] = 1.0
    node_to_idx = {str(i): i for i in range(num_nodes)}
    return init_adj, attrs, labels.astype(np.int64), batches, node_to_idx


def save_metrics_curves_svg(metrics_rows: List[Dict[str, float]], output_path: str) -> None:
    """将指标曲线保存为 SVG（无第三方绘图库依赖）。"""
    xs = [int(r["snapshot"]) for r in metrics_rows]
    if len(xs) == 0:
        return

    # 上图：macro/micro/auc；下图：latency
    top_series = {
        "macro_f1": [float(r["macro_f1"]) for r in metrics_rows],
        "micro_f1": [float(r["micro_f1"]) for r in metrics_rows],
        "link_auc": [float(r["link_auc"]) for r in metrics_rows],
        "link_ap": [float(r["link_ap"]) for r in metrics_rows],
        "reconstruction_auc": [float(r["reconstruction_auc"]) for r in metrics_rows],
    }
    latency = [float(r["update_latency_ms"]) for r in metrics_rows]

    width, height = 1100, 680
    margin_l, margin_r = 70, 30
    margin_t, margin_b = 45, 40
    plot_w = width - margin_l - margin_r

    top_y0, top_h = margin_t, 260
    bot_y0, bot_h = top_y0 + top_h + 70, 220

    x_min, x_max = min(xs), max(xs)
    if x_max == x_min:
        x_max = x_min + 1

    def sx(x: float) -> float:
        return margin_l + (x - x_min) / (x_max - x_min) * plot_w

    top_min, top_max = 0.0, 1.0
    lat_min = 0.0
    lat_max = max(max(latency), 1.0) * 1.05

    def sy_top(y: float) -> float:
        return top_y0 + top_h - (y - top_min) / max(top_max - top_min, 1e-12) * top_h

    def sy_bot(y: float) -> float:
        return bot_y0 + bot_h - (y - lat_min) / max(lat_max - lat_min, 1e-12) * bot_h

    def polyline(points: List[Tuple[float, float]], color: str) -> str:
        pts = " ".join([f"{x:.2f},{y:.2f}" for x, y in points])
        return f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{pts}" />'

    svg_lines: List[str] = []
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    svg_lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    svg_lines.append('<text x="70" y="25" font-size="18" font-family="Arial">Embedding Metrics Curves</text>')

    # Axes
    svg_lines.append(f'<line x1="{margin_l}" y1="{top_y0}" x2="{margin_l}" y2="{top_y0 + top_h}" stroke="#333"/>')
    svg_lines.append(f'<line x1="{margin_l}" y1="{top_y0 + top_h}" x2="{margin_l + plot_w}" y2="{top_y0 + top_h}" stroke="#333"/>')
    svg_lines.append(f'<line x1="{margin_l}" y1="{bot_y0}" x2="{margin_l}" y2="{bot_y0 + bot_h}" stroke="#333"/>')
    svg_lines.append(f'<line x1="{margin_l}" y1="{bot_y0 + bot_h}" x2="{margin_l + plot_w}" y2="{bot_y0 + bot_h}" stroke="#333"/>')

    # Grid + ticks
    for i in range(6):
        yv = i / 5
        y = sy_top(yv)
        svg_lines.append(f'<line x1="{margin_l}" y1="{y:.2f}" x2="{margin_l + plot_w}" y2="{y:.2f}" stroke="#eee"/>')
        svg_lines.append(f'<text x="8" y="{y + 4:.2f}" font-size="11" font-family="Arial">{yv:.1f}</text>')

    for i in range(6):
        yv = lat_min + (lat_max - lat_min) * i / 5
        y = sy_bot(yv)
        svg_lines.append(f'<line x1="{margin_l}" y1="{y:.2f}" x2="{margin_l + plot_w}" y2="{y:.2f}" stroke="#eee"/>')
        svg_lines.append(f'<text x="8" y="{y + 4:.2f}" font-size="11" font-family="Arial">{yv:.0f}</text>')

    x_ticks = min(10, len(xs))
    for i in range(x_ticks):
        xv = x_min + (x_max - x_min) * i / max(x_ticks - 1, 1)
        xx = sx(xv)
        svg_lines.append(f'<line x1="{xx:.2f}" y1="{top_y0 + top_h}" x2="{xx:.2f}" y2="{bot_y0 + bot_h}" stroke="#f3f3f3"/>')
        svg_lines.append(f'<text x="{xx - 8:.2f}" y="{bot_y0 + bot_h + 18}" font-size="11" font-family="Arial">{int(round(xv))}</text>')

    # Curves
    colors = {
        "macro_f1": "#1f77b4",
        "micro_f1": "#2ca02c",
        "link_auc": "#d62728",
        "link_ap": "#ff7f0e",
        "reconstruction_auc": "#17becf",
    }
    for name, values in top_series.items():
        points = [(sx(x), sy_top(y)) for x, y in zip(xs, values)]
        svg_lines.append(polyline(points, colors[name]))

    lat_points = [(sx(x), sy_bot(y)) for x, y in zip(xs, latency)]
    svg_lines.append(polyline(lat_points, "#9467bd"))

    # Titles and legend
    svg_lines.append(f'<text x="{margin_l}" y="{top_y0 - 12}" font-size="13" font-family="Arial">F1/AUC over snapshots</text>')
    svg_lines.append(f'<text x="{margin_l}" y="{bot_y0 - 12}" font-size="13" font-family="Arial">Update latency (ms)</text>')

    legend_x = width - 280
    legend_y = 28
    legends = [
        ("macro_f1", "#1f77b4"),
        ("micro_f1", "#2ca02c"),
        ("link_auc", "#d62728"),
        ("link_ap", "#ff7f0e"),
        ("reconstruction_auc", "#17becf"),
        ("latency_ms", "#9467bd"),
    ]
    for i, (name, color) in enumerate(legends):
        yy = legend_y + i * 18
        svg_lines.append(f'<line x1="{legend_x}" y1="{yy}" x2="{legend_x + 22}" y2="{yy}" stroke="{color}" stroke-width="2.2"/>')
        svg_lines.append(f'<text x="{legend_x + 28}" y="{yy + 4}" font-size="12" font-family="Arial">{name}</text>')

    svg_lines.append("</svg>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))


_ALL_RESULTS_COLUMN_MAPPINGS = [
    ("dataset", ("dataset",)),
    ("variant", ("variant", "ablation_tag")),
    ("binary", ("binary", "binary_quantize")),
    ("update_rate", ("update_rate",)),
    ("nodes", ("nodes", "num_nodes")),
    ("init_ms", ("init_ms", "initialization_latency_ms")),
    ("update_ms", ("update_ms", "avg_update_latency_ms")),
    ("macro_f1", ("macro_f1", "final_macro_f1")),
    ("micro_f1", ("micro_f1", "final_micro_f1")),
    ("link_auc", ("link_auc", "final_link_auc")),
    ("link_ap", ("link_ap", "final_link_ap")),
    ("recon_auc", ("recon_auc", "final_reconstruction_auc")),
    ("compression_x", ("compression_x", "quantization_compression_ratio")),
]

_ALL_RESULTS_COLUMNS = [col for col, _ in _ALL_RESULTS_COLUMN_MAPPINGS]


def _pick_first_available(row: dict, keys: Sequence[str]) -> str:
    """按顺序读取首个可用字段值，用于兼容旧版汇总表。"""
    for key in keys:
        value = row.get(key, "")
        if value is None:
            continue
        text = str(value).strip()
        if text != "":
            return value
    return ""


def _append_to_all_results(
    project_root: str, dataset_tag: str, summary: dict
) -> None:
    """将本次实验结果追加到 all_results.csv。

    设计原则：
    1. 每次运行保留一行，避免同一数据集不同配置互相覆盖；
    2. 汇总表只保留“关键实验特征 + 关键指标”，便于阅读与论文整理；
    3. 若历史 all_results.csv 存在更多旧字段，则在重写时自动裁剪到当前精简字段集合。
    """
    csv_path = os.path.join(project_root, "all_results.csv")
    existing_rows: list = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(
                    {
                        col: _pick_first_available(row, aliases)
                        for col, aliases in _ALL_RESULTS_COLUMN_MAPPINGS
                    }
                )

    new_row = {
        "dataset": dataset_tag,
        "variant": summary.get("ablation_tag", ""),
        "binary": summary.get("binary_quantize", ""),
        "update_rate": summary.get("update_rate", ""),
        "nodes": summary.get("num_nodes", ""),
        "init_ms": summary.get("initialization_latency_ms", ""),
        "update_ms": summary.get("avg_update_latency_ms", ""),
        "macro_f1": summary.get("final_macro_f1", ""),
        "micro_f1": summary.get("final_micro_f1", ""),
        "link_auc": summary.get("final_link_auc", ""),
        "link_ap": summary.get("final_link_ap", ""),
        "recon_auc": summary.get("final_reconstruction_auc", ""),
        "compression_x": summary.get("quantization_compression_ratio", ""),
    }
    existing_rows.append(new_row)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_ALL_RESULTS_COLUMNS)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({col: row.get(col, "") for col in _ALL_RESULTS_COLUMNS})


def _build_run_tag(args: argparse.Namespace) -> str:
    """根据运行参数生成可读的目录名标签。"""
    if args.mode == "synthetic":
        return "synthetic"
    return "oag"


def _resolve_oag_dataset_paths(project_root: str) -> tuple[str, str, str, str]:
    """Resolve the fixed OAG dataset directory for file-mode experiments."""
    dataset_root = os.path.join(project_root, "data", "OAG")
    if not os.path.isdir(dataset_root):
        raise ValueError(
            f"file 模式固定使用 data/OAG，但未找到目录: {dataset_root}。"
            "请先将 OAG CSV 数据放到 data/OAG/ 下。"
        )

    edges_path = os.path.join(dataset_root, "edges.csv")
    features_path = os.path.join(dataset_root, "features.csv")
    labels_path = os.path.join(dataset_root, "labels.csv")
    attr_updates_path = os.path.join(dataset_root, "attr_updates.csv")
    required = [edges_path, features_path, labels_path]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"data/OAG 缺少必需文件: {missing_text}")
    return edges_path, features_path, labels_path, attr_updates_path if os.path.isfile(attr_updates_path) else ""


def _build_ablation_tag(args: argparse.Namespace) -> str:
    """根据开关生成消融标签。"""
    if args.model != "edane":
        return "w/o-Inc" if args.no_inc else "full"
    tags: list[str] = []
    if args.no_attr:
        tags.append("w/o-Attr")
    if args.no_hyperbolic:
        tags.append("w/o-Hyperbolic")
    if args.no_inc:
        tags.append("w/o-Inc")
    return "+".join(tags) if tags else "full"


def build_model(args: argparse.Namespace):
    """Construct embedding model from CLI args."""
    if args.model != "edane" and (args.no_attr or args.no_hyperbolic):
        raise ValueError("--no-attr and --no-hyperbolic are EDANE-specific ablations and are not supported for non-EDANE baselines.")
    if args.model == "edane":
        return EDANE(
            dim=args.dim,
            order=args.order,
            projection_density=args.projection_density,
            learning_rate=args.learning_rate,
            quantize=args.quantize,
            binary_quantize=args.binary_quantize,
            random_state=args.seed,
            use_attr_fusion=not args.no_attr,
            use_hyperbolic_fusion=not args.no_hyperbolic,
            backend=args.backend,
        )
    if args.model == "dane":
        return DANE(
            dim=args.dim,
            attr_topk=args.dane_attr_topk,
            similarity_block_size=args.dane_similarity_block_size,
            perturbation_rank=args.dane_perturbation_rank,
            random_state=args.seed,
        )
    return DTFormer(
        dim=args.dim,
        patch_size=args.dtformer_patch_size,
        history_snapshots=args.dtformer_history_snapshots,
        transformer_hidden_dim=args.dtformer_hidden_dim,
        attention_temperature=args.dtformer_attention_temperature,
        random_state=args.seed,
    )


def _quantization_enabled_for_model(args: argparse.Namespace) -> tuple[bool, bool]:
    """Return effective quantization flags for the selected model."""
    if args.model in {"dane", "dtformer"}:
        return False, False
    return bool(args.quantize), bool(args.binary_quantize)


def _count_batch_events(batch: DynamicBatch) -> int:
    """统计一个快照批次中的事件数（边变化 + 属性变化）。"""
    return int(len(batch.edge_additions) + len(batch.edge_removals) + len(batch.attr_updates))


def _apply_batch_to_graph(
    adj: sparse.csr_matrix,
    attrs: np.ndarray,
    batch: DynamicBatch,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """将一个动态批次作用到原始图与属性（用于 w/o-Inc 全量重训练）。"""
    work = adj.tolil(copy=True)
    for u, v in batch.edge_additions:
        if u == v:
            continue
        work[u, v] = 1.0
        work[v, u] = 1.0
    for u, v in batch.edge_removals:
        if u == v:
            continue
        work[u, v] = 0.0
        work[v, u] = 0.0

    next_adj = sparse.csr_matrix(work, dtype=np.float64)
    next_adj.eliminate_zeros()
    if next_adj.nnz > 0:
        next_adj.data[:] = 1.0

    next_attrs = attrs.copy()
    for idx, value in batch.attr_updates.items():
        if 0 <= idx < next_attrs.shape[0]:
            next_attrs[idx] = value
    return next_adj, next_attrs


def run_pipeline(args: argparse.Namespace) -> None:
    """主流程入口：读数、训练、更新、评估、导出。"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = _build_run_tag(args)
    out_dir = args.output_dir or os.path.join(project_root, "outputs", f"{tag}_{timestamp}")
    ensure_dir(out_dir)

    if args.mode == "file":
        args.edges_path, args.features_path, args.labels_path, args.attr_updates_path = _resolve_oag_dataset_paths(project_root)
        adj, attrs, labels, batches, node_to_idx = build_graph_from_files(
            edges_path=args.edges_path,
            features_path=args.features_path,
            labels_path=args.labels_path,
            attr_updates_path=args.attr_updates_path,
            snapshots=args.snapshots,
            snapshot_mode=args.snapshot_mode,
            max_nodes=args.max_nodes,
        )
    else:
        adj, attrs, labels, batches, node_to_idx = build_synthetic_graph(
            num_nodes=args.synthetic_nodes,
            num_classes=args.synthetic_classes,
            feature_dim=args.synthetic_feat_dim,
            rounds=args.synthetic_rounds,
            seed=args.seed,
        )

    # 初始化嵌入模型。
    model = build_model(args)

    start = time.perf_counter()
    model.fit(adj, attrs)
    init_latency_ms = (time.perf_counter() - start) * 1000.0
    emb = model.get_embedding(dequantize=False)
    metrics0 = evaluate_snapshot(
        emb,
        labels,
        model.adj if model.adj is not None else adj,
        seed=args.seed + 1,
        classifier=args.classifier,
        logreg_epochs=args.logreg_epochs,
        logreg_lr=args.logreg_lr,
        logreg_weight_decay=args.logreg_weight_decay,
        eval_protocol=args.eval_protocol,
        eval_repeats=args.eval_repeats,
        eval_train_ratio=args.eval_train_ratio,
        label_cleanup_mode=args.label_cleanup_mode,
        min_class_support=args.min_class_support,
        logreg_class_weight=args.logreg_class_weight,
    )
    metrics_rows = [
        {
            "snapshot": 0,
            "model": args.model,
            "update_latency_ms": init_latency_ms,
            **metrics0,
        }
    ]

    update_latencies = []
    compute_update_latencies = []
    pacing_waits = []
    total_events = 0
    total_update_wall_s = 0.0
    state_adj = sparse.csr_matrix(adj, dtype=np.float64)
    state_attrs = np.asarray(attrs, dtype=np.float64).copy()
    for i, batch in enumerate(batches, start=1):
        start = time.perf_counter()
        if args.no_inc:
            state_adj, state_attrs = _apply_batch_to_graph(state_adj, state_attrs, batch)
            model.fit(state_adj, state_attrs)
        else:
            model.apply_updates(
                edge_additions=batch.edge_additions,
                edge_removals=batch.edge_removals,
                attr_updates=batch.attr_updates if len(batch.attr_updates) > 0 else None,
            )
        compute_latency_s = time.perf_counter() - start
        latency_s = compute_latency_s

        wait_ms = 0.0
        event_count = _count_batch_events(batch)
        if args.update_rate > 0 and event_count > 0:
            target_s = event_count / float(args.update_rate)
            if latency_s < target_s:
                wait_ms = (target_s - latency_s) * 1000.0
                time.sleep(target_s - latency_s)
                latency_s = target_s

        latency_ms = latency_s * 1000.0
        compute_latency_ms = compute_latency_s * 1000.0
        update_latencies.append(latency_ms)
        compute_update_latencies.append(compute_latency_ms)
        pacing_waits.append(wait_ms)
        total_events += event_count
        total_update_wall_s += latency_s

        emb = model.get_embedding(dequantize=False)
        metrics = evaluate_snapshot(
            emb,
            labels,
            model.adj if model.adj is not None else adj,
            seed=args.seed + 100 + i,
            classifier=args.classifier,
            logreg_epochs=args.logreg_epochs,
            logreg_lr=args.logreg_lr,
            logreg_weight_decay=args.logreg_weight_decay,
            eval_protocol=args.eval_protocol,
            eval_repeats=args.eval_repeats,
            eval_train_ratio=args.eval_train_ratio,
            label_cleanup_mode=args.label_cleanup_mode,
            min_class_support=args.min_class_support,
            logreg_class_weight=args.logreg_class_weight,
        )
        metrics_rows.append(
                {
                    "snapshot": i,
                    "model": args.model,
                    "update_latency_ms": latency_ms,
                    "compute_update_latency_ms": compute_latency_ms,
                    "simulated_wait_ms": wait_ms,
                    "batch_events": event_count,
                    **metrics,
                }
            )

    # ---- 结果导出 ----
    final_embedding = model.get_embedding(dequantize=False)
    np.save(os.path.join(out_dir, "final_embedding.npy"), final_embedding)
    if model.quantized_embedding_ is not None:
        np.save(os.path.join(out_dir, "final_embedding_int8.npy"), model.quantized_embedding_.values)
        np.save(os.path.join(out_dir, "final_embedding_scale.npy"), model.quantized_embedding_.scale)
    binary_embedding = getattr(model, "binary_embedding_", None)
    if binary_embedding is not None:
        np.save(os.path.join(out_dir, "final_embedding_binary.npy"), binary_embedding.values)

    with open(os.path.join(out_dir, "metrics_per_snapshot.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snapshot",
                "model",
                "update_latency_ms",
                "compute_update_latency_ms",
                "simulated_wait_ms",
                "batch_events",
                "macro_f1",
                "macro_f1_std",
                "micro_f1",
                "micro_f1_std",
                "link_auc",
                "link_ap",
                "reconstruction_auc",
                "labeled_nodes_raw",
                "labeled_nodes",
                "eval_dropped_labeled_nodes",
                "eval_class_count_raw",
                "eval_class_count",
                "eval_dropped_class_count",
                "label_cleanup_mode",
                "min_class_support",
                "f1_eval_repeats",
                "f1_eval_successful_repeats",
                "f1_eval_protocol",
            ],
        )
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)

    with open(os.path.join(out_dir, "node_mapping.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "index"])
        for node_id, idx in sorted(node_to_idx.items(), key=lambda x: x[1]):
            writer.writerow([node_id, idx])

    compression_ratio = float(getattr(model, "quantization_compression_ratio_", 1.0))

    effective_quantize, effective_binary_quantize = _quantization_enabled_for_model(args)

    summary = {
        "mode": args.mode,
        "dataset": "OAG" if args.mode == "file" else "synthetic",
        "dataset_source_url": "https://open.aminer.cn/open/article?id=67aaf63af4cbd12984b6a5f0" if args.mode == "file" else "",
        "model": args.model,
        "implementation_fidelity": "native" if args.model == "edane" else "paper_approximation",
        "snapshot_mode": args.snapshot_mode,
        "seed": int(args.seed),
        "classifier": args.classifier,
        "logreg_class_weight": args.logreg_class_weight,
        "f1_eval_protocol": args.eval_protocol,
        "f1_eval_protocol_used": str(metrics_rows[-1]["f1_eval_protocol"]),
        "f1_eval_repeats": int(args.eval_repeats if args.eval_protocol == "repeated_stratified" else 1),
        "f1_eval_successful_repeats": int(metrics_rows[-1]["f1_eval_successful_repeats"]),
        "f1_eval_train_ratio": float(args.eval_train_ratio),
        "label_cleanup_mode": args.label_cleanup_mode,
        "min_class_support": int(args.min_class_support),
        "final_labeled_nodes_raw": float(metrics_rows[-1]["labeled_nodes_raw"]),
        "final_labeled_nodes": float(metrics_rows[-1]["labeled_nodes"]),
        "final_eval_dropped_labeled_nodes": float(metrics_rows[-1]["eval_dropped_labeled_nodes"]),
        "final_eval_class_count_raw": float(metrics_rows[-1]["eval_class_count_raw"]),
        "final_eval_class_count": float(metrics_rows[-1]["eval_class_count"]),
        "final_eval_dropped_class_count": float(metrics_rows[-1]["eval_dropped_class_count"]),
        "backend": args.backend,
        "supports_incremental_updates": bool(getattr(model, "supports_incremental_updates_", False)),
        "online_update_mode": str(getattr(model, "online_update_mode_", "unknown")),
        "quantize": effective_quantize,
        "binary_quantize": effective_binary_quantize,
        "num_nodes": int(final_embedding.shape[0]),
        "feature_dim": int(attrs.shape[1]),
        "embedding_dim": int(final_embedding.shape[1]),
        "num_snapshots": int(len(metrics_rows)),
        "initialization_latency_ms": float(init_latency_ms),
        "avg_update_latency_ms": float(np.mean(update_latencies)) if update_latencies else 0.0,
        "avg_compute_update_latency_ms": float(np.mean(compute_update_latencies)) if compute_update_latencies else 0.0,
        "avg_pacing_wait_ms": float(np.mean(pacing_waits)) if pacing_waits else 0.0,
        "p95_update_latency_ms": float(np.percentile(update_latencies, 95)) if update_latencies else 0.0,
        "final_macro_f1": float(metrics_rows[-1]["macro_f1"]),
        "final_macro_f1_std": float(metrics_rows[-1]["macro_f1_std"]),
        "final_micro_f1": float(metrics_rows[-1]["micro_f1"]),
        "final_micro_f1_std": float(metrics_rows[-1]["micro_f1_std"]),
        "final_link_auc": float(metrics_rows[-1]["link_auc"]),
        "final_link_ap": float(metrics_rows[-1]["link_ap"]),
        "final_reconstruction_auc": float(metrics_rows[-1]["reconstruction_auc"]),
        "ablation_tag": _build_ablation_tag(args),
        "update_rate": int(args.update_rate),
        "effective_update_rate": float(total_events / total_update_wall_s) if total_update_wall_s > 0 else 0.0,
        "quantization_compression_ratio": float(compression_ratio),
        "quantization_error": float(getattr(model, "quantization_error_", 0.0)),
        "binary_compression_ratio": float(getattr(model, "binary_compression_ratio_", 1.0)),
        "binary_error": float(getattr(model, "binary_error_", 0.0)),
        "output_dir": out_dir,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 导出曲线图，便于论文直接引用。
    curves_path = os.path.join(out_dir, "metrics_curves.svg")
    save_metrics_curves_svg(metrics_rows, curves_path)

    # 自动追加到项目根目录的 all_results.csv 汇总表。
    _append_to_all_results(project_root, tag, summary)

    print(f"{args.model.upper()} 全流程实验完成")
    print("=" * 50)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("=" * 50)
    print(f"指标明细: {os.path.join(out_dir, 'metrics_per_snapshot.csv')}")
    print(f"曲线图: {curves_path}")
    print(f"汇总表: {os.path.join(project_root, 'all_results.csv')}")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="动态图嵌入端到端实验流水线")
    parser.add_argument("--mode", choices=["synthetic", "file"], default="synthetic")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--model", choices=["edane", "dane", "dtformer"], default="edane")
    parser.add_argument("--snapshots", type=int, default=8)
    parser.add_argument("--snapshot-mode", choices=["window", "cumulative"], default="window")
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=10000,
        help="file 模式最大节点数（默认 10000；0 表示不限制，可能导致内存溢出）",
    )

    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--projection-density", type=float, default=0.12)
    parser.add_argument("--learning-rate", type=float, default=0.55)
    parser.add_argument("--dane-attr-topk", type=int, default=20)
    parser.add_argument("--dane-similarity-block-size", type=int, default=512)
    parser.add_argument("--dane-perturbation-rank", type=int, default=64)
    parser.add_argument("--dtformer-patch-size", type=int, default=2)
    parser.add_argument("--dtformer-history-snapshots", type=int, default=8)
    parser.add_argument("--dtformer-hidden-dim", type=int, default=96)
    parser.add_argument("--dtformer-attention-temperature", type=float, default=1.0)
    parser.add_argument("--update-rate", type=int, default=0, help="目标变化速率（次/秒）；0 表示不进行速率控制")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--binary-quantize", action="store_true")
    parser.add_argument("--no-attr", action="store_true", help="消融：禁用属性融合（w/o-Attr）")
    parser.add_argument("--no-hyperbolic", action="store_true", help="消融：禁用双曲融合，改为欧氏门控融合（w/o-Hyperbolic）")
    parser.add_argument("--no-inc", action="store_true", help="消融：禁用增量更新，改为每快照全量重训练（w/o-Inc）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier", choices=["centroid", "logreg"], default="logreg")
    parser.add_argument(
        "--eval-protocol",
        choices=["single_random", "repeated_stratified"],
        default="repeated_stratified",
        help="节点分类评估协议：单次随机切分或重复分层切分",
    )
    parser.add_argument("--eval-repeats", type=int, default=10, help="重复分层评估次数（single_random 模式下忽略）")
    parser.add_argument("--eval-train-ratio", type=float, default=0.7, help="分类评估训练集比例")
    parser.add_argument(
        "--label-cleanup-mode",
        choices=["off", "eval_only"],
        default="off",
        help="评估前标签清洗模式：off 保持原始标签，eval_only 仅在 F1 评估时过滤低支持类别",
    )
    parser.add_argument("--min-class-support", type=int, default=5, help="评估时保留类别的最小样本数（建议 >= 2）")
    parser.add_argument(
        "--backend",
        choices=["numpy", "torch"],
        default="numpy",
        help="dense 数值后端；torch 模式仍保留 SciPy 稀疏图操作",
    )
    parser.add_argument("--logreg-epochs", type=int, default=260)
    parser.add_argument("--logreg-lr", type=float, default=0.35)
    parser.add_argument("--logreg-weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--logreg-class-weight",
        choices=["none", "balanced"],
        default="none",
        help="逻辑回归分类头的类别权重策略；balanced 按训练折类别频次做反比加权",
    )

    parser.add_argument("--synthetic-nodes", type=int, default=600)
    parser.add_argument("--synthetic-classes", type=int, default=6)
    parser.add_argument("--synthetic-feat-dim", type=int, default=24)
    parser.add_argument("--synthetic-rounds", type=int, default=50)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_pipeline(args)
