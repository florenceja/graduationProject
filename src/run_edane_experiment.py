"""EDANE 轻量实验脚本（单文件版本）。

用途：
- 快速演示算法在合成动态属性图上的完整流程
- 输出初始化时延、增量更新时延、分类 / 链路预测 / 网络重构指标
- 便于在没有真实数据时先做功能自检
"""

import argparse
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from edane import EDANE, cosine_scores


def generate_synthetic_dynamic_graph(
    num_nodes: int = 240,
    num_classes: int = 4,
    feature_dim: int = 16,
    intra_prob: float = 0.18,
    inter_prob: float = 0.03,
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成带社区结构的合成图 + 属性 + 标签。"""
    rng = np.random.default_rng(seed)
    labels = np.arange(num_nodes) % num_classes
    rng.shuffle(labels)

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            p = intra_prob if labels[i] == labels[j] else inter_prob
            if rng.random() < p:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    centroids = rng.normal(0.0, 1.0, size=(num_classes, feature_dim))
    attrs = np.vstack(
        [centroids[label] + 0.30 * rng.normal(size=feature_dim) for label in labels]
    )
    return adj, attrs, labels


def train_test_split(num_nodes: int, train_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """随机划分训练/测试索引。"""
    rng = np.random.default_rng(seed)
    indices = np.arange(num_nodes)
    rng.shuffle(indices)
    split = int(num_nodes * train_ratio)
    return indices[:split], indices[split:]


def macro_micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """手工计算 Macro-F1 和 Micro-F1。"""
    labels = np.unique(y_true)
    f1_values: List[float] = []
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
            f1_values.append(0.0)
        else:
            f1_values.append(2.0 * precision * recall / (precision + recall))

    macro = float(np.mean(f1_values))
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
    train_idx: Sequence[int],
    test_idx: Sequence[int],
) -> np.ndarray:
    """最近类中心分类器（轻量评估，不依赖 sklearn）。"""
    classes = np.unique(labels)
    centroids = []
    for cls in classes:
        cls_idx = [idx for idx in train_idx if labels[idx] == cls]
        centroids.append(embedding[cls_idx].mean(axis=0))
    centroids = np.vstack(centroids)
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)

    test_emb = embedding[list(test_idx)]
    test_emb /= np.maximum(np.linalg.norm(test_emb, axis=1, keepdims=True), 1e-12)
    scores = test_emb @ centroids.T
    return classes[np.argmax(scores, axis=1)]


def sample_link_pairs(adj: np.ndarray, sample_size: int, seed: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """采样正边与负边节点对。"""
    rng = np.random.default_rng(seed)
    pos = np.argwhere(np.triu(adj, 1) > 0.0)
    neg = np.argwhere(np.triu(1.0 - adj - np.eye(adj.shape[0]), 1) > 0.0)

    pos_idx = rng.choice(len(pos), size=min(sample_size, len(pos)), replace=False)
    neg_idx = rng.choice(len(neg), size=min(sample_size, len(neg)), replace=False)

    pos_pairs = [(int(pair[0]), int(pair[1])) for pair in pos[pos_idx]]
    neg_pairs = [(int(pair[0]), int(pair[1])) for pair in neg[neg_idx]]
    return pos_pairs, neg_pairs


def auc_from_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """根据正负样本打分计算 AUC。"""
    comparisons = (pos_scores[:, None] > neg_scores[None, :]).astype(np.float64)
    ties = (pos_scores[:, None] == neg_scores[None, :]).astype(np.float64) * 0.5
    return float(np.mean(comparisons + ties))


def average_precision_from_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """根据正负样本得分计算 Average Precision。"""
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0
    y_true = np.concatenate([np.ones(len(pos_scores), dtype=np.int64), np.zeros(len(neg_scores), dtype=np.int64)])
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


def evaluate_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    adj: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    """在当前快照上评估分类、链路预测与网络重构效果。"""
    train_idx, test_idx = train_test_split(len(labels), train_ratio=0.6, seed=seed)
    pred = nearest_centroid_predict(embedding.copy(), labels, train_idx.tolist(), test_idx.tolist())
    macro_f1, micro_f1 = macro_micro_f1(labels[test_idx], pred)

    pos_pairs, neg_pairs = sample_link_pairs(adj, sample_size=512, seed=seed)
    pos_scores = cosine_scores(embedding, pos_pairs)
    neg_scores = cosine_scores(embedding, neg_pairs)
    auc = auc_from_scores(pos_scores, neg_scores)
    ap = average_precision_from_scores(pos_scores, neg_scores)

    recon_pos_pairs, recon_neg_pairs = sample_link_pairs(adj, sample_size=1024, seed=seed + 1000)
    recon_pos_scores = cosine_scores(embedding, recon_pos_pairs)
    recon_neg_scores = cosine_scores(embedding, recon_neg_pairs)
    reconstruction_auc = auc_from_scores(recon_pos_scores, recon_neg_scores)

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "link_auc": auc,
        "link_ap": ap,
        "reconstruction_auc": reconstruction_auc,
    }


def generate_dynamic_events(
    adj: np.ndarray,
    attrs: np.ndarray,
    labels: np.ndarray,
    rounds: int = 50,
    seed: int = 9,
) -> Tuple[List[List[Tuple[int, int]]], List[Dict[int, np.ndarray]]]:
    """生成动态事件流：边翻转 + 属性漂移。"""
    rng = np.random.default_rng(seed)
    edge_batches: List[List[Tuple[int, int]]] = []
    attr_batches: List[Dict[int, np.ndarray]] = []

    class_centers = []
    for cls in np.unique(labels):
        class_centers.append(attrs[labels == cls].mean(axis=0))
    class_centers = np.vstack(class_centers)

    for _ in range(rounds):
        batch_edges = []
        for _ in range(3):
            u, v = rng.choice(len(labels), size=2, replace=False)
            batch_edges.append((int(u), int(v)))
        edge_batches.append(batch_edges)

        attr_update: Dict[int, np.ndarray] = {}
        for _ in range(2):
            idx = int(rng.integers(0, len(labels)))
            cls = labels[idx]
            attr_update[idx] = class_centers[cls] + 0.25 * rng.normal(size=attrs.shape[1])
        attr_batches.append(attr_update)

    return edge_batches, attr_batches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EDANE 轻量实验脚本")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    return parser


def main() -> None:
    """实验主入口。"""
    args = build_parser().parse_args()
    adj, attrs, labels = generate_synthetic_dynamic_graph()

    model = EDANE(
        dim=32,
        order=2,
        projection_density=0.12,
        learning_rate=0.55,
        quantize=True,
        random_state=42,
        backend=args.backend,
    )

    start = time.perf_counter()
    model.fit(adj, attrs)
    init_latency_ms = (time.perf_counter() - start) * 1000.0

    embedding = model.get_embedding(dequantize=False)
    initial_metrics = evaluate_embedding(embedding, labels, adj, seed=1)

    edge_batches, attr_batches = generate_dynamic_events(adj, attrs, labels)
    dynamic_adj = adj.copy()
    update_times = []

    for edge_batch, attr_batch in zip(edge_batches, attr_batches):
        # 先修改评估图，再触发模型增量更新。
        edge_additions = []
        edge_removals = []
        for u, v in edge_batch:
            if dynamic_adj[u, v] > 0.0:
                dynamic_adj[u, v] = 0.0
                dynamic_adj[v, u] = 0.0
                edge_removals.append((u, v))
            else:
                dynamic_adj[u, v] = 1.0
                dynamic_adj[v, u] = 1.0
                edge_additions.append((u, v))

        start = time.perf_counter()
        model.apply_updates(
            edge_additions=edge_additions,
            edge_removals=edge_removals,
            attr_updates=attr_batch,
        )
        update_times.append((time.perf_counter() - start) * 1000.0)

    updated_embedding = model.get_embedding(dequantize=False)
    updated_metrics = evaluate_embedding(updated_embedding, labels, dynamic_adj, seed=2)

    print("EDANE experiment summary")
    print("=" * 40)
    print(f"nodes: {adj.shape[0]}")
    print(f"feature_dim: {attrs.shape[1]}")
    print(f"embedding_dim: {updated_embedding.shape[1]}")
    print(f"backend: {args.backend}")
    print(f"initialization_latency_ms: {init_latency_ms:.3f}")
    print(f"avg_update_latency_ms: {np.mean(update_times):.3f}")
    print(f"p95_update_latency_ms: {np.percentile(update_times, 95):.3f}")
    print("-" * 40)
    print("initial metrics")
    for key, value in initial_metrics.items():
        print(f"{key}: {value:.4f}")
    print("-" * 40)
    print("updated metrics")
    for key, value in updated_metrics.items():
        print(f"{key}: {value:.4f}")
    print("-" * 40)
    print(f"quantization_compression_ratio: {getattr(model, 'quantization_compression_ratio_', 1.0):.3f}x")
    print(f"quantization_error: {getattr(model, 'quantization_error_', 0.0):.6f}")


if __name__ == "__main__":
    main()
