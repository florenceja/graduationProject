"""数据整理脚本：将原始大数据集转换为 EDANE 实验输入格式。

输出到项目根目录下的 data/<preset>/

已支持的预设：
- reddit_sample    — Reddit 社交网络子图
- amazon2m_sample  — Amazon-2M 商品网络子图
- amazon3m_sample  — Amazon-3M 商品标签共现图
- mag_sample       — MAG 学术引用网络子图
- twitter_sample   — Twitter 社交网络子图

输出文件规范：
- edges.csv  (src, dst, time)
- features.csv  (node_id, f1...fN)
- labels.csv  (node_id, label)
- attr_updates.csv  (可选)
"""

import argparse
import csv
import gzip
import json
import os
from typing import Dict, List, Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    """若目录不存在则创建。"""
    os.makedirs(path, exist_ok=True)


def source_root() -> str:
    """原始数据集根目录。"""
    return "D:/" + "".join(map(chr, [0x6bd5, 0x8bbe, 0x8d44, 0x6599])) + "/dataset"


def project_root() -> str:
    """当前项目根目录（src/ 的上一级）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_csv(path: str, header: List[str], rows: List[List[object]]) -> None:
    """统一 CSV 写入封装。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _safe_node_id(raw: object) -> str:
    """将任意节点 ID 规范化为字符串。"""
    return str(raw).strip()


def prepare_reddit_sample(max_nodes: int = 15000, seed: int = 42) -> str:
    """从 Reddit 原始文件构建可直接实验的样本集。

    处理逻辑：
    1) 使用 reddit-id_map.json 确定节点与特征数组行号映射
    2) 截取前 max_nodes 个节点构造子图
    3) 导出 features / labels / edges
    4) 额外合成一份 attr_updates 便于动态实验
    """
    rng = np.random.default_rng(seed)
    src = os.path.join(source_root(), "Reddit", "reddit")
    out_dir = os.path.join(project_root(), "data", "reddit_sample")
    ensure_dir(out_dir)

    id_map_path = os.path.join(src, "reddit-id_map.json")
    class_map_path = os.path.join(src, "reddit-class_map.json")
    graph_path = os.path.join(src, "reddit-G.json")
    feats_path = os.path.join(src, "reddit-feats.npy")

    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map_raw = json.load(f)
    id_map = {k: int(v) for k, v in id_map_raw.items()}
    node_ids = sorted(id_map.keys(), key=lambda x: id_map[x])
    selected_nodes = node_ids[: max_nodes]
    selected_index = {node_id: i for i, node_id in enumerate(selected_nodes)}
    old_index_to_new = {id_map[node_id]: selected_index[node_id] for node_id in selected_nodes}

    feats = np.load(feats_path, mmap_mode="r")
    feat_dim = feats.shape[1]
    feature_rows: List[List[object]] = []
    for node_id in selected_nodes:
        old_idx = id_map[node_id]
        vec = feats[old_idx]
        feature_rows.append([node_id] + [float(v) for v in vec.tolist()])

    with open(class_map_path, "r", encoding="utf-8") as f:
        class_map_raw = json.load(f)
    label_rows: List[List[object]] = []
    for node_id in selected_nodes:
        if node_id not in class_map_raw:
            continue
        label_val = class_map_raw[node_id]
        if isinstance(label_val, list):
            arr = np.asarray(label_val, dtype=np.float64)
            label = int(np.argmax(arr))
        else:
            label = int(label_val)
        label_rows.append([node_id, label])

    with open(graph_path, "r", encoding="utf-8") as f:
        graph_obj = json.load(f)
    links = graph_obj.get("links", [])
    edge_rows: List[List[object]] = []
    seen = set()
    for link in links:
        s_old = int(link["source"])
        d_old = int(link["target"])
        if s_old not in old_index_to_new or d_old not in old_index_to_new:
            continue
        s = selected_nodes[old_index_to_new[s_old]]
        d = selected_nodes[old_index_to_new[d_old]]
        if s == d:
            continue
        a, b = (s, d) if s < d else (d, s)
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        t = int(rng.integers(1710000000, 1715000000))
        edge_rows.append([a, b, t])

    rng.shuffle(edge_rows)

    feature_header = ["node_id"] + [f"f{i+1}" for i in range(feat_dim)]
    write_csv(os.path.join(out_dir, "features.csv"), feature_header, feature_rows)
    write_csv(os.path.join(out_dir, "labels.csv"), ["node_id", "label"], label_rows)
    write_csv(os.path.join(out_dir, "edges.csv"), ["src", "dst", "time"], edge_rows)

    # 生成可控的属性更新样本，模拟动态属性漂移。
    attr_updates: List[List[object]] = []
    label_map = {row[0]: int(row[1]) for row in label_rows}
    if label_map:
        groups: Dict[int, List[str]] = {}
        for node_id, lab in label_map.items():
            groups.setdefault(lab, []).append(node_id)
        for _ in range(3000):
            lab = int(rng.choice(list(groups.keys())))
            node_id = str(rng.choice(groups[lab]))
            row_idx = selected_index[node_id]
            base = np.asarray(feature_rows[row_idx][1:], dtype=np.float64)
            noise = rng.normal(0.0, 0.03, size=feat_dim)
            new_vec = base + noise
            t = int(rng.integers(1710000000, 1715000000))
            attr_updates.append([t, node_id] + [float(v) for v in new_vec.tolist()])
    write_csv(
        os.path.join(out_dir, "attr_updates.csv"),
        ["time", "node_id"] + [f"f{i+1}" for i in range(feat_dim)],
        attr_updates,
    )
    return out_dir


def prepare_amazon2m_sample(max_nodes: int = 30000, max_edges: int = 250000, seed: int = 42) -> str:
    """从 Amazon2M 构建可直接实验的样本集。

    原始邻接为 CSR（indices/indptr），此处会抽取前 max_nodes 节点，
    并限制最多 max_edges 条边，减少实验机内存与运行时间压力。
    """
    rng = np.random.default_rng(seed)
    src = os.path.join(source_root(), "Amazon2M")
    out_dir = os.path.join(project_root(), "data", "amazon2m_sample")
    ensure_dir(out_dir)

    feat = np.load(os.path.join(src, "Amazon2M_feat.npy"), mmap_mode="r")
    labels_raw = np.load(os.path.join(src, "Amazon2M_labels.npy"), mmap_mode="r")
    csr = np.load(os.path.join(src, "Amazon2M_adj.npz"))
    indices = csr["indices"]
    indptr = csr["indptr"]
    shape = tuple(csr["shape"].tolist())
    num_nodes = min(max_nodes, shape[0], feat.shape[0], labels_raw.shape[0])

    feature_dim = feat.shape[1]
    feature_rows: List[List[object]] = []
    label_rows: List[List[object]] = []
    for i in range(num_nodes):
        node_id = f"a{i}"
        feature_rows.append([node_id] + [float(v) for v in feat[i].tolist()])
        lab_vec = np.asarray(labels_raw[i], dtype=np.float64)
        label_rows.append([node_id, int(np.argmax(lab_vec))])

    edge_rows: List[List[object]] = []
    seen = set()
    for i in range(num_nodes):
        st = int(indptr[i])
        ed = int(indptr[i + 1])
        neigh = indices[st:ed]
        for j in neigh:
            j = int(j)
            if j >= num_nodes or i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            t = int(rng.integers(1710000000, 1715000000))
            edge_rows.append([f"a{a}", f"a{b}", t])
            if len(edge_rows) >= max_edges:
                break
        if len(edge_rows) >= max_edges:
            break

    rng.shuffle(edge_rows)
    feature_header = ["node_id"] + [f"f{i+1}" for i in range(feature_dim)]
    write_csv(os.path.join(out_dir, "features.csv"), feature_header, feature_rows)
    write_csv(os.path.join(out_dir, "labels.csv"), ["node_id", "label"], label_rows)
    write_csv(os.path.join(out_dir, "edges.csv"), ["src", "dst", "time"], edge_rows)
    return out_dir


def prepare_mag_sample(max_nodes: int = 8000, max_edges: int = 200000, seed: int = 42) -> str:
    """从 MAG Scholar C 构建样本集。

    说明：
    - 原始 MAG 文件非常大（10M+ 节点），直接用全量会超出普通实验机容量。
    - 这里抽取前 max_nodes 个节点，并限制最多 max_edges 条边。
    - 为避免加载超大 attr_matrix，特征使用结构统计特征（度、对数度等）构造。
    """
    rng = np.random.default_rng(seed)
    src = os.path.join(source_root(), "MAG-", "mag_scholar_c.npz")
    out_dir = os.path.join(project_root(), "data", "mag_sample")
    ensure_dir(out_dir)

    npz = np.load(src, allow_pickle=True)
    indptr = npz["adj_matrix.indptr"]
    indices = npz["adj_matrix.indices"]
    shape = npz["adj_matrix.shape"]
    labels = npz["labels"]

    total_nodes = int(shape[0])
    label_nodes = len(labels)
    total_nodes = min(total_nodes, label_nodes)

    # 先挑选“有边且相对连通”的节点，避免前缀节点过稀疏导致子图退化。
    selected_old: List[int] = []
    selected_set = set()
    for i in range(total_nodes):
        if len(selected_old) >= max_nodes:
            break
        st = int(indptr[i])
        ed = int(indptr[i + 1])
        if ed <= st:
            continue
        if i not in selected_set:
            selected_set.add(i)
            selected_old.append(i)
        neigh = indices[st:ed]
        # 从邻居中补齐节点，提升子图连通性。
        for j in neigh[:32]:
            if len(selected_old) >= max_nodes:
                break
            j = int(j)
            if j < 0 or j >= total_nodes:
                continue
            if j not in selected_set:
                selected_set.add(j)
                selected_old.append(j)

    if len(selected_old) == 0:
        raise ValueError("MAG 抽样失败：未采到可用节点。")

    selected_old = selected_old[:max_nodes]
    num_nodes = len(selected_old)
    old_to_new = {old: new for new, old in enumerate(selected_old)}
    node_ids = [f"m{i}" for i in range(num_nodes)]

    # 构造边列表（仅保留采样子图内部边）。
    edge_rows: List[List[object]] = []
    seen = set()
    degree = np.zeros(num_nodes, dtype=np.int64)
    for old_i in selected_old:
        i = old_to_new[old_i]
        st = int(indptr[old_i])
        ed = int(indptr[old_i + 1])
        neigh = indices[st:ed]
        for old_j in neigh:
            old_j = int(old_j)
            if old_j not in old_to_new:
                continue
            j = old_to_new[old_j]
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            degree[a] += 1
            degree[b] += 1
            t = int(rng.integers(1710000000, 1715000000))
            edge_rows.append([f"m{a}", f"m{b}", t])
            if len(edge_rows) >= max_edges:
                break
        if len(edge_rows) >= max_edges:
            break

    rng.shuffle(edge_rows)

    # 使用结构统计构建轻量特征，避免读取超大 attr_matrix。
    deg_max = max(int(degree.max()), 1)
    feature_rows: List[List[object]] = []
    for i in range(num_nodes):
        d = float(degree[i])
        feature_rows.append(
            [
                f"m{i}",
                d,
                np.log1p(d),
                d / deg_max,
                float(rng.normal(0.0, 0.1)),
                float(rng.normal(0.0, 0.1)),
            ]
        )
    feature_header = ["node_id", "f1", "f2", "f3", "f4", "f5"]

    label_rows = [[f"m{i}", int(labels[selected_old[i]])] for i in range(num_nodes)]
    write_csv(os.path.join(out_dir, "features.csv"), feature_header, feature_rows)
    write_csv(os.path.join(out_dir, "labels.csv"), ["node_id", "label"], label_rows)
    write_csv(os.path.join(out_dir, "edges.csv"), ["src", "dst", "time"], edge_rows)

    # 生成一份轻量属性更新。
    attr_updates: List[List[object]] = []
    for _ in range(min(2500, num_nodes * 2)):
        idx = int(rng.integers(0, num_nodes))
        base = np.asarray(feature_rows[idx][1:], dtype=np.float64)
        new_vec = base + rng.normal(0.0, 0.02, size=5)
        t = int(rng.integers(1710000000, 1715000000))
        attr_updates.append([t, f"m{idx}"] + [float(v) for v in new_vec.tolist()])
    write_csv(
        os.path.join(out_dir, "attr_updates.csv"),
        ["time", "node_id", "f1", "f2", "f3", "f4", "f5"],
        attr_updates,
    )
    return out_dir


def prepare_twitter_sample(max_nodes: int = 12000, max_edges: int = 180000, seed: int = 42) -> str:
    """从 twitter_sampled/twitter_combined.txt.gz 构建样本集。

    说明：
    - 原始 twitter_full 非常大，优先使用 twitter_sampled 中的 combined 边文件。
    - 数据本身无标准标签和属性，这里构造结构统计特征与伪标签（度分桶）。
    """
    rng = np.random.default_rng(seed)
    src = os.path.join(source_root(), "twitter", "twitter_sampled", "twitter_combined.txt.gz")
    out_dir = os.path.join(project_root(), "data", "twitter_sample")
    ensure_dir(out_dir)

    selected_nodes: Dict[str, int] = {}
    edge_rows: List[List[object]] = []
    seen = set()

    with gzip.open(src, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u = _safe_node_id(parts[0])
            v = _safe_node_id(parts[1])
            if u == "" or v == "" or u == v:
                continue

            # 控制节点规模：若超限则跳过引入新节点的边。
            new_nodes = []
            if u not in selected_nodes:
                new_nodes.append(u)
            if v not in selected_nodes:
                new_nodes.append(v)
            if len(selected_nodes) + len(new_nodes) > max_nodes:
                continue
            for node in new_nodes:
                selected_nodes[node] = len(selected_nodes)

            a, b = (u, v) if u < v else (v, u)
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            t = int(rng.integers(1710000000, 1715000000))
            edge_rows.append([a, b, t])
            if len(edge_rows) >= max_edges:
                break

    node_list = sorted(selected_nodes.keys(), key=lambda x: selected_nodes[x])
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    degree = np.zeros(len(node_list), dtype=np.int64)
    for s, d, _ in edge_rows:
        i = node_to_idx[s]
        j = node_to_idx[d]
        degree[i] += 1
        degree[j] += 1

    deg_max = max(int(degree.max()), 1)
    feature_rows: List[List[object]] = []
    for node in node_list:
        i = node_to_idx[node]
        d = float(degree[i])
        feature_rows.append(
            [
                node,
                d,
                np.log1p(d),
                d / deg_max,
                float(rng.normal(0.0, 0.1)),
                float(rng.normal(0.0, 0.1)),
            ]
        )
    feature_header = ["node_id", "f1", "f2", "f3", "f4", "f5"]

    # 伪标签：按度分位数分 5 档，便于分类评估链路跑通。
    if len(node_list) > 0:
        quantiles = np.quantile(degree.astype(np.float64), [0.2, 0.4, 0.6, 0.8])
    else:
        quantiles = np.array([0, 0, 0, 0], dtype=np.float64)
    label_rows: List[List[object]] = []
    for node in node_list:
        d = float(degree[node_to_idx[node]])
        label = int(np.sum(d > quantiles))
        label_rows.append([node, label])

    write_csv(os.path.join(out_dir, "features.csv"), feature_header, feature_rows)
    write_csv(os.path.join(out_dir, "labels.csv"), ["node_id", "label"], label_rows)
    write_csv(os.path.join(out_dir, "edges.csv"), ["src", "dst", "time"], edge_rows)

    # 生成属性更新，模拟节点行为漂移。
    attr_updates: List[List[object]] = []
    for _ in range(min(3000, len(node_list) * 2)):
        idx = int(rng.integers(0, len(node_list)))
        node = node_list[idx]
        base = np.asarray(feature_rows[idx][1:], dtype=np.float64)
        new_vec = base + rng.normal(0.0, 0.03, size=5)
        t = int(rng.integers(1710000000, 1715000000))
        attr_updates.append([t, node] + [float(v) for v in new_vec.tolist()])
    write_csv(
        os.path.join(out_dir, "attr_updates.csv"),
        ["time", "node_id", "f1", "f2", "f3", "f4", "f5"],
        attr_updates,
    )
    return out_dir


def prepare_amazon3m_sample(
    max_nodes: int = 10000, max_edges: int = 200000, seed: int = 42
) -> str:
    """从 Amazon-3M.raw 构建样本集。

    Amazon-3M 是极端多标签分类数据集，每条记录为一个商品，包含标题、描述文本
    及其关联的标签索引列表。这里通过共享标签（co-label）关系构建商品图：
    若两个商品共享 >= 2 个相同的 target_ind，则建一条边。

    特征采用文本统计量（标题词数、内容词数、内容字符数、关联标签数）加随机哈希
    特征，不依赖外部 NLP 库。标签取商品最频繁出现的 target_ind 再分桶。
    """
    rng = np.random.default_rng(seed)
    src_dir = os.path.join(source_root(), "Amazon-3M.raw")
    trn_path = os.path.join(src_dir, "trn.json.gz")
    out_dir = os.path.join(project_root(), "data", "amazon3m_sample")
    ensure_dir(out_dir)

    import gzip as _gzip

    read_limit = max_nodes * 4

    products: List[dict] = []
    with _gzip.open(trn_path, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= read_limit:
                break
            obj = json.loads(line)
            tids = obj.get("target_ind", [])
            if len(tids) < 2:
                continue
            products.append(obj)

    if len(products) > max_nodes:
        idx_sel = rng.choice(len(products), size=max_nodes, replace=False)
        idx_sel.sort()
        products = [products[i] for i in idx_sel]

    num_nodes = len(products)
    uid_to_idx = {p["uid"]: i for i, p in enumerate(products)}
    node_ids = [p["uid"] for p in products]

    label_to_products: Dict[int, List[int]] = {}
    for i, p in enumerate(products):
        for tid in p["target_ind"]:
            label_to_products.setdefault(tid, []).append(i)

    edge_set: set = set()
    for tid, members in label_to_products.items():
        if len(members) > 500:
            continue
        for ai in range(len(members)):
            for bi in range(ai + 1, len(members)):
                u, v = members[ai], members[bi]
                if u > v:
                    u, v = v, u
                edge_set.add((u, v))
                if len(edge_set) >= max_edges * 3:
                    break
            if len(edge_set) >= max_edges * 3:
                break
        if len(edge_set) >= max_edges * 3:
            break

    if len(edge_set) > max_edges:
        edge_list = list(edge_set)
        sel = rng.choice(len(edge_list), size=max_edges, replace=False)
        edge_set = {edge_list[i] for i in sel}

    edge_rows: List[List[object]] = []
    degree = np.zeros(num_nodes, dtype=np.int64)
    for u, v in edge_set:
        t = int(rng.integers(1710000000, 1715000000))
        edge_rows.append([node_ids[u], node_ids[v], t])
        degree[u] += 1
        degree[v] += 1
    rng.shuffle(edge_rows)

    feature_rows: List[List[object]] = []
    for i, p in enumerate(products):
        title = p.get("title", "")
        content = p.get("content", "")
        title_words = len(title.split())
        content_words = len(content.split())
        content_chars = len(content)
        num_targets = len(p.get("target_ind", []))
        d = float(degree[i])
        h1 = float((hash(title) % 10000) / 10000.0)
        h2 = float((hash(content[:50]) % 10000) / 10000.0)
        feature_rows.append([
            node_ids[i],
            float(title_words),
            float(content_words),
            float(content_chars),
            float(num_targets),
            d,
            np.log1p(d),
            h1,
            h2,
        ])
    feature_header = ["node_id", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]

    all_tids: List[int] = []
    for p in products:
        all_tids.extend(p.get("target_ind", []))
    from collections import Counter
    tid_counts = Counter(all_tids)
    top_labels = [tid for tid, _ in tid_counts.most_common(20)]
    top_set = set(top_labels)
    top_to_cls = {tid: ci for ci, tid in enumerate(top_labels)}

    label_rows: List[List[object]] = []
    for i, p in enumerate(products):
        tids = p.get("target_ind", [])
        best = -1
        for tid in tids:
            if tid in top_set:
                best = top_to_cls[tid]
                break
        if best < 0:
            best = len(top_labels)
        label_rows.append([node_ids[i], best])

    write_csv(os.path.join(out_dir, "features.csv"), feature_header, feature_rows)
    write_csv(os.path.join(out_dir, "labels.csv"), ["node_id", "label"], label_rows)
    write_csv(os.path.join(out_dir, "edges.csv"), ["src", "dst", "time"], edge_rows)

    attr_updates: List[List[object]] = []
    for _ in range(min(3000, num_nodes * 2)):
        idx = int(rng.integers(0, num_nodes))
        base = np.asarray(feature_rows[idx][1:], dtype=np.float64)
        new_vec = base + rng.normal(0.0, 0.03, size=len(base))
        t = int(rng.integers(1710000000, 1715000000))
        attr_updates.append([t, node_ids[idx]] + [float(v) for v in new_vec.tolist()])
    write_csv(
        os.path.join(out_dir, "attr_updates.csv"),
        ["time", "node_id"] + feature_header[1:],
        attr_updates,
    )

    print(f"  amazon3m_sample: {num_nodes} nodes, {len(edge_set)} edges")
    return out_dir


def main() -> None:
    """命令行入口：可分别准备各数据集样本。"""
    parser = argparse.ArgumentParser(description="将 D:/毕设资料/dataset 数据整理到当前实验目录")
    parser.add_argument("--prepare-reddit", action="store_true")
    parser.add_argument("--prepare-amazon", action="store_true")
    parser.add_argument("--prepare-amazon3m", action="store_true")
    parser.add_argument("--prepare-mag", action="store_true")
    parser.add_argument("--prepare-twitter", action="store_true")
    parser.add_argument("--reddit-max-nodes", type=int, default=15000)
    parser.add_argument("--amazon-max-nodes", type=int, default=30000)
    parser.add_argument("--amazon-max-edges", type=int, default=250000)
    parser.add_argument("--amazon3m-max-nodes", type=int, default=10000)
    parser.add_argument("--amazon3m-max-edges", type=int, default=200000)
    parser.add_argument("--mag-max-nodes", type=int, default=8000)
    parser.add_argument("--mag-max-edges", type=int, default=200000)
    parser.add_argument("--twitter-max-nodes", type=int, default=12000)
    parser.add_argument("--twitter-max-edges", type=int, default=180000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    any_flag = (
        args.prepare_reddit or args.prepare_amazon or args.prepare_amazon3m
        or args.prepare_mag or args.prepare_twitter
    )
    if not any_flag:
        args.prepare_reddit = True
        args.prepare_amazon = True
        args.prepare_amazon3m = True
        args.prepare_mag = True
        args.prepare_twitter = True

    outputs = []
    if args.prepare_reddit:
        outputs.append(("reddit_sample", prepare_reddit_sample(args.reddit_max_nodes, args.seed)))
    if args.prepare_amazon:
        outputs.append(
            (
                "amazon2m_sample",
                prepare_amazon2m_sample(
                    args.amazon_max_nodes,
                    args.amazon_max_edges,
                    args.seed,
                ),
            )
        )
    if args.prepare_mag:
        outputs.append(
            (
                "mag_sample",
                prepare_mag_sample(
                    args.mag_max_nodes,
                    args.mag_max_edges,
                    args.seed,
                ),
            )
        )
    if args.prepare_twitter:
        outputs.append(
            (
                "twitter_sample",
                prepare_twitter_sample(
                    args.twitter_max_nodes,
                    args.twitter_max_edges,
                    args.seed,
                ),
            )
        )
    if args.prepare_amazon3m:
        outputs.append(
            (
                "amazon3m_sample",
                prepare_amazon3m_sample(
                    args.amazon3m_max_nodes,
                    args.amazon3m_max_edges,
                    args.seed,
                ),
            )
        )

    print("数据整理完成：")
    for name, path in outputs:
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
