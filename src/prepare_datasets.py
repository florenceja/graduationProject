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
import hashlib
import json
import os
import re
import tarfile
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    """若目录不存在则创建。"""
    os.makedirs(path, exist_ok=True)


def source_root() -> str:
    """原始数据集根目录。

    优先使用当前项目下的 dataset/ 目录，便于仓库内自包含管理；
    若项目内不存在该目录，则回退到历史外部路径，保持兼容性。
    """
    project_dataset = os.path.join(project_root(), "dataset")
    if os.path.isdir(project_dataset):
        return project_dataset
    return "D:/" + "".join(map(chr, [0x6bd5, 0x8bbe, 0x8d44, 0x6599])) + "/dataset"


def project_root() -> str:
    """当前项目根目录（src/ 的上一级）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    """统一 CSV 写入封装。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _safe_node_id(raw: object) -> str:
    """将任意节点 ID 规范化为字符串。"""
    return str(raw).strip()


def _remove_if_exists(path: str) -> None:
    """删除旧文件，避免过期产物残留。"""
    if os.path.exists(path):
        os.remove(path)


def _text_hash_features(*texts: str, dim: int) -> np.ndarray:
    """使用稳定哈希生成固定维度文本特征。"""
    vec = np.zeros(dim, dtype=np.float64)
    for text in texts:
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            pos = int.from_bytes(digest[:4], byteorder="little", signed=False) % dim
            vec[pos] += 1.0
    return vec


def _dense_row_from_selected_cols(
    row_idx: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    col_to_pos: Dict[int, int],
    width: int,
) -> np.ndarray:
    """从 CSR 稀疏矩阵中抽取指定列形成定长稠密向量。"""
    row = np.zeros(width, dtype=np.float64)
    st = int(indptr[row_idx])
    ed = int(indptr[row_idx + 1])
    for col, value in zip(indices[st:ed], data[st:ed]):
        pos = col_to_pos.get(int(col))
        if pos is not None:
            row[pos] = float(value)
    return row


def _read_tar_text(tf: tarfile.TarFile, member_name: str) -> str:
    """读取 tar.gz 中的文本文件。"""
    member = tf.getmember(member_name)
    extracted = tf.extractfile(member)
    if extracted is None:
        raise ValueError(f"无法读取压缩包成员: {member_name}")
    return extracted.read().decode("utf-8", errors="ignore")


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

    if edge_rows:
        order = rng.permutation(len(edge_rows))
        edge_rows = [edge_rows[int(i)] for i in order]

    feature_header = ["node_id"] + [f"f{i+1}" for i in range(feat_dim)]
    write_csv(os.path.join(out_dir, "features.csv"), feature_header, feature_rows)
    write_csv(os.path.join(out_dir, "labels.csv"), ["node_id", "label"], label_rows)
    write_csv(os.path.join(out_dir, "edges.csv"), ["src", "dst", "time"], edge_rows)

    # 生成可控的属性更新样本，模拟动态属性漂移。
    attr_updates: List[List[object]] = []
    label_map = {str(row[0]): int(float(str(row[1]))) for row in label_rows}
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
        row: List[object] = [node_id]
        row.extend(float(v) for v in feat[i].tolist())
        feature_rows.append(row)
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

    if edge_rows:
        order = rng.permutation(len(edge_rows))
        edge_rows = [edge_rows[int(i)] for i in order]
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
    - 特征优先来自原始 attr_matrix 在采样子图上的高频属性列，避免退化成纯结构代理特征。
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
            edge_rows.append([f"m{a}", f"m{b}"])
            if len(edge_rows) >= max_edges:
                break
        if len(edge_rows) >= max_edges:
            break

    if edge_rows:
        order = rng.permutation(len(edge_rows))
        edge_rows = [edge_rows[int(i)] for i in order]

    selected_attr_cols: List[int] = []
    col_to_pos: Dict[int, int] = {}
    attr_indptr: Optional[np.ndarray] = None
    attr_indices: Optional[np.ndarray] = None
    attr_data: Optional[np.ndarray] = None
    try:
        attr_indptr = npz["attr_matrix.indptr"]
        attr_indices = npz["attr_matrix.indices"]
        attr_data = npz["attr_matrix.data"]
        assert attr_indptr is not None
        assert attr_indices is not None
        attr_counter: Counter[int] = Counter()
        for old_i in selected_old:
            st = int(attr_indptr[old_i])
            ed = int(attr_indptr[old_i + 1])
            attr_counter.update(int(col) for col in attr_indices[st:ed])
        selected_attr_cols = [col for col, _ in attr_counter.most_common(64)]
        col_to_pos = {col: pos for pos, col in enumerate(selected_attr_cols)}
    except MemoryError:
        selected_attr_cols = []
    feature_rows: List[List[object]] = []
    if selected_attr_cols and attr_indptr is not None and attr_indices is not None and attr_data is not None:
        width = len(selected_attr_cols)
        local_attr_indptr = attr_indptr
        local_attr_indices = attr_indices
        local_attr_data = attr_data
        for i, old_i in enumerate(selected_old):
            row = _dense_row_from_selected_cols(
                old_i,
                local_attr_indptr,
                local_attr_indices,
                local_attr_data,
                col_to_pos,
                width,
            )
            feature_row: List[object] = [f"m{i}"]
            feature_row.extend(float(v) for v in row.tolist())
            feature_rows.append(feature_row)
        feature_header = ["node_id"] + [f"f{i+1}" for i in range(width)]
    else:
        deg_max = max(int(degree.max()), 1)
        for i in range(num_nodes):
            d = float(degree[i])
            feature_rows.append([f"m{i}", d, np.log1p(d), d / deg_max])
        feature_header = ["node_id", "f1", "f2", "f3"]

    label_rows = [[f"m{i}", int(labels[selected_old[i]])] for i in range(num_nodes)]
    write_csv(os.path.join(out_dir, "features.csv"), feature_header, feature_rows)
    write_csv(os.path.join(out_dir, "labels.csv"), ["node_id", "label"], label_rows)
    write_csv(os.path.join(out_dir, "edges.csv"), ["src", "dst"], edge_rows)
    _remove_if_exists(os.path.join(out_dir, "attr_updates.csv"))
    return out_dir


def prepare_twitter_sample(max_nodes: int = 12000, max_edges: int = 180000, seed: int = 42) -> str:
    """从 twitter_sampled/twitter_combined.txt.gz 构建样本集。

    说明：
    - 优先使用 twitter_sampled/twitter.tar.gz 中的 ego-network 原始属性与圈层文件。
    - 输出静态属性图，不再伪造时间戳和属性更新。
    """
    src = os.path.join(source_root(), "twitter", "twitter_sampled", "twitter.tar.gz")
    out_dir = os.path.join(project_root(), "data", "twitter_sample")
    ensure_dir(out_dir)

    with tarfile.open(src, "r:gz") as tf:
        member_names = tf.getnames()
        stems: Dict[str, set] = defaultdict(set)
        for name in member_names:
            if not name.startswith("twitter/") or name.count(".") == 0:
                continue
            stem, ext = name.rsplit(".", 1)
            stems[stem].add(ext)

        required = {"feat", "egofeat", "circles", "edges", "featnames"}
        ego_stems = sorted(stem for stem, exts in stems.items() if required.issubset(exts))

        node_feature_names: Dict[str, set] = defaultdict(set)
        node_primary_circle: Dict[str, str] = {}
        edge_set: set[Tuple[str, str]] = set()
        node_order: Dict[str, int] = {}

        for stem in ego_stems:
            ego_id = stem.rsplit("/", 1)[-1]
            featnames_text = _read_tar_text(tf, f"{stem}.featnames")
            feat_names: List[str] = []
            for line in featnames_text.splitlines():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    feat_names.append(parts[1].strip())
            if not feat_names:
                continue

            feat_text = _read_tar_text(tf, f"{stem}.feat")
            egofeat_text = _read_tar_text(tf, f"{stem}.egofeat")
            circles_text = _read_tar_text(tf, f"{stem}.circles")
            edges_text = _read_tar_text(tf, f"{stem}.edges")

            local_nodes: List[str] = []

            ego_bits = [int(v) for v in egofeat_text.strip().split() if v != ""]
            if len(ego_bits) == len(feat_names):
                ego_attrs = {feat_names[i] for i, bit in enumerate(ego_bits) if bit != 0}
                node_feature_names[ego_id].update(ego_attrs)
            if ego_id not in node_order and len(node_order) < max_nodes:
                node_order[ego_id] = len(node_order)
            local_nodes.append(ego_id)

            for line in feat_text.splitlines():
                parts = line.strip().split()
                if len(parts) != len(feat_names) + 1:
                    continue
                node_id = _safe_node_id(parts[0])
                if node_id not in node_order and len(node_order) >= max_nodes:
                    continue
                if node_id not in node_order:
                    node_order[node_id] = len(node_order)
                attrs = {feat_names[i] for i, bit in enumerate(parts[1:]) if bit == "1"}
                node_feature_names[node_id].update(attrs)
                local_nodes.append(node_id)

            local_node_set = set(local_nodes)
            for line in circles_text.splitlines():
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                circle_name = f"{ego_id}:{parts[0]}"
                for node_id in parts[1:]:
                    node_id = _safe_node_id(node_id)
                    if node_id in local_node_set and node_id not in node_primary_circle:
                        node_primary_circle[node_id] = circle_name

            for node_id in local_node_set:
                if node_id == ego_id:
                    continue
                a, b = (ego_id, node_id) if ego_id < node_id else (node_id, ego_id)
                edge_set.add((a, b))

            for line in edges_text.splitlines():
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                u = _safe_node_id(parts[0])
                v = _safe_node_id(parts[1])
                if u not in local_node_set or v not in local_node_set or u == v:
                    continue
                a, b = (u, v) if u < v else (v, u)
                edge_set.add((a, b))

            if len(node_order) >= max_nodes and len(edge_set) >= max_edges:
                break

        selected_node_ids = [node for node, _ in sorted(node_order.items(), key=lambda item: item[1])]
        selected_set = set(selected_node_ids)
        edge_rows = [[u, v] for u, v in edge_set if u in selected_set and v in selected_set][:max_edges]

        feature_counter: Counter[str] = Counter()
        for node_id in selected_node_ids:
            feature_counter.update(node_feature_names.get(node_id, set()))
        selected_features = [name for name, _ in feature_counter.most_common(128)]
        feature_to_pos = {name: idx for idx, name in enumerate(selected_features)}

        feature_rows: List[List[object]] = []
        for node_id in selected_node_ids:
            row = np.zeros(len(selected_features), dtype=np.float64)
            for feat_name in node_feature_names.get(node_id, set()):
                pos = feature_to_pos.get(feat_name)
                if pos is not None:
                    row[pos] = 1.0
            feature_row: List[object] = [node_id]
            feature_row.extend(float(v) for v in row.tolist())
            feature_rows.append(feature_row)

        label_names = sorted({name for node, name in node_primary_circle.items() if node in selected_set})
        label_to_idx = {name: idx for idx, name in enumerate(label_names)}
        label_rows = [
            [node_id, label_to_idx[node_primary_circle[node_id]]]
            for node_id in selected_node_ids
            if node_id in node_primary_circle
        ]

    feature_header = ["node_id"] + [f"f{i+1}" for i in range(len(selected_features))]
    write_csv(os.path.join(out_dir, "features.csv"), feature_header, feature_rows)
    write_csv(os.path.join(out_dir, "labels.csv"), ["node_id", "label"], label_rows)
    write_csv(os.path.join(out_dir, "edges.csv"), ["src", "dst"], edge_rows)
    _remove_if_exists(os.path.join(out_dir, "attr_updates.csv"))
    return out_dir


def prepare_amazon3m_sample(
    max_nodes: int = 10000, max_edges: int = 200000, seed: int = 42
) -> str:
    """从 Amazon-3M.raw 构建样本集。

    Amazon-3M 是极端多标签分类数据集，每条记录为一个商品，包含标题、描述文本
    及其关联的标签索引列表。这里通过共享标签（co-label）关系构建商品图：
    若两个商品共享 >= 2 个相同的 target_ind，则建一条边。

    特征采用文本统计量 + 稳定哈希词袋，不依赖外部 NLP 库。
    标签使用每个商品 target_rel 最强的原始 target_ind，再重映射为连续整数。
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
        edge_rows.append([node_ids[u], node_ids[v]])
        degree[u] += 1
        degree[v] += 1
    if edge_rows:
        order = rng.permutation(len(edge_rows))
        edge_rows = [edge_rows[int(i)] for i in order]

    feature_rows: List[List[object]] = []
    for i, p in enumerate(products):
        title = p.get("title", "")
        content = p.get("content", "")
        title_words = len(title.split())
        content_words = len(content.split())
        content_chars = len(content)
        num_targets = len(p.get("target_ind", []))
        d = float(degree[i])
        rel_values = [float(v) for v in p.get("target_rel", [])]
        rel_mean = float(np.mean(rel_values)) if rel_values else 0.0
        rel_max = float(np.max(rel_values)) if rel_values else 0.0
        text_hash = _text_hash_features(title, content, dim=16)
        row: List[object] = [
            node_ids[i],
            float(title_words),
            float(content_words),
            float(content_chars),
            float(num_targets),
            d,
            np.log1p(d),
            rel_mean,
            rel_max,
        ]
        row.extend(float(v) for v in text_hash.tolist())
        feature_rows.append(row)
    feature_header = ["node_id", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"] + [
        f"f{i+9}" for i in range(16)
    ]

    raw_best_labels: List[int] = []
    for p in products:
        tids = [int(v) for v in p.get("target_ind", [])]
        rels = [float(v) for v in p.get("target_rel", [])]
        if not tids:
            raw_best_labels.append(-1)
            continue
        if len(rels) == len(tids) and rels:
            best_idx = int(np.argmax(np.asarray(rels, dtype=np.float64)))
        else:
            best_idx = 0
        raw_best_labels.append(tids[best_idx])

    valid_raw_labels = sorted({lab for lab in raw_best_labels if lab >= 0})
    label_to_dense = {lab: idx for idx, lab in enumerate(valid_raw_labels)}

    label_rows: List[List[object]] = []
    for node_id, raw_label in zip(node_ids, raw_best_labels):
        dense_label = label_to_dense.get(raw_label, -1)
        label_rows.append([node_id, dense_label])

    write_csv(os.path.join(out_dir, "features.csv"), feature_header, feature_rows)
    write_csv(os.path.join(out_dir, "labels.csv"), ["node_id", "label"], label_rows)
    write_csv(os.path.join(out_dir, "edges.csv"), ["src", "dst"], edge_rows)
    _remove_if_exists(os.path.join(out_dir, "attr_updates.csv"))

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
