"""OAG 数据转换与校验脚本。

当前项目的 file 模式固定使用 `data/OAG/` 作为真实数据输入目录。
本脚本负责两类工作：
1) 将 `dataset/OAG/v5_oag_publication_*.zip` 转换为当前流水线可直接消费的 CSV；
2) 检查 `data/OAG/` 是否已按统一 CSV 格式放置完成。

必需文件：
- edges.csv  (src, dst, time)
- features.csv  (node_id, f1...fN)
- labels.csv  (node_id, label)
- attr_updates.csv  (可选)
"""

import argparse
import csv
import glob
import gzip
import hashlib
import json
import os
import re
import sqlite3
import tarfile
import tempfile
import zipfile
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def oag_dataset_dir() -> str:
    return os.path.join(project_root(), "data", "OAG")


def oag_raw_dir() -> str:
    return os.path.join(project_root(), "dataset", "OAG")


def validate_oag_dataset(dataset_dir: Optional[str] = None) -> str:
    dataset_dir = dataset_dir or oag_dataset_dir()
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"未找到固定数据集目录: {dataset_dir}")
    required_files = ["edges.csv", "features.csv", "labels.csv"]
    missing = [name for name in required_files if not os.path.isfile(os.path.join(dataset_dir, name))]
    if missing:
        raise ValueError(f"data/OAG 缺少必需文件: {', '.join(missing)}")
    return dataset_dir


def _normalize_venue(raw: object) -> str:
    text = str(raw or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _coerce_year(raw: object) -> Optional[int]:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    try:
        year = int(float(text))
    except ValueError:
        return None
    if year < 1000 or year > 2100:
        return None
    return year


def _stringify_keywords(raw: object) -> str:
    if isinstance(raw, list):
        parts = [str(item).strip() for item in raw if str(item).strip()]
        return " ".join(parts)
    return str(raw or "").strip()


def _discover_oag_archives(input_glob: str) -> List[str]:
    paths = sorted(glob.glob(input_glob))
    if not paths:
        raise ValueError(f"未找到 OAG 压缩包，匹配模式: {input_glob}")
    return paths


def resolve_oag_subset_profile(profile: str) -> Dict[str, int]:
    normalized = str(profile or "custom").strip().lower()
    if normalized == "test":
        return {"max_papers": 50000, "feature_dim": 128, "min_venue_support": 50, "max_record_bytes": 2_000_000, "candidate_multiplier": 3}
    if normalized == "small":
        return {"max_papers": 200000, "feature_dim": 128, "min_venue_support": 100, "max_record_bytes": 2_000_000, "candidate_multiplier": 3}
    if normalized == "medium":
        return {"max_papers": 1000000, "feature_dim": 128, "min_venue_support": 200, "max_record_bytes": 2_000_000, "candidate_multiplier": 2}
    if normalized == "full":
        return {"max_papers": 0, "feature_dim": 128, "min_venue_support": 500, "max_record_bytes": 2_000_000, "candidate_multiplier": 2}
    return {"max_papers": 0, "feature_dim": 128, "min_venue_support": 50, "max_record_bytes": 4_000_000, "candidate_multiplier": 2}


def _iter_oag_records(
    zip_paths: Sequence[str],
    fail_on_malformed: bool = False,
    max_record_bytes: int = 4_000_000,
) -> Iterable[Tuple[dict, str, int]]:
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path, "r") as zf:
            json_members = [name for name in zf.namelist() if name.lower().endswith(".json")]
            if not json_members:
                raise ValueError(f"压缩包内未找到 JSON 文件: {zip_path}")
            for member in json_members:
                with zf.open(member, "r") as f:
                    for line_no, raw_line in enumerate(f, start=1):
                        if not raw_line.strip():
                            continue
                        if max_record_bytes > 0 and len(raw_line) > max_record_bytes:
                            continue
                        try:
                            obj = json.loads(raw_line)
                        except json.JSONDecodeError:
                            if fail_on_malformed:
                                raise ValueError(f"JSON 解析失败: {zip_path}::{member}:{line_no}") from None
                            continue
                        if isinstance(obj, dict):
                            yield obj, zip_path, line_no


def _write_atomic_csv(final_path: str, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    ensure_dir(os.path.dirname(final_path))
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(final_path) + ".", suffix=".tmp", dir=os.path.dirname(final_path))
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)
        os.replace(tmp_path, final_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def convert_oag_archives(
    input_glob: str,
    output_dir: str,
    feature_dim: int = 128,
    min_venue_support: int = 50,
    max_papers: int = 0,
    min_year: int = 0,
    max_year: int = 0,
    keep_unlabeled: bool = False,
    include_attr_updates: bool = False,
    overwrite: bool = False,
    fail_on_malformed: bool = False,
    dry_run: bool = False,
    report_every: int = 100000,
    max_record_bytes: int = 4_000_000,
    selection_strategy: str = "dense",
    candidate_multiplier: int = 2,
) -> Dict[str, int]:
    zip_paths = _discover_oag_archives(input_glob)
    ensure_dir(output_dir)
    output_edges = os.path.join(output_dir, "edges.csv")
    output_features = os.path.join(output_dir, "features.csv")
    output_labels = os.path.join(output_dir, "labels.csv")
    output_attr_updates = os.path.join(output_dir, "attr_updates.csv")

    if not overwrite:
        existing = [path for path in (output_edges, output_features, output_labels) if os.path.exists(path)]
        if existing:
            raise ValueError(f"目标 CSV 已存在，请使用 --overwrite 覆盖: {', '.join(existing)}")

    stats: Counter[str] = Counter()
    venue_counts: Counter[str] = Counter()
    db_fd, db_path = tempfile.mkstemp(prefix="oag_meta_", suffix=".sqlite", dir=output_dir)
    os.close(db_fd)
    conn = sqlite3.connect(db_path)
    feature_lookup = None
    edge_src_lookup = None
    edge_dst_lookup = None
    try:
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute(
            "CREATE TABLE papers (paper_id TEXT PRIMARY KEY, year INTEGER NOT NULL, venue TEXT NOT NULL, kept INTEGER NOT NULL DEFAULT 0, label INTEGER NOT NULL DEFAULT -1, outgoing_count INTEGER NOT NULL DEFAULT 0, incoming_count INTEGER NOT NULL DEFAULT 0, score REAL NOT NULL DEFAULT 0.0)"
        )
        conn.execute("CREATE TABLE ref_counts (paper_id TEXT PRIMARY KEY, cnt INTEGER NOT NULL DEFAULT 0)")

        insert_sql = "INSERT OR IGNORE INTO papers(paper_id, year, venue, outgoing_count) VALUES(?, ?, ?, ?)"
        ref_upsert_sql = "INSERT INTO ref_counts(paper_id, cnt) VALUES(?, 1) ON CONFLICT(paper_id) DO UPDATE SET cnt = cnt + 1"
        batch: List[Tuple[str, int, str, int, List[str]]] = []
        inserted_papers = 0
        selection_strategy = selection_strategy.strip().lower()
        candidate_limit = 0
        if max_papers > 0:
            if selection_strategy == "legacy":
                candidate_limit = max_papers
            else:
                candidate_limit = max(max_papers, max_papers * max(1, candidate_multiplier))
        for obj, _, _ in _iter_oag_records(
            zip_paths,
            fail_on_malformed=fail_on_malformed,
            max_record_bytes=max_record_bytes,
        ):
            stats["records_total"] += 1
            paper_id = _safe_node_id(obj.get("id", ""))
            if paper_id == "":
                stats["missing_id"] += 1
                continue
            year = _coerce_year(obj.get("year"))
            if year is None:
                stats["missing_or_invalid_year"] += 1
                continue
            if min_year > 0 and year < min_year:
                stats["year_filtered"] += 1
                continue
            if max_year > 0 and year > max_year:
                stats["year_filtered"] += 1
                continue

            venue = _normalize_venue(obj.get("venue", ""))
            if venue:
                venue_counts[venue] += 1
            else:
                stats["missing_venue"] += 1

            refs = obj.get("references", [])
            outgoing_count = len(refs) if isinstance(refs, list) else 0

            normalized_refs: List[str] = []
            if isinstance(refs, list):
                for ref in refs:
                    dst = _safe_node_id(ref)
                    if dst and dst != paper_id:
                        normalized_refs.append(dst)
            batch.append((paper_id, year, venue, outgoing_count, normalized_refs))
            should_flush = len(batch) >= 5000 or (candidate_limit > 0 and inserted_papers + len(batch) >= candidate_limit)
            if should_flush:
                if candidate_limit > 0 and inserted_papers + len(batch) > candidate_limit:
                    keep = max(candidate_limit - inserted_papers, 0)
                    batch = batch[:keep]
                insert_rows = [(paper_id, year, venue, outgoing_count) for paper_id, year, venue, outgoing_count, _ in batch]
                ref_batch = [(dst,) for _, _, _, _, refs_list in batch for dst in refs_list]
                before = int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
                conn.executemany(insert_sql, insert_rows)
                if ref_batch:
                    conn.executemany(ref_upsert_sql, ref_batch)
                conn.commit()
                after = int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
                inserted_papers += after - before
                batch.clear()
                if candidate_limit > 0 and inserted_papers >= candidate_limit:
                    break
            if report_every > 0 and stats["records_total"] % report_every == 0:
                print(f"[OAG pass1] scanned {stats['records_total']} records...")

        if batch and (candidate_limit <= 0 or inserted_papers < candidate_limit):
            insert_rows = [(paper_id, year, venue, outgoing_count) for paper_id, year, venue, outgoing_count, _ in batch]
            ref_batch = [(dst,) for _, _, _, _, refs_list in batch for dst in refs_list]
            before = int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
            conn.executemany(insert_sql, insert_rows)
            if ref_batch:
                conn.executemany(ref_upsert_sql, ref_batch)
            conn.commit()
            after = int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
            inserted_papers += after - before

        stats["duplicate_ids"] = max(stats["records_total"] - stats["missing_id"] - stats["missing_or_invalid_year"] - stats["year_filtered"] - inserted_papers, 0)
        conn.execute(
            "UPDATE papers SET incoming_count = COALESCE((SELECT cnt FROM ref_counts WHERE ref_counts.paper_id = papers.paper_id), 0)"
        )

        supported_venues = sorted([venue for venue, count in venue_counts.items() if count >= max(1, min_venue_support)])
        venue_to_label = {venue: idx for idx, venue in enumerate(supported_venues)}
        conn.execute("CREATE TABLE candidate_refs (src TEXT NOT NULL, dst TEXT NOT NULL)")

        for venue, label in venue_to_label.items():
            conn.execute(
                "UPDATE papers SET label=?, score=(incoming_count + 0.5 * outgoing_count + 5.0) WHERE venue=?",
                (label, venue),
            )
        conn.execute(
            "UPDATE papers SET score=(incoming_count + 0.5 * outgoing_count) WHERE label < 0"
        )

        if max_papers > 0 and selection_strategy == "dense":
            candidate_pool = max(max_papers, max_papers * max(1, candidate_multiplier))
            conn.execute("UPDATE papers SET kept=0")
            if keep_unlabeled:
                conn.execute(
                    "UPDATE papers SET kept=1 WHERE paper_id IN (SELECT paper_id FROM papers ORDER BY score DESC, year DESC, paper_id ASC LIMIT ?)",
                    (candidate_pool,),
                )
            else:
                conn.execute(
                    "UPDATE papers SET kept=1 WHERE paper_id IN (SELECT paper_id FROM papers WHERE label >= 0 ORDER BY score DESC, year DESC, paper_id ASC LIMIT ?)",
                    (candidate_pool,),
                )
            conn.commit()

            candidate_ids = {row[0] for row in conn.execute("SELECT paper_id FROM papers WHERE kept=1")}
            remaining_candidate_ids = set(candidate_ids)
            for obj, _, _ in _iter_oag_records(
                zip_paths,
                fail_on_malformed=fail_on_malformed,
                max_record_bytes=max_record_bytes,
            ):
                src = _safe_node_id(obj.get("id", ""))
                if src not in remaining_candidate_ids:
                    continue
                refs = obj.get("references", [])
                if not isinstance(refs, list):
                    remaining_candidate_ids.discard(src)
                    continue
                local_pairs = []
                seen = set()
                for ref in refs:
                    dst = _safe_node_id(ref)
                    if dst in candidate_ids and dst != src and dst not in seen:
                        seen.add(dst)
                        local_pairs.append((src, dst))
                if local_pairs:
                    conn.executemany("INSERT INTO candidate_refs(src, dst) VALUES(?, ?)", local_pairs)
                remaining_candidate_ids.discard(src)
                if not remaining_candidate_ids:
                    break
            conn.execute(
                "UPDATE papers SET score = score + 2.0 * COALESCE((SELECT COUNT(*) FROM candidate_refs WHERE candidate_refs.src = papers.paper_id), 0) + 2.0 * COALESCE((SELECT COUNT(*) FROM candidate_refs WHERE candidate_refs.dst = papers.paper_id), 0)"
            )
            conn.execute("UPDATE papers SET kept=0")

        if max_papers > 0:
            if keep_unlabeled:
                conn.execute(
                    "UPDATE papers SET kept=1 WHERE paper_id IN (SELECT paper_id FROM papers ORDER BY score DESC, year DESC, paper_id ASC LIMIT ?)",
                    (max_papers,),
                )
            else:
                conn.execute(
                    "UPDATE papers SET kept=1 WHERE paper_id IN (SELECT paper_id FROM papers WHERE label >= 0 ORDER BY score DESC, year DESC, paper_id ASC LIMIT ?)",
                    (max_papers,),
                )
        else:
            if keep_unlabeled:
                conn.execute("UPDATE papers SET kept=1")
            else:
                conn.execute("UPDATE papers SET kept=1 WHERE label >= 0")
        conn.commit()

        stats["kept_papers"] = int(conn.execute("SELECT COUNT(*) FROM papers WHERE kept=1").fetchone()[0])
        stats["supported_venues"] = len(venue_to_label)
        stats["labeled_papers"] = int(conn.execute("SELECT COUNT(*) FROM papers WHERE kept=1 AND label>=0").fetchone()[0])
        stats["unlabeled_kept_papers"] = int(conn.execute("SELECT COUNT(*) FROM papers WHERE kept=1 AND label<0").fetchone()[0])
        stats["candidate_papers"] = inserted_papers
        kept_paper_ids = {row[0] for row in conn.execute("SELECT paper_id FROM papers WHERE kept=1")}

        if dry_run:
            return dict(stats)

        feature_lookup = conn.cursor()
        edge_src_lookup = conn.cursor()
        edge_dst_lookup = conn.cursor()

        def iter_feature_rows() -> Iterable[Sequence[object]]:
            written = 0
            remaining_ids = set(kept_paper_ids)
            for obj, _, _ in _iter_oag_records(
                zip_paths,
                fail_on_malformed=fail_on_malformed,
                max_record_bytes=max_record_bytes,
            ):
                paper_id = _safe_node_id(obj.get("id", ""))
                if paper_id not in remaining_ids:
                    continue
                row = feature_lookup.execute("SELECT kept FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
                if row is None or int(row[0]) != 1:
                    remaining_ids.discard(paper_id)
                    continue
                text_vec = _text_hash_features(
                    str(obj.get("title", "") or ""),
                    str(obj.get("abstract", "") or ""),
                    _stringify_keywords(obj.get("keywords", [])),
                    dim=feature_dim,
                )
                written += 1
                remaining_ids.discard(paper_id)
                yield [paper_id] + [float(v) for v in text_vec.tolist()]
                if not remaining_ids:
                    break
            stats["written_features"] = written

        def iter_label_rows() -> Iterable[Sequence[object]]:
            written = 0
            for paper_id, label in conn.execute("SELECT paper_id, label FROM papers WHERE kept=1 ORDER BY paper_id"):
                written += 1
                yield [paper_id, label]
            stats["written_labels"] = written

        def iter_edge_rows() -> Iterable[Sequence[object]]:
            written = 0
            unresolved = 0
            remaining_src_ids = set(kept_paper_ids)
            for obj, _, _ in _iter_oag_records(
                zip_paths,
                fail_on_malformed=fail_on_malformed,
                max_record_bytes=max_record_bytes,
            ):
                src = _safe_node_id(obj.get("id", ""))
                if src not in remaining_src_ids:
                    continue
                src_row = edge_src_lookup.execute("SELECT kept, year FROM papers WHERE paper_id=?", (src,)).fetchone()
                if src_row is None or int(src_row[0]) != 1:
                    remaining_src_ids.discard(src)
                    continue
                year = int(src_row[1])
                refs = obj.get("references", [])
                if not isinstance(refs, list):
                    remaining_src_ids.discard(src)
                    continue
                local_seen = set()
                for ref in refs:
                    dst = _safe_node_id(ref)
                    if dst == "" or dst == src or dst in local_seen:
                        continue
                    local_seen.add(dst)
                    dst_row = edge_dst_lookup.execute("SELECT kept FROM papers WHERE paper_id=?", (dst,)).fetchone()
                    if dst_row is None or int(dst_row[0]) != 1:
                        unresolved += 1
                        continue
                    written += 1
                    yield [src, dst, year]
                remaining_src_ids.discard(src)
                if not remaining_src_ids:
                    break
            stats["written_edges"] = written
            stats["unresolved_references"] = unresolved

        feature_header = ["node_id"] + [f"f{i + 1}" for i in range(feature_dim)]
        _write_atomic_csv(output_features, feature_header, iter_feature_rows())
        _write_atomic_csv(output_labels, ["node_id", "label"], iter_label_rows())
        _write_atomic_csv(output_edges, ["src", "dst", "time"], iter_edge_rows())

        if include_attr_updates:
            _write_atomic_csv(output_attr_updates, ["time", "node_id"] + [f"f{i + 1}" for i in range(feature_dim)], [])
            stats["written_attr_updates"] = 0
        else:
            _remove_if_exists(output_attr_updates)

        validate_oag_dataset(output_dir)
        return dict(stats)
    finally:
        for cursor in (feature_lookup, edge_src_lookup, edge_dst_lookup):
            if cursor is not None:
                cursor.close()
        conn.close()
        for cleanup_path in (db_path, db_path + "-wal", db_path + "-shm"):
            if os.path.exists(cleanup_path):
                os.remove(cleanup_path)


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
    """命令行入口：转换或检查固定 OAG 数据目录。"""
    parser = argparse.ArgumentParser(description="将 OAG publication zip 转换为固定 CSV，或检查 data/OAG 完整性")
    parser.add_argument("--convert-oag", action="store_true", help="将 dataset/OAG/v5_oag_publication_*.zip 转换为 data/OAG CSV")
    parser.add_argument("--validate-only", action="store_true", help="仅检查 data/OAG/ 下 CSV 是否完整")
    parser.add_argument("--dry-run", action="store_true", help="只扫描统计，不实际写出 CSV")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的 OAG CSV 输出")
    parser.add_argument("--fail-on-malformed", action="store_true", help="遇到损坏 JSON 行时直接失败")
    parser.add_argument("--keep-unlabeled", action="store_true", help="保留未命中高频 venue 的论文，并在 labels.csv 中记为 -1")
    parser.add_argument("--include-attr-updates", action="store_true", help="额外写出空的 attr_updates.csv 占位文件")
    parser.add_argument("--subset-profile", choices=["custom", "test", "small", "medium", "full"], default="custom", help="快速套用 OAG 子集/全量转换参数")
    parser.add_argument("--selection-strategy", choices=["dense", "legacy"], default="dense", help="dense 优先保留引用更活跃的论文；legacy 按旧的前缀截断逻辑")
    parser.add_argument("--candidate-multiplier", type=int, default=2, help="dense 模式下候选池相对 max_papers 的倍数")
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--min-venue-support", type=int, default=50)
    parser.add_argument("--max-papers", type=int, default=0)
    parser.add_argument("--min-year", type=int, default=0)
    parser.add_argument("--max-year", type=int, default=0)
    parser.add_argument("--report-every", type=int, default=100000)
    parser.add_argument("--max-record-bytes", type=int, default=4000000, help="跳过超过该字节数的超大 JSON 行，避免内存峰值")
    parser.add_argument("--input-glob", type=str, default=os.path.join(oag_raw_dir(), "v5_oag_publication_*.zip"))
    parser.add_argument("--output-dir", type=str, default=oag_dataset_dir())
    args = parser.parse_args()

    profile_defaults = resolve_oag_subset_profile(args.subset_profile)
    if args.max_papers == 0 and profile_defaults["max_papers"] > 0:
        args.max_papers = profile_defaults["max_papers"]
    if args.feature_dim == 128:
        args.feature_dim = profile_defaults["feature_dim"]
    if args.min_venue_support == 50:
        args.min_venue_support = profile_defaults["min_venue_support"]
    if args.max_record_bytes == 4000000:
        args.max_record_bytes = profile_defaults["max_record_bytes"]
    if args.candidate_multiplier == 2:
        args.candidate_multiplier = profile_defaults["candidate_multiplier"]

    if args.validate_only or not args.convert_oag:
        dataset_dir = validate_oag_dataset()
        print("OAG 数据检查通过：")
        print(dataset_dir)
        print("必需文件：edges.csv, features.csv, labels.csv")
        print("可选文件：attr_updates.csv")
        if not args.convert_oag:
            return

    stats = convert_oag_archives(
        input_glob=args.input_glob,
        output_dir=args.output_dir,
        feature_dim=args.feature_dim,
        min_venue_support=args.min_venue_support,
        max_papers=args.max_papers,
        min_year=args.min_year,
        max_year=args.max_year,
        keep_unlabeled=args.keep_unlabeled,
        include_attr_updates=args.include_attr_updates,
        overwrite=args.overwrite,
        fail_on_malformed=args.fail_on_malformed,
        dry_run=args.dry_run,
        report_every=args.report_every,
        max_record_bytes=args.max_record_bytes,
        selection_strategy=args.selection_strategy,
        candidate_multiplier=args.candidate_multiplier,
    )
    print("OAG 转换完成：")
    print(f"subset_profile: {args.subset_profile}")
    print(f"selection_strategy: {args.selection_strategy}")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
