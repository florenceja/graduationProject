import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest
import zipfile


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


def _load_module(module_name: str, file_name: str):
    module_path = os.path.join(SRC_ROOT, file_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _load_module("prepare_oag_module", "prepare_datasets.py")


class TestPrepareDatasetsOagConverter(unittest.TestCase):
    def test_resolve_oag_subset_profile_test(self) -> None:
        profile = prepare.resolve_oag_subset_profile("test")
        self.assertEqual(profile["max_papers"], 50000)
        self.assertEqual(profile["feature_dim"], 128)
        self.assertEqual(profile["min_venue_support"], 50)
        self.assertEqual(profile["candidate_multiplier"], 3)

    def _read_csv(self, path: str):
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.reader(f))

    def _write_zip(self, path: str, member_name: str, records: list[dict]) -> None:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
            zf.writestr(member_name, payload)

    def test_convert_oag_archives_writes_pipeline_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            output_dir = os.path.join(tmpdir, "dataset", "OAG")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            self._write_zip(
                os.path.join(input_dir, "v5_oag_publication_1.zip"),
                "v5_oag_publication_1.json",
                [
                    {"id": "p1", "title": "Graph Mining", "abstract": "Dynamic graph embedding", "keywords": ["graph", "mining"], "year": 2020, "references": ["p2"], "venue": "KDD"},
                    {"id": "p2", "title": "Network Learning", "abstract": "Attributed network", "keywords": ["network"], "year": 2019, "references": [], "venue": "KDD"},
                    {"id": "p3", "title": "Ignored", "abstract": "Other venue", "keywords": [], "year": 2021, "references": ["p1"], "venue": "WWW"},
                ],
            )

            stats = prepare.convert_oag_archives(
                input_glob=os.path.join(input_dir, "v5_oag_publication_*.zip"),
                output_dir=output_dir,
                feature_dim=8,
                min_venue_support=2,
                overwrite=True,
            )

            self.assertEqual(stats["kept_papers"], 2)
            self.assertEqual(stats["supported_venues"], 1)

            edges = self._read_csv(os.path.join(output_dir, "edges.csv"))
            features = self._read_csv(os.path.join(output_dir, "features.csv"))
            labels = self._read_csv(os.path.join(output_dir, "labels.csv"))

            self.assertEqual(edges[0], ["src", "dst", "time"])
            self.assertEqual(features[0][0], "node_id")
            self.assertEqual(labels[0], ["node_id", "label"])
            self.assertEqual(len(features) - 1, 2)
            self.assertEqual(len(labels) - 1, 2)
            self.assertEqual(len(edges) - 1, 1)
            self.assertEqual(edges[1], ["p1", "p2", "2020"])
            self.assertTrue(all(row[1] == "0" for row in labels[1:]))

    def test_convert_oag_archives_can_keep_unlabeled_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            output_dir = os.path.join(tmpdir, "dataset", "OAG")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            self._write_zip(
                os.path.join(input_dir, "v5_oag_publication_1.zip"),
                "v5_oag_publication_1.json",
                [
                    {"id": "p1", "title": "Graph", "abstract": "Embedding", "keywords": ["graph"], "year": 2020, "references": ["p2"], "venue": "KDD"},
                    {"id": "p2", "title": "Network", "abstract": "Learning", "keywords": ["network"], "year": 2019, "references": [], "venue": "KDD"},
                    {"id": "p3", "title": "Temporal", "abstract": "Transformer", "keywords": ["time"], "year": 2021, "references": ["p1"], "venue": "WWW"},
                ],
            )

            prepare.convert_oag_archives(
                input_glob=os.path.join(input_dir, "v5_oag_publication_*.zip"),
                output_dir=output_dir,
                feature_dim=8,
                min_venue_support=2,
                keep_unlabeled=True,
                overwrite=True,
            )

            labels = self._read_csv(os.path.join(output_dir, "labels.csv"))
            label_map = {row[0]: row[1] for row in labels[1:]}
            self.assertEqual(label_map["p1"], "0")
            self.assertEqual(label_map["p2"], "0")
            self.assertEqual(label_map["p3"], "-1")

    def test_convert_oag_archives_can_skip_oversized_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            output_dir = os.path.join(tmpdir, "dataset", "OAG")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            oversized_abstract = "x" * 2000
            self._write_zip(
                os.path.join(input_dir, "v5_oag_publication_1.zip"),
                "v5_oag_publication_1.json",
                [
                    {"id": "p1", "title": "Graph", "abstract": "Embedding", "keywords": ["graph"], "year": 2020, "references": ["p2"], "venue": "KDD"},
                    {"id": "p2", "title": "Network", "abstract": "Learning", "keywords": ["network"], "year": 2019, "references": [], "venue": "KDD"},
                    {"id": "p_big", "title": "Huge", "abstract": oversized_abstract, "keywords": [], "year": 2021, "references": [], "venue": "KDD"},
                ],
            )

            stats = prepare.convert_oag_archives(
                input_glob=os.path.join(input_dir, "v5_oag_publication_*.zip"),
                output_dir=output_dir,
                feature_dim=8,
                min_venue_support=2,
                overwrite=True,
                max_record_bytes=500,
            )

            self.assertEqual(stats["kept_papers"], 2)

    def test_dense_selection_keeps_more_edges_than_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            legacy_dir = os.path.join(tmpdir, "legacy")
            dense_dir = os.path.join(tmpdir, "dense")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(legacy_dir, exist_ok=True)
            os.makedirs(dense_dir, exist_ok=True)
            self._write_zip(
                os.path.join(input_dir, "v5_oag_publication_1.zip"),
                "v5_oag_publication_1.json",
                [
                    {"id": "a1", "title": "A1", "abstract": "", "keywords": [], "year": 2020, "references": [], "venue": "kdd"},
                    {"id": "a2", "title": "A2", "abstract": "", "keywords": [], "year": 2020, "references": [], "venue": "kdd"},
                    {"id": "a3", "title": "A3", "abstract": "", "keywords": [], "year": 2020, "references": [], "venue": "kdd"},
                    {"id": "a4", "title": "A4", "abstract": "", "keywords": [], "year": 2020, "references": [], "venue": "kdd"},
                    {"id": "b1", "title": "B1", "abstract": "", "keywords": [], "year": 2021, "references": ["b2", "b3", "b4"], "venue": "kdd"},
                    {"id": "b2", "title": "B2", "abstract": "", "keywords": [], "year": 2021, "references": ["b1", "b3", "b4"], "venue": "kdd"},
                    {"id": "b3", "title": "B3", "abstract": "", "keywords": [], "year": 2021, "references": ["b1", "b2", "b4"], "venue": "kdd"},
                    {"id": "b4", "title": "B4", "abstract": "", "keywords": [], "year": 2021, "references": ["b1", "b2", "b3"], "venue": "kdd"},
                ],
            )

            legacy_stats = prepare.convert_oag_archives(
                input_glob=os.path.join(input_dir, "v5_oag_publication_*.zip"),
                output_dir=legacy_dir,
                feature_dim=8,
                min_venue_support=1,
                max_papers=4,
                overwrite=True,
                selection_strategy="legacy",
            )
            dense_stats = prepare.convert_oag_archives(
                input_glob=os.path.join(input_dir, "v5_oag_publication_*.zip"),
                output_dir=dense_dir,
                feature_dim=8,
                min_venue_support=1,
                max_papers=4,
                overwrite=True,
                selection_strategy="dense",
                candidate_multiplier=2,
            )

            self.assertEqual(legacy_stats["kept_papers"], 4)
            self.assertEqual(dense_stats["kept_papers"], 4)
            self.assertGreater(dense_stats["written_edges"], legacy_stats["written_edges"])


if __name__ == "__main__":
    unittest.main()
