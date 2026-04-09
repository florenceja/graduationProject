import csv
import importlib.util
import os
import sys
import tempfile
import unittest


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


build_graph_from_files = _load_module("edane_pipeline_module", "edane_full_pipeline.py").build_graph_from_files


class TestEdaneFullPipelineIngest(unittest.TestCase):
    def _write_csv(self, path: str, header: list[str], rows: list[list[object]]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    def test_rejects_non_finite_feature_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            edges_path = os.path.join(tmpdir, "edges.csv")
            features_path = os.path.join(tmpdir, "features.csv")
            self._write_csv(edges_path, ["src", "dst", "time"], [["u1", "u2", 1], ["u2", "u3", 2]])
            self._write_csv(features_path, ["node_id", "f1", "f2"], [["u1", "nan", 1.0]])

            with self.assertRaisesRegex(ValueError, "NaN 或 Inf"):
                build_graph_from_files(
                    edges_path=edges_path,
                    features_path=features_path,
                    labels_path=None,
                    attr_updates_path=None,
                    snapshots=3,
                    snapshot_mode="window",
                )

    def test_rejects_non_finite_attr_update_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            edges_path = os.path.join(tmpdir, "edges.csv")
            features_path = os.path.join(tmpdir, "features.csv")
            updates_path = os.path.join(tmpdir, "attr_updates.csv")
            self._write_csv(edges_path, ["src", "dst", "time"], [["u1", "u2", 1], ["u2", "u3", 2]])
            self._write_csv(
                features_path,
                ["node_id", "f1", "f2"],
                [["u1", 1.0, 2.0], ["u2", 3.0, 4.0], ["u3", 5.0, 6.0]],
            )
            self._write_csv(updates_path, ["time", "node_id", "f1", "f2"], [[2, "u1", "inf", 5.0]])

            with self.assertRaisesRegex(ValueError, "NaN 或 Inf"):
                build_graph_from_files(
                    edges_path=edges_path,
                    features_path=features_path,
                    labels_path=None,
                    attr_updates_path=updates_path,
                    snapshots=3,
                    snapshot_mode="window",
                )


if __name__ == "__main__":
    unittest.main()
