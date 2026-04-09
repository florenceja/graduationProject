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


pipeline = _load_module("edane_pipeline_oag_module", "edane_full_pipeline.py")
stage23 = _load_module("stage23_oag_module", "run_stage23_experiments.py")


class TestEdaneFileModeOag(unittest.TestCase):
    def test_resolve_oag_dataset_paths_requires_fixed_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "data/OAG"):
                pipeline._resolve_oag_dataset_paths(tmpdir)

    def test_resolve_oag_dataset_paths_returns_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            oag_dir = os.path.join(tmpdir, "data", "OAG")
            os.makedirs(oag_dir, exist_ok=True)
            for name in ["edges.csv", "features.csv", "labels.csv"]:
                with open(os.path.join(oag_dir, name), "w", encoding="utf-8") as f:
                    f.write("stub\n")
            paths = pipeline._resolve_oag_dataset_paths(tmpdir)
            self.assertEqual(os.path.basename(paths[0]), "edges.csv")
            self.assertEqual(os.path.basename(paths[1]), "features.csv")
            self.assertEqual(os.path.basename(paths[2]), "labels.csv")
            self.assertEqual(paths[3], "")

    def test_stage23_file_mode_no_longer_uses_dataset_preset(self) -> None:
        args = stage23.build_parser().parse_args(["--mode", "file", "--model", "edane"])
        common = stage23._build_common_args(args)
        self.assertNotIn("--dataset-preset", common)


if __name__ == "__main__":
    unittest.main()
