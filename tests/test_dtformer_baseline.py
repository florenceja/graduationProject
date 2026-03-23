import importlib.util
import os
import sys
import unittest

import numpy as np


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


DTFormer = _load_module("dtformer_module", "dtformer.py").DTFormer
pipeline = _load_module("pipeline_module_for_dtformer", "edane_full_pipeline.py")


class TestDTFormerBaseline(unittest.TestCase):
    def test_parser_accepts_dtformer_model(self) -> None:
        args = pipeline.build_parser().parse_args(["--model", "dtformer", "--mode", "synthetic"])
        self.assertEqual(args.model, "dtformer")

    def test_dtformer_fit_returns_expected_embedding_shape(self) -> None:
        adj = np.array(
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        attrs = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.1, 0.8, 0.1],
                [0.0, 0.1, 0.9],
            ],
            dtype=np.float64,
        )
        model = DTFormer(dim=3, patch_size=2, history_snapshots=4, transformer_hidden_dim=6, random_state=7)
        model.fit(adj, attrs)
        emb = model.get_embedding()
        self.assertEqual(emb.shape, (4, 3))
        self.assertTrue(np.all(np.isfinite(emb)))
        self.assertFalse(model.supports_incremental_updates_)

    def test_dtformer_apply_updates_refits_history(self) -> None:
        adj = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        attrs = np.array(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        model = DTFormer(dim=2, patch_size=2, history_snapshots=4, transformer_hidden_dim=4, random_state=11)
        model.fit(adj, attrs)
        touched = model.apply_updates(edge_additions=[(0, 2)], attr_updates={3: np.array([0.1, 0.9], dtype=np.float64)})
        self.assertTrue(len(touched) >= 2)
        self.assertEqual(model.online_update_mode_, "refit")
        emb = model.get_embedding()
        self.assertEqual(emb.shape, (4, 2))
        self.assertTrue(np.all(np.isfinite(emb)))

    def test_dtformer_rejects_edane_only_ablation_flags(self) -> None:
        args = pipeline.build_parser().parse_args(["--model", "dtformer", "--mode", "synthetic", "--no-attr"])
        with self.assertRaisesRegex(ValueError, "EDANE-specific"):
            pipeline.build_model(args)

    def test_dtformer_summary_uses_honest_metadata(self) -> None:
        quantize, binary_quantize = pipeline._quantization_enabled_for_model(
            pipeline.build_parser().parse_args(["--model", "dtformer", "--mode", "synthetic", "--quantize", "--binary-quantize"])
        )
        self.assertFalse(quantize)
        self.assertFalse(binary_quantize)


if __name__ == "__main__":
    unittest.main()
