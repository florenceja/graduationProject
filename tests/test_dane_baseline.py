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


DANE = _load_module("dane_module", "dane.py").DANE
pipeline = _load_module("pipeline_module_for_dane", "edane_full_pipeline.py")


class TestDaneBaseline(unittest.TestCase):
    def test_parser_accepts_dane_model(self) -> None:
        args = pipeline.build_parser().parse_args(["--model", "dane", "--mode", "synthetic"])
        self.assertEqual(args.model, "dane")

    def test_dane_fit_returns_expected_embedding_shape(self) -> None:
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
        model = DANE(dim=3, attr_topk=2, random_state=7)
        model.fit(adj, attrs)
        emb = model.get_embedding()
        self.assertEqual(emb.shape, (4, 3))
        self.assertTrue(np.all(np.isfinite(emb)))

    def test_dane_state_keeps_raw_eigenvectors(self) -> None:
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
        model = DANE(dim=3, attr_topk=2, random_state=7)
        model.fit(adj, attrs)
        self.assertIsNotNone(model.structure_state_)
        assert model.structure_state_ is not None
        row_norms = np.linalg.norm(model.structure_state_.eigvecs, axis=1)
        self.assertFalse(np.allclose(row_norms, np.ones_like(row_norms), atol=1e-6))

    def test_dane_apply_updates_keeps_pipeline_contract(self) -> None:
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
        model = DANE(dim=2, attr_topk=2, random_state=11)
        model.fit(adj, attrs)
        touched = model.apply_updates(edge_additions=[(0, 2)], attr_updates={3: np.array([0.1, 0.9], dtype=np.float64)})
        self.assertTrue(len(touched) >= 2)
        emb = model.get_embedding()
        self.assertEqual(emb.shape, (4, 2))
        self.assertTrue(np.all(np.isfinite(emb)))

    def test_dane_rejects_edane_only_ablation_flags(self) -> None:
        args = pipeline.build_parser().parse_args(["--model", "dane", "--mode", "synthetic", "--no-attr"])
        with self.assertRaisesRegex(ValueError, "EDANE-specific"):
            pipeline.build_model(args)


if __name__ == "__main__":
    unittest.main()
