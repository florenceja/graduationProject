import os
import sys
import unittest
import importlib.util

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


EDANE = _load_module("edane_module", "edane.py").EDANE


class TestEdaneAttrUpdates(unittest.TestCase):
    def test_attr_updates_reuse_fit_standardization(self) -> None:
        adj = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        attrs = np.array(
            [
                [1.0, 10.0],
                [3.0, 20.0],
                [5.0, 30.0],
            ],
            dtype=np.float64,
        )
        model = EDANE(dim=4, quantize=False, random_state=0)
        model.fit(adj, attrs)

        updated_raw = np.array([7.0, 40.0], dtype=np.float64)
        model.apply_updates(attr_updates={1: updated_raw})

        expected = (updated_raw - attrs.mean(axis=0)) / np.maximum(attrs.std(axis=0), 1e-12)
        np.testing.assert_allclose(model.attrs[1], expected, rtol=1e-7, atol=1e-7)

    def test_attr_updates_reject_non_finite_values(self) -> None:
        adj = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        attrs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        model = EDANE(dim=4, quantize=False, random_state=0)
        model.fit(adj, attrs)

        with self.assertRaisesRegex(ValueError, "finite"):
            model.apply_updates(attr_updates={0: np.array([np.nan, 1.0], dtype=np.float64)})


if __name__ == "__main__":
    unittest.main()
