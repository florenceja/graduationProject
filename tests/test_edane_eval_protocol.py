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


pipeline = _load_module("edane_eval_pipeline_module", "edane_full_pipeline.py")


class TestEdaneEvaluationProtocol(unittest.TestCase):
    def test_softmax_logreg_predict_accepts_balanced_class_weight(self) -> None:
        train_x = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [0.1, 0.1],
                [0.2, 0.1],
                [0.8, 0.9],
                [0.9, 0.8],
            ],
            dtype=np.float64,
        )
        train_y = np.array([0, 0, 0, 0, 0, 1, 1], dtype=np.int64)
        test_x = np.array([[0.85, 0.85], [0.05, 0.05]], dtype=np.float64)

        pred = pipeline.softmax_logreg_predict(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            epochs=60,
            lr=0.25,
            weight_decay=1e-4,
            seed=42,
            class_weight_mode="balanced",
        )

        np.testing.assert_array_equal(pred, np.array([1, 0], dtype=np.int64))

    def test_prepare_labels_for_evaluation_filters_rare_classes(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 3, -1], dtype=np.int64)
        filtered, meta = pipeline.prepare_labels_for_evaluation(labels, cleanup_mode="eval_only", min_class_support=3)

        np.testing.assert_array_equal(filtered, np.array([0, 0, 0, 1, 1, 1, -1, -1, -1], dtype=np.int64))
        self.assertEqual(meta["labeled_nodes_raw"], 8.0)
        self.assertEqual(meta["eval_dropped_labeled_nodes"], 2.0)
        self.assertEqual(meta["eval_class_count_raw"], 4.0)
        self.assertEqual(meta["eval_class_count"], 2.0)

    def test_prepare_labels_for_evaluation_drops_all_when_two_classes_not_left(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 2, -1], dtype=np.int64)
        filtered, meta = pipeline.prepare_labels_for_evaluation(labels, cleanup_mode="eval_only", min_class_support=3)

        np.testing.assert_array_equal(filtered, np.array([-1, -1, -1, -1, -1, -1, -1], dtype=np.int64))
        self.assertEqual(meta["eval_class_count"], 0.0)
        self.assertEqual(meta["eval_dropped_class_count"], 3.0)

    def test_stratified_train_test_split_covers_each_class(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
        indices = np.arange(len(labels), dtype=np.int64)
        split = pipeline.stratified_train_test_split(labels, indices, train_ratio=0.67, seed=7)
        self.assertIsNotNone(split)
        train_idx, test_idx = split or (np.array([]), np.array([]))

        self.assertEqual(set(labels[train_idx].tolist()), {0, 1, 2})
        self.assertEqual(set(labels[test_idx].tolist()), {0, 1, 2})

    def test_evaluate_snapshot_reports_repeated_stratified_stats(self) -> None:
        embedding = np.array(
            [
                [3.0, 0.0],
                [2.9, 0.1],
                [3.1, -0.1],
                [3.2, 0.05],
                [0.0, 3.0],
                [0.1, 2.9],
                [-0.1, 3.1],
                [0.05, 3.2],
                [-3.0, 0.0],
                [-2.9, -0.1],
                [-3.1, 0.1],
                [-3.2, 0.05],
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
        adj = np.array(
            [
                [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            ],
            dtype=np.float64,
        )

        metrics = pipeline.evaluate_snapshot(
            embedding=embedding,
            labels=labels,
            adj=adj,
            seed=42,
            classifier="logreg",
            logreg_epochs=80,
            logreg_lr=0.25,
            logreg_weight_decay=1e-4,
            eval_protocol="repeated_stratified",
            eval_repeats=5,
            eval_train_ratio=0.67,
            label_cleanup_mode="off",
            min_class_support=5,
            logreg_class_weight="none",
        )

        self.assertEqual(metrics["f1_eval_protocol"], "repeated_stratified")
        self.assertEqual(metrics["f1_eval_repeats"], 5.0)
        self.assertEqual(metrics["f1_eval_successful_repeats"], 5.0)
        self.assertGreater(metrics["macro_f1"], 0.95)
        self.assertGreater(metrics["micro_f1"], 0.95)
        self.assertGreaterEqual(metrics["macro_f1_std"], 0.0)
        self.assertGreaterEqual(metrics["micro_f1_std"], 0.0)

    def test_evaluate_snapshot_cleanup_can_preserve_stratified_protocol(self) -> None:
        embedding = np.array(
            [
                [3.0, 0.0],
                [2.9, 0.1],
                [3.1, -0.1],
                [3.2, 0.05],
                [0.0, 3.0],
                [0.1, 2.9],
                [0.0, 2.8],
                [0.05, 3.1],
                [-3.0, 0.0],
                [-2.9, -0.1],
                [-3.1, 0.1],
                [-3.2, 0.05],
                [1.5, 1.5],
                [1.6, 1.4],
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4], dtype=np.int64)
        adj = np.eye(14, dtype=np.float64)

        metrics = pipeline.evaluate_snapshot(
            embedding=embedding,
            labels=labels,
            adj=adj,
            seed=11,
            classifier="logreg",
            logreg_epochs=60,
            logreg_lr=0.25,
            logreg_weight_decay=1e-4,
            eval_protocol="repeated_stratified",
            eval_repeats=4,
            eval_train_ratio=0.67,
            label_cleanup_mode="eval_only",
            min_class_support=3,
            logreg_class_weight="balanced",
        )

        self.assertEqual(metrics["label_cleanup_mode"], "eval_only")
        self.assertEqual(metrics["eval_class_count_raw"], 5.0)
        self.assertEqual(metrics["eval_class_count"], 3.0)
        self.assertEqual(metrics["eval_dropped_class_count"], 2.0)
        self.assertEqual(metrics["f1_eval_protocol"], "repeated_stratified")
        self.assertEqual(metrics["f1_eval_successful_repeats"], 4.0)


if __name__ == "__main__":
    unittest.main()
