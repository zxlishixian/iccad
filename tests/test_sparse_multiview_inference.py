from __future__ import annotations

import gzip
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from bounded_evidence import read_bounded_sample
from sparse_multiview_inference import (
    build_active_documents,
    centroid_sparse_plan,
    sparse_refine_from_edges,
)


class SparseMultiviewInferenceTests(unittest.TestCase):
    def test_dual_active_documents_skip_custom_views(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            input_csv = root / "input.csv"
            input_csv.write_text(
                "Case,Regr Log,Sim Log,Trace Log\n1,a/regr.log,a/sim.log,a/trace.log\n",
                encoding="utf-8",
            )
            with patch(
                "sparse_multiview_inference.bmi.build_feature_documents",
                return_value=(["feature"], ["summary"]),
            ), patch(
                "sparse_multiview_inference.gm.build_all_view_documents"
            ) as custom:
                docs = build_active_documents(
                    input_csv, np.asarray([0]), "drain", 80, "dual"
                )
            self.assertEqual(set(docs), {"features", "summary"})
            custom.assert_not_called()

    def test_five_active_documents_keep_custom_views(self):
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            input_csv = root / "input.csv"
            input_csv.write_text(
                "Case,Regr Log,Sim Log,Trace Log\n1,a/regr.log,a/sim.log,a/trace.log\n",
                encoding="utf-8",
            )
            custom_docs = {name: [name] for name in ("event", "object", "context")}
            with patch(
                "sparse_multiview_inference.bmi.build_feature_documents",
                return_value=(["feature"], ["summary"]),
            ), patch(
                "sparse_multiview_inference.gm.build_all_view_documents",
                return_value=custom_docs,
            ) as custom:
                docs = build_active_documents(
                    input_csv, np.asarray([0]), "drain", 80, "five"
                )
            self.assertEqual(
                set(docs), {"features", "summary", "event", "object", "context"}
            )
            custom.assert_called_once()

    def test_centroid_plan_is_bounded_and_keeps_each_cluster_anchored(self):
        rng = np.random.default_rng(7)
        labels = np.repeat(np.arange(4), 25).astype(np.int32)
        vectors = rng.normal(size=(100, 16)).astype(np.float32)
        vectors += np.eye(4, 16, dtype=np.float32)[labels] * 3.0
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        selected, anchors, stats = centroid_sparse_plan(labels, vectors, 0.50, 12, 1)
        self.assertEqual(int(np.sum(selected)), 12)
        self.assertEqual(set(anchors), {0, 1, 2, 3})
        self.assertTrue(all(len(values) == 1 for values in anchors.values()))
        self.assertTrue(all(not selected[idx] for values in anchors.values() for idx in values))
        self.assertLessEqual(stats["difficulty_min"], stats["difficulty_max"])

    def test_sparse_edge_refinement_moves_only_supported_case(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
        vectors = np.asarray([
            [1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8],
        ], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        selected = np.asarray([False, True, False, False])
        anchors = {0: [0], 1: [2]}
        edges = {(0, 1): 0.10, (1, 2): 0.95}
        refined, stats = sparse_refine_from_edges(
            labels, vectors, edges, selected, anchors,
            expert_weight=1.0, min_probability=0.60, margin=0.20,
        )
        self.assertEqual(refined.tolist(), [0, 1, 1, 1])
        self.assertEqual(stats["moved_cases"], 1)
        self.assertEqual(stats["expert_edges"], 2)

    def test_sparse_edge_refinement_preserves_singleton_cluster(self):
        labels = np.asarray([0, 1, 1], dtype=np.int32)
        vectors = np.eye(3, dtype=np.float32)
        selected = np.asarray([True, False, False])
        anchors = {0: [0], 1: [1]}
        refined, stats = sparse_refine_from_edges(
            labels, vectors, {(0, 1): 0.99}, selected, anchors,
            expert_weight=1.0, min_probability=0.50, margin=0.0,
        )
        self.assertEqual(refined.tolist(), labels.tolist())
        self.assertEqual(stats["moved_cases"], 0)


    def test_bounded_gzip_reader_stops_at_limit(self):
        with TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "sim.log.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write("UVM_INFO repeated line\n" * 10000)
            text, status = read_bounded_sample(path, max_bytes=4096)
        self.assertEqual(len(text), 4096)
        self.assertEqual(status, "ok_head_truncated")

    def test_structured_conflict_vetoes_move(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
        vectors = np.asarray([
            [1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8],
        ], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        selected = np.asarray([False, True, False, False])
        infos = [
            {"primary_type": "UVM_FATAL"},
            {"primary_type": "UVM_FATAL"},
            {"primary_type": "UVM_ERROR"},
            {"primary_type": "UVM_ERROR"},
        ]
        refined, stats = sparse_refine_from_edges(
            labels, vectors, {(0, 1): 0.10, (1, 2): 0.95}, selected,
            {0: [0], 1: [2]}, expert_weight=1.0,
            min_probability=0.60, margin=0.20,
            structured_infos=infos, max_conflict_ratio=0.0,
        )
        self.assertEqual(refined.tolist(), labels.tolist())
        self.assertEqual(stats["rejected_conflict"], 1)

    def test_min_support_requires_multiple_expert_edges(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
        vectors = np.asarray([
            [1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8],
        ], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        selected = np.asarray([False, True, False, False])
        refined, stats = sparse_refine_from_edges(
            labels, vectors,
            {(0, 1): 0.10, (1, 2): 0.95, (1, 3): 0.40}, selected,
            {0: [0], 1: [2, 3]}, expert_weight=1.0,
            min_probability=0.60, margin=0.20,
            min_support=2, support_probability=0.80,
        )
        self.assertEqual(refined.tolist(), labels.tolist())
        self.assertEqual(stats["rejected_support"], 1)


if __name__ == "__main__":
    unittest.main()
