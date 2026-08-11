from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import graph_clustering as gc
import run_graph_multiview_experiments as rgm
import theta_clustering as tc
import theta_features as tf
from train_theta import Episode, deduplicate_within_families


ROOT = Path(__file__).resolve().parent


class ThetaTests(unittest.TestCase):
    def _minimal_case(self, idx: int, fingerprint: str, signature: str = ""):
        base = SimpleNamespace(
            det_vec=np.asarray([1.0, float(idx), 0.5], dtype=np.float32),
            info={"primary_signature": signature, "op_pair": "", "mismatch_type": "", "fatal_file": ""},
        )
        return SimpleNamespace(base=base, fingerprint=fingerprint, evidence_reduced=np.asarray([1.0, idx], dtype=np.float32), context_reduced=np.asarray([idx, 1.0], dtype=np.float32))

    def _write_dataset(self, root: Path) -> Path:
        rows = []
        gold = []
        for idx in range(6):
            case = root / f"case_{idx + 1}"
            case.mkdir(parents=True)
            bug = "bug_a" if idx < 3 else "bug_b"
            signal = "PC mismatch DUT retired : 80001234 register x5" if idx < 3 else "HANDLING_IRQ timeout mcause csr"
            (case / "sim.log").write_text(f"UVM_ERROR {signal}\nTEST FAILED\n", encoding="utf-8")
            (case / "regr.log").write_text(f"Mismatch[1] {signal}\n", encoding="utf-8")
            rows.append({"Case": str(idx + 1), "Regr Log": f"case_{idx + 1}/regr.log", "Sim Log": f"case_{idx + 1}/sim.log", "Trace Log": ""})
            gold.append({"Case": str(idx + 1), "bucket": bug})
        with (root / "input.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        with (root / "gold.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(gold[0])); writer.writeheader(); writer.writerows(gold)
        return root

    def test_candidate_pairs_are_bounded_and_keep_signature_edges(self):
        cases = [self._minimal_case(i, f"hash-{i}", "shared" if i in {0, 9} else "") for i in range(20)]
        pairs, debug = tf.candidate_pairs(cases, top_l=3, full_pair_limit=5, block_size=7)
        self.assertEqual(debug["mode"], "concat")
        self.assertIn((0, 9), pairs)
        self.assertLess(len(pairs), debug["all_pairs"])

    def test_candidate_anchors_use_reference_k(self):
        cases = [self._minimal_case(i, f"hash-{i}") for i in range(24)]
        pairs, debug = tf.candidate_pairs(
            cases,
            top_l=2,
            full_pair_limit=5,
            mode="multiview_anchor",
            reference_k=4,
            anchors_per_cluster=1,
            anchor_cluster_count=2,
            random_state=0,
        )
        self.assertEqual(debug["prototype_clusters"], 4)
        self.assertEqual(debug["prototype_anchors"], 4)
        self.assertGreater(debug["prototype_edges"], 0)
        self.assertLess(len(pairs), debug["all_pairs"])

    def test_family_dedup_keeps_largest_episode(self):
        small = Episode(Path("small"), "small", "old", [self._minimal_case(0, "a")], ["x"])
        large = Episode(Path("large"), "large", "old", [self._minimal_case(0, "a"), self._minimal_case(1, "b")], ["x", "y"])
        output = deduplicate_within_families([small, large])
        by_name = {episode.name: episode for episode in output}
        self.assertEqual(len(by_name["large"].cases), 2)
        self.assertEqual(len(by_name["small"].cases), 0)
        self.assertEqual(by_name["small"].dropped_duplicates, 1)

    def test_sparse_graph_preserves_k_and_respects_conflict(self):
        case_matrix = np.asarray([[1, 0], [.95, .05], [0, 1], [.05, .95]], dtype=np.float32)
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        prob = np.asarray([.99, .2, .2, .2, .2, .95], dtype=np.float32)
        conflict = np.asarray([1.0, 0, 0, 0, 0, 0], dtype=np.float32)
        result = tc.sparse_signed_graph_cluster(
            case_matrix, pairs, prob, 2, conflict,
            conflict_penalty=20.0, balance_weight=0.0, max_iter=10,
        )
        self.assertEqual(len(set(result.labels)), 2)
        self.assertNotEqual(result.labels[0], result.labels[1])

    def test_balanced_signed_graph_preserves_clear_groups(self):
        prob = np.asarray([
            [1.00, 0.92, 0.88, 0.10, 0.12],
            [0.92, 1.00, 0.90, 0.15, 0.10],
            [0.88, 0.90, 1.00, 0.18, 0.16],
            [0.10, 0.15, 0.18, 1.00, 0.91],
            [0.12, 0.10, 0.16, 0.91, 1.00],
        ], dtype=np.float32)
        result = gc.signed_graph_balanced(prob, 2, signed_move_margin=0.1)
        self.assertEqual(len(set(result.labels)), 2)
        self.assertEqual(result.labels[0], result.labels[1])
        self.assertEqual(result.labels[1], result.labels[2])
        self.assertEqual(result.labels[3], result.labels[4])
        self.assertNotEqual(result.labels[0], result.labels[3])

    def test_quality_selector_is_label_free_and_preserves_k(self):
        prob = np.asarray([
            [1.00, 0.90, 0.15, 0.10],
            [0.90, 1.00, 0.12, 0.14],
            [0.15, 0.12, 1.00, 0.88],
            [0.10, 0.14, 0.88, 1.00],
        ], dtype=np.float32)
        result = gc.quality_selected_clustering(prob, 2, selector_balance_weight=0.2)
        self.assertEqual(len(set(result.labels)), 2)
        candidates = [row for row in result.trajectory if row["action"] == "quality_candidate"]
        self.assertEqual(len(candidates), 4)
        self.assertEqual(sum(bool(row["selected"]) for row in candidates), 1)

    def test_multiview_gated_model_smoke(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(32, 12)).astype(np.float32)
        y = np.asarray([0, 1] * 16, dtype=np.float32)
        args = SimpleNamespace(
            view_device="cpu", view_epochs=1, view_batch_size=8,
            view_lr=1e-3, view_weight_decay=1e-4, view_dropout=0.1,
            view_early_stop_patience=1, view_focal_gamma=2.0,
            view_gate_reg=1e-4,
        )
        model = rgm.train_view_model(
            X, y, "gated_mlp", 0, sample_weight=np.ones(len(y)), train_args=args
        )
        prob = rgm.predict_view_probabilities_flat(model, X)
        self.assertEqual(prob.shape, (len(y),))
        self.assertTrue(np.all((prob >= 0.0) & (prob <= 1.0)))

    def test_structure_only_train_and_infer_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._write_dataset(root / "dataset")
            model = root / "model"
            pred = root / "pred.csv"
            subprocess.run([
                sys.executable, str(ROOT / "train_theta.py"),
                "--train-datasets", str(dataset), "--output-dir", str(model),
                "--embedding-mode", "none", "--model-type", "gbdt",
                "--max-pairs-per-family", "200", "--random-state", "0",
            ], check=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            subprocess.run([
                sys.executable, str(ROOT / "theta_inference.py"),
                "--input", str(dataset / "input.csv"), "--output", str(pred),
                "--k", "2", "--model-dir", str(model), "--clusterer", "theta_graph",
            ], check=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            with pred.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertEqual(list(rows[0]), ["Case", "bucket"])
            self.assertEqual(len({row["bucket"] for row in rows}), 2)

    def test_gated_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._write_dataset(root / "dataset")
            model = root / "model"
            pred = root / "pred.csv"
            subprocess.run([
                sys.executable, str(ROOT / "train_theta.py"),
                "--train-datasets", str(dataset), "--output-dir", str(model),
                "--embedding-mode", "none", "--model-type", "gated_mlp",
                "--max-pairs-per-family", "200", "--epochs", "2",
                "--batch-size", "8", "--random-state", "0",
            ], check=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertTrue((model / "pair_student.pt").exists())
            subprocess.run([
                sys.executable, str(ROOT / "theta_inference.py"),
                "--input", str(dataset / "input.csv"), "--output", str(pred),
                "--k", "2", "--model-dir", str(model), "--clusterer", "average",
            ], check=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            with pred.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertEqual(len({row["bucket"] for row in rows}), 2)


if __name__ == "__main__":
    unittest.main()
