#!/usr/bin/env python3

from __future__ import annotations

import csv
import gzip
import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

import pairwise_llm_features as plf
import theta_trace_features as ttf
import theta_trilog_model as ttm
import train_pairwise_llm as tpl


def _trace_lines(count: int, anchor_index: int) -> list[str]:
    lines = ["Time\tCycle\tPC\tInsn\tDecoded instruction\tRegister and memory contents"]
    opcodes = ("addi", "lw", "sw", "beq", "csrrw", "mret")
    for idx in range(count):
        pc = "80000100" if idx == anchor_index else f"8000{idx:04x}"
        opcode = opcodes[idx % len(opcodes)]
        lines.append(
            f"{1000 + idx * 10}\t{idx}\t{pc}\t00000013\t{opcode}\tx1,x2,4 PA:0x80002000"
        )
    return lines


class HierarchicalTraceFeatureTest(unittest.TestCase):
    def test_trilog_two_tower_forward(self) -> None:
        import torch

        model = ttm._build_network(input_dim=18, base_dim=10, dropout=0.1)
        logits = model(torch.zeros((4, 18), dtype=torch.float32))
        self.assertEqual((4,), tuple(logits.shape))
        self.assertTrue(torch.isfinite(logits).all())

    def test_official_pair_adaptation_round_trip(self) -> None:
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(7)
        pretrain_x = rng.normal(size=(20, 18)).astype(np.float32)
        scaler = StandardScaler().fit(pretrain_x)
        package = {
            "model": ttm._build_network(input_dim=18, base_dim=10, dropout=0.0).eval(),
            "scaler": scaler,
            "model_type": "theta_trilog_mlp",
        }
        official_x = rng.normal(size=(6, 18)).astype(np.float32)
        official_y = np.asarray([1, 0, 0, 0, 0, 1], dtype=np.float32)
        official_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        args = argparse.Namespace(
            random_state=0, device="cpu", finetune_scope="last",
            finetune_epochs=2, finetune_lr=1e-3, finetune_weight_decay=0.0,
            label_smoothing=0.02, affine_reg=0.2,
            ranking_weight=0.2, ranking_margin=0.5,
            connectivity_weight=0.1, connectivity_top_m=1,
            transitivity_weight=0.05, replay_weight=0.3,
        )
        affine = ttm.fit_official_affine_calibration(
            package, official_x, official_y, np.ones(6, dtype=np.float32), args
        )
        affine_probability = ttm.predict_trilog_pair_model(affine, official_x)
        self.assertEqual((6,), affine_probability.shape)
        self.assertTrue(np.isfinite(affine_probability).all())
        adapted = ttm.fine_tune_official_pair_model(
            package, official_x, official_y, official_pairs,
            np.ones(6, dtype=np.float32), pretrain_x[:8], args,
        )
        adapted_probability = ttm.predict_trilog_pair_model(adapted, official_x)
        self.assertEqual((6,), adapted_probability.shape)
        self.assertTrue(np.isfinite(adapted_probability).all())

    def test_capped_sampler_reserves_negative_budget(self) -> None:
        features = []
        for idx in range(12):
            vec = np.asarray([float(idx), 1.0], dtype=np.float32)
            features.append(plf.LLMCaseFeature(
                case_id=str(idx),
                det_vec=vec,
                llm_vec=vec,
                llm_vec_reduced=None,
                llm_summary_vec=vec,
                llm_summary_vec_reduced=None,
                trace_vec=np.zeros(0, dtype=np.float32),
                trace_vec_reduced=None,
                tokens=[],
                token_set=set(),
                primary_tokens=set(),
                sim_tokens=set(),
                regr_tokens=set(),
                info={"primary_signature": f"sig_{idx % 3}"},
            ))
        labels = ["bug_a"] * 6 + ["bug_b"] * 6
        _, y, stats = tpl.sample_pairs(
            features,
            labels,
            negative_ratio=2.0,
            hard_negative_ratio=0.5,
            hard_positive_ratio=1.0,
            max_train_pairs=30,
            random_state=0,
            positive_sampling="diverse",
            negative_sampling="confusable",
        )
        self.assertEqual(30, len(y))
        self.assertEqual(10, int(np.sum(y == 1.0)))
        self.assertEqual(20, int(np.sum(y == 0.0)))
        self.assertEqual(20, stats["negative_pairs"])

    def test_connectivity_sampler_covers_each_non_singleton_case(self) -> None:
        features = []
        for idx in range(10):
            vec = np.asarray([float(idx), float(idx % 2), 1.0], dtype=np.float32)
            features.append(plf.LLMCaseFeature(
                case_id=str(idx), det_vec=vec, llm_vec=vec,
                llm_vec_reduced=None, llm_summary_vec=vec,
                llm_summary_vec_reduced=None,
                trace_vec=np.zeros(0, dtype=np.float32), trace_vec_reduced=None,
                tokens=[], token_set=set(), primary_tokens=set(), sim_tokens=set(),
                regr_tokens=set(), info={"primary_signature": f"sig_{idx}"},
            ))
        labels = ["bug_a"] * 5 + ["bug_b"] * 5
        pairs, y, stats = tpl.sample_pairs(
            features, labels, negative_ratio=1.0,
            hard_negative_ratio=0.0, hard_positive_ratio=0.0,
            max_train_pairs=24, random_state=3,
            positive_sampling="diverse", negative_sampling="confusable",
            connectivity_positive_fraction=1.0,
        )
        positive_pairs = [pair for pair, label in zip(pairs, y) if label > 0.5]
        covered = {index for pair in positive_pairs for index in pair}
        self.assertEqual(set(range(10)), covered)
        self.assertGreaterEqual(stats["connectivity_positive_selected"], 8)

    def test_full_scan_anchor_and_pair_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for case_id, compressed in (("1", False), ("2", True)):
                case_dir = root / f"case_{case_id}"
                case_dir.mkdir()
                (case_dir / "sim.log").write_text(
                    "UVM_FATAL @ 1050: PC mismatch, DUT retired : 80000100\n",
                    encoding="utf-8",
                )
                (case_dir / "regr.log").write_text(
                    "PC mismatch, DUT retired : 80000100, ISS retired: 80000200\n",
                    encoding="utf-8",
                )
                trace_name = "trace.log.gz" if compressed else "trace.log"
                trace_path = case_dir / trace_name
                text = "\n".join(_trace_lines(40 + int(case_id), 7)) + "\n"
                if compressed:
                    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
                        handle.write(text)
                else:
                    trace_path.write_text(text, encoding="utf-8")
                rows.append({
                    "Case": case_id,
                    "Regr Log": f"case_{case_id}/regr.log",
                    "Sim Log": f"case_{case_id}/sim.log",
                    "Trace Log": f"case_{case_id}/{trace_name}",
                })
            with (root / "input.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            features, debug = ttf.build_hierarchical_trace_features(
                root / "input.csv",
                cache_dir=root / "cache",
                segment_count=4,
                chunk_size=8,
                anchor_sizes=(2, 4, 8),
            )
            self.assertEqual([41, 42], [feature.instruction_count for feature in features])
            self.assertTrue(all(feature.file_status == "ok" for feature in features))
            self.assertTrue(all(feature.located_by == "pc" for feature in features))
            self.assertTrue(all(feature.target_pc_present for feature in features))
            self.assertEqual((4, len(ttf.TRACE_CLASSES) + 4), features[0].segment_matrix.shape)
            self.assertEqual(2, len(debug))

            _, matrices = ttf.fit_transform_trace_views(
                features,
                train_indices=[0, 1],
                global_struct_dim=4,
                global_text_dim=4,
                anchor_struct_dim=2,
                anchor_text_dim=2,
            )
            pair_matrix = ttf.build_trace_pair_feature_matrix(features, matrices, [(0, 1)], mode="full")
            components = ttf.build_trace_pair_feature_components(features, matrices, [(0, 1)])
            self.assertEqual(1, pair_matrix.shape[0])
            self.assertGreater(pair_matrix.shape[1], 20)
            self.assertTrue(np.isfinite(pair_matrix).all())
            np.testing.assert_allclose(pair_matrix, components["full"], atol=1e-6)
            self.assertGreater(components["residual"].shape[1], 20)
            self.assertTrue(np.isfinite(components["full_residual"]).all())

            _, cached_debug = ttf.build_hierarchical_trace_features(
                root / "input.csv",
                cache_dir=root / "cache",
                segment_count=4,
                chunk_size=8,
                anchor_sizes=(2, 4, 8),
            )
            self.assertTrue(all(int(row["cache_hit"]) == 1 for row in cached_debug))

    def test_missing_trace_is_valid_fallback_feature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sim.log").write_text("UVM_FATAL @ 10: timeout\n", encoding="utf-8")
            (root / "regr.log").write_text("[FAILED]: timeout\n", encoding="utf-8")
            (root / "input.csv").write_text(
                "Case,Regr Log,Sim Log,Trace Log\n1,regr.log,sim.log,missing.log.gz\n",
                encoding="utf-8",
            )
            features, _ = ttf.build_hierarchical_trace_features(root / "input.csv", cache_dir=root / "cache")
            self.assertEqual(1, len(features))
            self.assertFalse(features[0].has_trace)
            self.assertEqual("missing", features[0].file_status)
            self.assertEqual("TRACE_MISSING", features[0].global_document)


if __name__ == "__main__":
    unittest.main()
