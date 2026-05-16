#!/usr/bin/env python3
"""Pairwise error analysis for calibrated blend predictions.

Experimental analysis script: reads gold.csv for evaluation only. Case metadata is
limited to sim.log/regr.log-derived features produced by pairwise_llm_features.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import pairwise_llm_features as plf
from run_experiments import pairwise_scores, read_gold
from run_half_split_experiments import DEFAULT_DATASETS
from run_input_signal_calibration import _find_ensemble_models, _find_model, _temperature
from run_input_signal_experiments import ENSEMBLE_WEIGHTS, build_val_parts

PROJECT_ROOT = Path(__file__).resolve().parent


def _write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _quantiles(vals: Sequence[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0, "mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def _info(features: Sequence[object], idx: int, key: str) -> str:
    return str((getattr(features[idx], "info", {}) or {}).get(key, "") or "")


def _case_ids(n: int) -> list[str]:
    return [f"case_{idx + 1:06d}" for idx in range(n)]


def analyze(args: argparse.Namespace) -> tuple[list[dict], dict[str, list[str]]]:
    datasets = [(d if d.is_absolute() else (PROJECT_ROOT / d).resolve()) for d in args.datasets]
    error_rows: list[dict] = []
    report_lines: dict[str, list[str]] = defaultdict(list)
    aggregates: dict[str, dict] = {}

    for ds in datasets:
        aggregates[ds.name] = {
            "scores": [], "pos_probs": [], "neg_probs": [], "fn_probs": [], "fp_probs": [],
            "fn_bug_counts": Counter(), "fp_bug_pair_counts": Counter(),
            "fragmented": Counter(), "mixed": Counter(),
            "fn_primary": Counter(), "fn_mismatch": Counter(), "fn_op": Counter(), "fn_fatal": Counter(),
            "fp_primary": Counter(), "fp_mismatch": Counter(), "fp_op": Counter(), "fp_fatal": Counter(),
        }

    for seed in args.seeds:
        model_path = _find_model(args.model_root, args.model_tag, seed, args.combo)
        rich_model = plf.load_model_pkg(model_path)
        feature_mode = str(rich_model.get("feature_mode", ""))
        rich_args = plf._make_llm_args("embedding", llm_doc_style="features", llm_cache_dir=args.llm_cache_dir, svd_dim=args.svd_dim, llm_dual=feature_mode in plf.DUAL_FEATURE_MODES)
        ensemble_args = plf._make_llm_args("embedding", llm_doc_style="features", llm_cache_dir=args.llm_cache_dir, svd_dim=args.svd_dim)
        ensemble_pkgs = [plf.load_model_pkg(path) for path in _find_ensemble_models(args.ensemble_model_dir, seed, args.combo)]
        val_parts = build_val_parts(datasets, seed, args.combo, args.split_root / f"seed_{seed}")
        for part in val_parts:
            ds = part["dataset"]
            features, _ = plf.build_llm_case_features(part["input"], svd_dim=args.svd_dim, llm_args=rich_args)
            p_rich = _temperature(plf.predict_probability_matrix_sklearn(rich_model, features, batch_size=args.predict_batch_size), args.rich_temperature)
            ensemble_features, _ = plf.build_llm_case_features(part["input"], svd_dim=args.svd_dim, llm_args=ensemble_args)
            p_ens = _temperature(plf.predict_probability_matrix_ensemble(ensemble_pkgs, list(ENSEMBLE_WEIGHTS), ensemble_features, ensemble_mode="prob_average", batch_size=args.predict_batch_size), args.ensemble_temperature)
            prob = float(args.alpha) * p_rich + (1.0 - float(args.alpha)) * p_ens
            labels = plf.cluster_from_probability(prob.astype(np.float32), part["k"])
            pred = [f"bucket_{x:03d}" for x in labels]
            gold = read_gold(part["gold"])
            ba, tpr, tnr = pairwise_scores(gold, pred)
            agg = aggregates[ds]
            agg["scores"].append((ba, tpr, tnr))
            n = len(gold)
            case_ids = _case_ids(n)

            by_bug: dict[str, set[str]] = defaultdict(set)
            by_bucket: dict[str, list[int]] = defaultdict(list)
            for idx, (g, pr) in enumerate(zip(gold, pred)):
                by_bug[g].add(pr)
                by_bucket[pr].append(idx)
            for bug, buckets in by_bug.items():
                if len(buckets) > 1:
                    agg["fragmented"][bug] += len(buckets) - 1
            for bucket, idxs in by_bucket.items():
                counts = Counter(gold[i] for i in idxs)
                if len(counts) > 1:
                    purity = counts.most_common(1)[0][1] / len(idxs)
                    agg["mixed"][f"{bucket} purity={purity:.2f} size={len(idxs)}"] += len(idxs) - counts.most_common(1)[0][1]

            for i in range(n):
                for j in range(i + 1, n):
                    same_gold = gold[i] == gold[j]
                    same_pred = pred[i] == pred[j]
                    p = float(prob[i, j])
                    if same_gold:
                        agg["pos_probs"].append(p)
                    else:
                        agg["neg_probs"].append(p)
                    if same_gold and not same_pred:
                        agg["fn_probs"].append(p)
                        agg["fn_bug_counts"][gold[i]] += 1
                        for key, counter_name in [("primary_signature", "fn_primary"), ("mismatch_type", "fn_mismatch"), ("op_pair", "fn_op"), ("fatal_file", "fn_fatal")]:
                            agg[counter_name][f"{_info(features,i,key)} | {_info(features,j,key)}"] += 1
                        if len(error_rows) < args.max_error_pairs:
                            error_rows.append(_pair_row(ds, seed, "FN", i, j, case_ids, gold, pred, prob, features))
                    elif not same_gold and same_pred:
                        agg["fp_probs"].append(p)
                        pair = tuple(sorted((gold[i], gold[j])))
                        agg["fp_bug_pair_counts"][pair] += 1
                        for key, counter_name in [("primary_signature", "fp_primary"), ("mismatch_type", "fp_mismatch"), ("op_pair", "fp_op"), ("fatal_file", "fp_fatal")]:
                            agg[counter_name][f"{_info(features,i,key)} | {_info(features,j,key)}"] += 1
                        if len(error_rows) < args.max_error_pairs:
                            error_rows.append(_pair_row(ds, seed, "FP", i, j, case_ids, gold, pred, prob, features))

    for ds, agg in aggregates.items():
        scores = agg["scores"]
        bas = [x[0] for x in scores]; tprs = [x[1] for x in scores]; tnrs = [x[2] for x in scores]
        lines = [f"# Pairwise Error Report: {ds}", ""]
        lines.append(f"BA={statistics.mean(bas):.6f}, TPR={statistics.mean(tprs):.6f}, TNR={statistics.mean(tnrs):.6f}, runs={len(scores)}")
        lines.append("")
        lines.append("## Probability Distributions")
        for name in ["pos_probs", "neg_probs", "fn_probs", "fp_probs"]:
            q = _quantiles(agg[name])
            lines.append(f"- {name}: n={q['n']} mean={q['mean']:.4f} p10={q['p10']:.4f} p50={q['p50']:.4f} p90={q['p90']:.4f}")
        lines.append("")
        lines.append("## Top Fragmented Gold Bugs")
        lines.extend([f"- {k}: {v}" for k, v in agg["fragmented"].most_common(10)] or ["- none"])
        lines.append("")
        lines.append("## Top Mixed Predicted Buckets")
        lines.extend([f"- {k}: {v}" for k, v in agg["mixed"].most_common(10)] or ["- none"])
        lines.append("")
        lines.append("## Top FN Bugs")
        lines.extend([f"- {k}: {v}" for k, v in agg["fn_bug_counts"].most_common(10)] or ["- none"])
        lines.append("")
        lines.append("## Top FP Bug Pairs")
        lines.extend([f"- {k[0]} / {k[1]}: {v}" for k, v in agg["fp_bug_pair_counts"].most_common(10)] or ["- none"])
        for title, key in [("FN primary_signature", "fn_primary"), ("FN mismatch_type", "fn_mismatch"), ("FN op_pair", "fn_op"), ("FN fatal_file", "fn_fatal"), ("FP primary_signature", "fp_primary"), ("FP mismatch_type", "fp_mismatch"), ("FP op_pair", "fp_op"), ("FP fatal_file", "fp_fatal")]:
            lines.append("")
            lines.append(f"## {title}")
            lines.extend([f"- {k}: {v}" for k, v in agg[key].most_common(8)] or ["- none"])
        report_lines[ds] = lines
    return error_rows, report_lines


def _pair_row(ds: str, seed: int, err_type: str, i: int, j: int, case_ids: Sequence[str], gold: Sequence[str], pred: Sequence[str], prob: np.ndarray, features: Sequence[object]) -> dict:
    row = {"dataset": ds, "seed": seed, "type": err_type, "i": i, "j": j, "case_i": case_ids[i], "case_j": case_ids[j], "gold_i": gold[i], "gold_j": gold[j], "pred_i": pred[i], "pred_j": pred[j], "prob": float(prob[i, j])}
    for key in ["primary_signature", "primary_type", "mismatch_type", "op_pair", "fatal_file"]:
        row[f"{key}_i"] = _info(features, i, key)
        row[f"{key}_j"] = _info(features, j, key)
    return row


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze calibrated pairwise blend errors.")
    p.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/calibrated_blend_error_analysis"))
    p.add_argument("--model-root", type=Path, default=Path("/tmp/input_signal_5seed_top/models/llm_dual_struct_det_summary_dim64"))
    p.add_argument("--model-tag", default="llm_dual_struct_det_summary_dim64")
    p.add_argument("--split-root", type=Path, default=Path("/tmp/input_signal_5seed_top/models/llm_dual_struct_det_summary_dim64/splits"))
    p.add_argument("--ensemble-model-dir", type=Path, default=Path("/tmp/pairwise_llm_exp_full/models"))
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--combo", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.85)
    p.add_argument("--rich-temperature", type=float, default=0.75)
    p.add_argument("--ensemble-temperature", type=float, default=1.25)
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--predict-batch-size", type=int, default=100000)
    p.add_argument("--max-error-pairs", type=int, default=2000)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, reports = analyze(args)
    fields = ["dataset", "seed", "type", "i", "j", "case_i", "case_j", "gold_i", "gold_j", "pred_i", "pred_j", "prob", "primary_signature_i", "primary_signature_j", "primary_type_i", "primary_type_j", "mismatch_type_i", "mismatch_type_j", "op_pair_i", "op_pair_j", "fatal_file_i", "fatal_file_j"]
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_ds[row["dataset"]].append(row)
    for ds, ds_rows in by_ds.items():
        _write_csv(args.output_dir / f"{ds}_error_pairs.csv", ds_rows, fields)
    _write_csv(args.output_dir / "error_pairs.csv", rows, fields)
    for ds, lines in reports.items():
        (args.output_dir / f"{ds}_error_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n" + "\n".join(lines[:28]))
    print(f"\nReports: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
