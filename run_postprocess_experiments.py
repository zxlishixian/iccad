#!/usr/bin/env python3
"""Search lightweight postprocess rules for calibrated pairwise blends.

Experimental only. Training/evaluation may read gold.csv, but postprocess itself
uses only predicted labels, pairwise probabilities, and sim/regr-derived features.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import pairwise_llm_features as plf
import pairwise_postprocess as pp
from run_experiments import pairwise_scores, read_gold
from run_input_signal_calibration import _find_ensemble_models, _find_model, _temperature
from run_input_signal_experiments import ENSEMBLE_WEIGHTS, build_val_parts, summarize
from run_half_split_experiments import DEFAULT_DATASETS

PROJECT_ROOT = Path(__file__).resolve().parent


def _write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _candidate_rows(args: argparse.Namespace) -> list[tuple[str, dict]]:
    candidates: list[tuple[str, dict]] = [("none", {})]
    if "merge_close" in args.postprocess:
        for prob in args.merge_prob_thresholds:
            for cons in args.merge_consistency_thresholds:
                for conflict in args.merge_conflict_maxes:
                    params = {
                        "merge_topk": args.merge_topk,
                        "merge_prob_threshold": prob,
                        "merge_consistency_threshold": cons,
                        "merge_conflict_max": conflict,
                    }
                    candidates.append(("merge_close", params))
    if "split_mixed" in args.postprocess:
        for min_bucket in args.split_min_bucket_sizes:
            for min_group in args.split_min_group_sizes:
                for key in args.split_keys:
                    params = {
                        "split_min_bucket_size": min_bucket,
                        "split_min_group_size": min_group,
                        "split_key": key,
                    }
                    candidates.append(("split_mixed", params))
    if "split_then_merge" in args.postprocess:
        for prob in args.merge_prob_thresholds:
            for min_bucket in args.split_min_bucket_sizes:
                params = {
                    "merge_topk": args.merge_topk,
                    "merge_prob_threshold": prob,
                    "merge_consistency_threshold": args.merge_consistency_thresholds[0],
                    "merge_conflict_max": args.merge_conflict_maxes[0],
                    "split_min_bucket_size": min_bucket,
                    "split_min_group_size": args.split_min_group_sizes[0],
                    "split_key": args.split_keys[0],
                }
                candidates.append(("split_then_merge", params))
    return candidates


def _run_name(mode: str, params: dict) -> str:
    if mode == "none":
        return "none"
    payload = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    return f"{mode}_{payload}".replace(".", "p")


def evaluate(args: argparse.Namespace) -> list[dict]:
    datasets = [(d if d.is_absolute() else (PROJECT_ROOT / d).resolve()) for d in args.datasets]
    candidates = _candidate_rows(args)
    rows: list[dict] = []
    for seed in args.seeds:
        model_path = _find_model(args.model_root, args.model_tag, seed, args.combo)
        rich_model = plf.load_model_pkg(model_path)
        feature_mode = str(rich_model.get("feature_mode", ""))
        rich_args = plf._make_llm_args(
            llm_mode="embedding", llm_doc_style="features", llm_cache_dir=args.llm_cache_dir,
            svd_dim=args.svd_dim, llm_dual=feature_mode in plf.DUAL_FEATURE_MODES,
        )
        ensemble_args = plf._make_llm_args(
            llm_mode="embedding", llm_doc_style="features", llm_cache_dir=args.llm_cache_dir,
            svd_dim=args.svd_dim,
        )
        ensemble_pkgs = [plf.load_model_pkg(path) for path in _find_ensemble_models(args.ensemble_model_dir, seed, args.combo)]
        val_parts = build_val_parts(datasets, seed, args.combo, args.split_root / f"seed_{seed}")
        for part in val_parts:
            print(f"[prepare] seed={seed} dataset={part['dataset']}", file=sys.stderr)
            t0 = time.perf_counter()
            rich_features, _ = plf.build_llm_case_features(part["input"], svd_dim=args.svd_dim, llm_args=rich_args)
            p_rich = plf.predict_probability_matrix_sklearn(rich_model, rich_features, batch_size=args.predict_batch_size)
            p_rich = _temperature(p_rich, args.rich_temperature)
            ensemble_features, _ = plf.build_llm_case_features(part["input"], svd_dim=args.svd_dim, llm_args=ensemble_args)
            p_ensemble = plf.predict_probability_matrix_ensemble(
                ensemble_pkgs, list(ENSEMBLE_WEIGHTS), ensemble_features,
                ensemble_mode="prob_average", batch_size=args.predict_batch_size,
            )
            p_ensemble = _temperature(p_ensemble, args.ensemble_temperature)
            prob = float(args.alpha) * p_rich + (1.0 - float(args.alpha)) * p_ensemble
            base_labels = plf.cluster_from_probability(prob.astype(np.float32), part["k"])
            gold = read_gold(part["gold"])
            prep_sec = time.perf_counter() - t0
            for mode, params in candidates:
                labels = pp.apply_postprocess(base_labels, prob, rich_features, mode, params)
                pred = [f"bucket_{label:03d}" for label in labels]
                ba, tpr, tnr = pairwise_scores(gold, pred)
                run_name = _run_name(mode, params)
                pred_path = ""
                if args.save_preds:
                    pred_file = args.output_dir / "preds" / f"seed{seed}_{part['dataset']}_{run_name}.csv"
                    pred_file.parent.mkdir(parents=True, exist_ok=True)
                    with pred_file.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(["bucket"])
                        for item in pred:
                            writer.writerow([item])
                    pred_path = str(pred_file)
                rows.append({
                    "seed": seed,
                    "combo": f"{args.combo:03b}",
                    "dataset": part["dataset"],
                    "method": "calibrated_blend_postprocess",
                    "postprocess": mode,
                    "params": json.dumps(params, sort_keys=True),
                    "run_name": run_name,
                    "BA": ba,
                    "TPR": tpr,
                    "TNR": tnr,
                    "num_cases": part["num_cases"],
                    "k": part["k"],
                    "num_pred_clusters": len(set(labels)),
                    "prep_runtime_sec": prep_sec,
                    "pred_path": pred_path,
                })
                print(
                    f"[eval] seed={seed} dataset={part['dataset']} post={run_name} "
                    f"clusters={len(set(labels))} BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f}",
                    file=sys.stderr,
                )
    return rows


def summarize_post(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["run_name"], row["dataset"])].append(row)
    out = []
    for (run_name, dataset), items in sorted(groups.items()):
        bas = [float(r["BA"]) for r in items]
        tprs = [float(r["TPR"]) for r in items]
        tnrs = [float(r["TNR"]) for r in items]
        clusters = [float(r["num_pred_clusters"]) for r in items]
        first = items[0]
        out.append({
            "run_name": run_name,
            "postprocess": first["postprocess"],
            "params": first["params"],
            "dataset": dataset,
            "mean_BA": statistics.mean(bas),
            "std_BA": statistics.stdev(bas) if len(bas) > 1 else 0.0,
            "mean_TPR": statistics.mean(tprs),
            "mean_TNR": statistics.mean(tnrs),
            "mean_num_pred_clusters": statistics.mean(clusters),
            "num_runs": len(items),
        })
    return out


def print_wide(summary_rows: Sequence[dict]) -> None:
    by_run: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in summary_rows:
        by_run[row["run_name"]][row["dataset"]] = row
    ranked = []
    for run, dsrows in by_run.items():
        if len(dsrows) == 3:
            mean_ba = statistics.mean(float(r["mean_BA"]) for r in dsrows.values())
            ranked.append((mean_ba, run, dsrows))
    print("\n| run_name | postprocess | first_BA | stage2_BA | stage3_BA | mean_BA | first_TPR/TNR | stage2_TPR/TNR | stage3_TPR/TNR |")
    print("|---|---|---:|---:|---:|---:|---|---|---|")
    for mean_ba, run, dsrows in sorted(ranked, reverse=True)[:20]:
        f = dsrows.get("first_batch_dataset", {})
        s2 = dsrows.get("stage2_dataset_working", {})
        s3 = dsrows.get("stage3_dataset_32bugs_640cases", {})
        def pair(r): return f"{float(r.get('mean_TPR',0)):.4f}/{float(r.get('mean_TNR',0)):.4f}"
        print(f"| {run} | {f.get('postprocess','')} | {float(f.get('mean_BA',0)):.6f} | {float(s2.get('mean_BA',0)):.6f} | {float(s3.get('mean_BA',0)):.6f} | {mean_ba:.6f} | {pair(f)} | {pair(s2)} | {pair(s3)} |")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run postprocess experiments for calibrated pairwise blends.")
    p.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/postprocess_search"))
    p.add_argument("--model-root", type=Path, default=Path("/tmp/input_signal_5seed_top/models/llm_dual_struct_det_summary_dim64"))
    p.add_argument("--model-tag", default="llm_dual_struct_det_summary_dim64")
    p.add_argument("--split-root", type=Path, default=Path("/tmp/input_signal_5seed_top/models/llm_dual_struct_det_summary_dim64/splits"))
    p.add_argument("--ensemble-model-dir", type=Path, default=Path("/tmp/pairwise_llm_exp_full/models"))
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--combo", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.85)
    p.add_argument("--rich-temperature", type=float, default=0.75)
    p.add_argument("--ensemble-temperature", type=float, default=1.25)
    p.add_argument("--postprocess", nargs="+", choices=("merge_close", "split_mixed", "split_then_merge"), default=["merge_close", "split_mixed"])
    p.add_argument("--merge-topk", type=int, default=10)
    p.add_argument("--merge-prob-thresholds", nargs="+", type=float, default=[0.70, 0.75, 0.80, 0.85])
    p.add_argument("--merge-consistency-thresholds", nargs="+", type=float, default=[0.50, 0.65])
    p.add_argument("--merge-conflict-maxes", nargs="+", type=float, default=[0.10, 0.20])
    p.add_argument("--split-min-bucket-sizes", nargs="+", type=int, default=[4, 6])
    p.add_argument("--split-min-group-sizes", nargs="+", type=int, default=[2, 3])
    p.add_argument("--split-keys", nargs="+", default=["auto"])
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--predict-batch-size", type=int, default=100000)
    p.add_argument("--save-preds", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = evaluate(args)
    result_fields = ["seed", "combo", "dataset", "method", "postprocess", "params", "run_name", "BA", "TPR", "TNR", "num_cases", "k", "num_pred_clusters", "prep_runtime_sec", "pred_path"]
    _write_csv(args.output_dir / "results.csv", rows, result_fields)
    summary_rows = summarize_post(rows)
    summary_fields = ["run_name", "postprocess", "params", "dataset", "mean_BA", "std_BA", "mean_TPR", "mean_TNR", "mean_num_pred_clusters", "num_runs"]
    _write_csv(args.output_dir / "summary.csv", summary_rows, summary_fields)
    print_wide(summary_rows)
    print(f"\nResults: {args.output_dir / 'results.csv'}")
    print(f"Summary: {args.output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
