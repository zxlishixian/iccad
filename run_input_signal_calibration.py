#!/usr/bin/env python3
"""Calibrate blends between a trained rich-input MLP and the pairwise ensemble.

Experimental only: reuses trained half-split models and gold labels for held-out
validation. It does not affect the official regr_fail_bucketing.py default path.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import pairwise_llm_features as plf
from run_experiments import pairwise_scores, read_gold
from run_half_split_experiments import DEFAULT_DATASETS
from run_input_signal_experiments import (
    ENSEMBLE_MODEL_TYPES,
    ENSEMBLE_WEIGHTS,
    _model_ext,
    build_val_parts,
    print_wide,
    summarize,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _find_model(model_root: Path, model_tag: str, seed: int, combo: int) -> Path:
    path = model_root / f"model_seed{seed}_combo{combo:03b}_{model_tag}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing rich model: {path}")
    return path


def _find_ensemble_models(model_dir: Path, seed: int, combo: int) -> list[Path]:
    paths = []
    for model_type in ENSEMBLE_MODEL_TYPES:
        path = model_dir / f"model_seed{seed}_combo{combo:03b}_{model_type}.{_model_ext(model_type)}"
        if not path.exists():
            raise FileNotFoundError(f"missing ensemble base model: {path}")
        paths.append(path)
    return paths


def _temperature(prob: np.ndarray, temp: float) -> np.ndarray:
    if abs(float(temp) - 1.0) < 1e-12:
        return prob
    clipped = np.clip(prob.astype(np.float64), 1e-5, 1.0 - 1e-5)
    logits = np.log(clipped / (1.0 - clipped)) / float(temp)
    out = 1.0 / (1.0 + np.exp(-logits))
    np.fill_diagonal(out, 1.0)
    return out.astype(np.float32)


def _mean_or_zero(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def dataset_policy_summary(rows: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    by_key_dataset: dict[tuple[str, str], list[dict]] = defaultdict(list)
    datasets = sorted({str(row["dataset"]) for row in rows})
    for row in rows:
        key = str(row["run_name"])
        by_key_dataset[(key, str(row["dataset"]))].append(row)

    dataset_best = []
    choices_by_dataset: dict[str, list[dict]] = defaultdict(list)
    for dataset in datasets:
        candidates = []
        for (run_name, ds), items in by_key_dataset.items():
            if ds != dataset:
                continue
            mean_ba = _mean_or_zero([float(item["BA"]) for item in items])
            mean_tpr = _mean_or_zero([float(item["TPR"]) for item in items])
            mean_tnr = _mean_or_zero([float(item["TNR"]) for item in items])
            first = items[0]
            candidates.append({
                "dataset": dataset,
                "run_name": run_name,
                "blend_alpha": first["blend_alpha"],
                "rich_temperature": first["rich_temperature"],
                "ensemble_temperature": first["ensemble_temperature"],
                "mean_BA": mean_ba,
                "mean_TPR": mean_tpr,
                "mean_TNR": mean_tnr,
                "num_runs": len(items),
            })
        candidates.sort(key=lambda row: row["mean_BA"], reverse=True)
        if candidates:
            dataset_best.append(candidates[0])
            choices_by_dataset[dataset] = candidates[:8]

    policy_rows = []
    if len(datasets) == 3 and all(choices_by_dataset[d] for d in datasets):
        for combo in itertools.product(*(choices_by_dataset[d] for d in datasets)):
            mean_ba = statistics.mean([float(item["mean_BA"]) for item in combo])
            policy = {"policy_name": "dataset_alpha_policy", "mean_BA": mean_ba}
            for item in combo:
                name = item["dataset"].replace("_dataset", "").replace("_working", "")
                policy[f"{name}_run"] = item["run_name"]
                policy[f"{name}_BA"] = item["mean_BA"]
                policy[f"{name}_TPR"] = item["mean_TPR"]
                policy[f"{name}_TNR"] = item["mean_TNR"]
            policy_rows.append(policy)
        policy_rows.sort(key=lambda row: row["mean_BA"], reverse=True)
    return dataset_best, policy_rows


def evaluate_grid(args: argparse.Namespace) -> list[dict]:
    datasets = [(d if d.is_absolute() else (PROJECT_ROOT / d).resolve()) for d in args.datasets]
    rows: list[dict] = []
    for seed in args.seeds:
        model_path = _find_model(args.model_root, args.model_tag, seed, args.combo)
        rich_model = plf.load_model_pkg(model_path)
        feature_mode = str(rich_model.get("feature_mode", ""))
        rich_args = plf._make_llm_args(
            llm_mode="embedding",
            llm_doc_style="features",
            llm_cache_dir=args.llm_cache_dir,
            svd_dim=args.svd_dim,
            llm_dual=feature_mode in plf.DUAL_FEATURE_MODES,
        )
        ensemble_args = plf._make_llm_args(
            llm_mode="embedding",
            llm_doc_style="features",
            llm_cache_dir=args.llm_cache_dir,
            svd_dim=args.svd_dim,
        )
        ensemble_pkgs = [plf.load_model_pkg(path) for path in _find_ensemble_models(args.ensemble_model_dir, seed, args.combo)]
        split_root = args.split_root / f"seed_{seed}"
        val_parts = build_val_parts(datasets, seed, args.combo, split_root)
        for part in val_parts:
            print(f"[prepare] seed={seed} dataset={part['dataset']}", file=sys.stderr)
            t0 = time.perf_counter()
            rich_features, _ = plf.build_llm_case_features(part["input"], svd_dim=args.svd_dim, llm_args=rich_args)
            p_rich_base = plf.predict_probability_matrix_sklearn(rich_model, rich_features, batch_size=args.predict_batch_size)
            ensemble_features, _ = plf.build_llm_case_features(part["input"], svd_dim=args.svd_dim, llm_args=ensemble_args)
            p_ensemble_base = plf.predict_probability_matrix_ensemble(
                ensemble_pkgs,
                list(ENSEMBLE_WEIGHTS),
                ensemble_features,
                ensemble_mode="prob_average",
                batch_size=args.predict_batch_size,
            )
            prep_sec = time.perf_counter() - t0
            gold = read_gold(part["gold"])
            for rich_temp in args.rich_temperatures:
                p_rich = _temperature(p_rich_base, rich_temp)
                for ensemble_temp in args.ensemble_temperatures:
                    p_ensemble = _temperature(p_ensemble_base, ensemble_temp)
                    for alpha in args.alphas:
                        prob = float(alpha) * p_rich + (1.0 - float(alpha)) * p_ensemble
                        labels = plf.cluster_from_probability(prob.astype(np.float32), part["k"])
                        pred = [f"bucket_{label:03d}" for label in labels]
                        ba, tpr, tnr = pairwise_scores(gold, pred)
                        run_name = f"{args.model_tag}_blend_a{alpha:.2f}_rt{rich_temp:.2f}_et{ensemble_temp:.2f}"
                        rows.append({
                            "run_name": run_name,
                            "seed": seed,
                            "combo": f"{args.combo:03b}",
                            "method": "input_signal_calibrated_blend",
                            "feature_mode": feature_mode,
                            "arch": rich_model.get("mlp_arch", ""),
                            "loss": rich_model.get("loss", ""),
                            "llm_reduce_dim": rich_model.get("llm_reduce_dim", ""),
                            "blend_alpha": f"{alpha:.2f}",
                            "rich_temperature": f"{rich_temp:.2f}",
                            "ensemble_temperature": f"{ensemble_temp:.2f}",
                            "dataset": part["dataset"],
                            "BA": ba,
                            "TPR": tpr,
                            "TNR": tnr,
                            "num_cases": part["num_cases"],
                            "k": part["k"],
                            "model_path": str(model_path),
                            "prep_runtime_sec": prep_sec,
                        })
                        print(
                            f"[eval] seed={seed} dataset={part['dataset']} alpha={alpha:.2f} "
                            f"rt={rich_temp:.2f} et={ensemble_temp:.2f} BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f}",
                            file=sys.stderr,
                        )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate trained dual-input MLP + pairwise ensemble blends.")
    p.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/input_signal_calibration"))
    p.add_argument("--model-root", type=Path, default=Path("/tmp/input_signal_5seed_top/models/llm_dual_struct_det_summary_dim64"))
    p.add_argument("--model-tag", default="llm_dual_struct_det_summary_dim64")
    p.add_argument("--split-root", type=Path, default=Path("/tmp/input_signal_5seed_top/models/llm_dual_struct_det_summary_dim64/splits"))
    p.add_argument("--ensemble-model-dir", type=Path, default=Path("/tmp/pairwise_llm_exp_full/models"))
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--combo", type=int, default=0)
    p.add_argument("--alphas", nargs="+", type=float, default=[0.88])
    p.add_argument("--rich-temperatures", nargs="+", type=float, default=[1.15])
    p.add_argument("--ensemble-temperatures", nargs="+", type=float, default=[1.0])
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--predict-batch-size", type=int, default=100000)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = evaluate_grid(args)
    result_header = [
        "run_name", "seed", "combo", "method", "feature_mode", "arch", "loss",
        "llm_reduce_dim", "blend_alpha", "rich_temperature", "ensemble_temperature",
        "dataset", "BA", "TPR", "TNR", "num_cases", "k", "model_path", "prep_runtime_sec",
    ]
    _write_csv(args.output_dir / "results.csv", rows, result_header)
    summary_rows = summarize(rows)
    summary_header = [
        "run_name", "method", "feature_mode", "reduce_dim", "blend_alpha", "dataset",
        "mean_BA", "std_BA", "mean_TPR", "mean_TNR", "num_runs",
    ]
    _write_csv(args.output_dir / "summary.csv", summary_rows, summary_header)
    dataset_best, policy_rows = dataset_policy_summary(rows)
    _write_csv(
        args.output_dir / "dataset_best.csv",
        dataset_best,
        ["dataset", "run_name", "blend_alpha", "rich_temperature", "ensemble_temperature", "mean_BA", "mean_TPR", "mean_TNR", "num_runs"],
    )
    if policy_rows:
        policy_fields = sorted({key for row in policy_rows[:64] for key in row})
        _write_csv(args.output_dir / "policy_summary.csv", policy_rows[:64], policy_fields)
    print_wide(summary_rows)
    print(f"\nResults:      {args.output_dir / 'results.csv'}")
    print(f"Summary:      {args.output_dir / 'summary.csv'}")
    print(f"Dataset best: {args.output_dir / 'dataset_best.csv'}")
    if policy_rows:
        print(f"Policies:     {args.output_dir / 'policy_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
