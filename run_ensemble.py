#!/usr/bin/env python3
"""Soft voting ensemble of pairwise same-bug models.

Fuses logistic + gbdt + mlp probability matrices via weighted averaging
(prob_average or logit_average), then clusters with AgglomerativeClustering.

Two modes:
  --search   Run seed=0 weight search across predefined configs
  (default)  Evaluate a single config on specified seeds
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
from run_experiments import pairwise_scores, read_gold, read_pred
from run_half_split_experiments import DEFAULT_DATASETS, opposite_part, part_for_bit, stratified_half_split

PROJECT_ROOT = Path(__file__).resolve().parent

# (w_logistic, w_gbdt, w_mlp)
WEIGHT_CONFIGS: list[tuple[float, float, float]] = [
    (1/3, 1/3, 1/3),       # 0: equal
    (0.20, 0.60, 0.20),    # 1: gbdt heavy
    (0.10, 0.80, 0.10),    # 2: gbdt dominant
    (0.30, 0.50, 0.20),    # 3: gbdt+logistic bias
    (0.20, 0.50, 0.30),    # 4: slight mlp
    (0.40, 0.40, 0.20),    # 5: logistic+gbdt
    (0.20, 0.40, 0.40),    # 6: gbdt+mlp
    (0.25, 0.50, 0.25),    # 7: symmetric gbdt center
    (0.15, 0.70, 0.15),    # 8: gbdt primary
    (0.30, 0.35, 0.35),    # 9: near-equal
]

ENSEMBLE_MODES = ["prob_average", "logit_average"]
BASE_MODEL_TYPES = ["logistic", "gbdt", "mlp"]


def _model_ext(model_type: str) -> str:
    return "pt" if model_type == "mlp" else "pkl"


def find_base_models(
    model_dir: Path, seed: int, combo: int
) -> list[Path]:
    paths: list[Path] = []
    for mt in BASE_MODEL_TYPES:
        p = model_dir / f"model_seed{seed}_combo{combo:03b}_{mt}.{_model_ext(mt)}"
        if not p.exists():
            raise FileNotFoundError(f"missing base model: {p}")
        paths.append(p)
    return paths


def build_val_parts(
    datasets: list[Path],
    seed: int,
    combo: int,
    split_root: Path,
) -> list[dict]:
    parts: list[dict] = []
    for idx, ds in enumerate(datasets):
        splits = stratified_half_split(ds, seed, split_root)
        train_part = part_for_bit((combo >> idx) & 1)
        val_part = opposite_part(train_part)
        info = dict(splits[val_part])
        info["dataset"] = ds.name
        parts.append(info)
    return parts


def evaluate_ensemble(
    model_dir: Path,
    output_dir: Path,
    seed: int,
    combo: int,
    weights: tuple[float, float, float],
    ensemble_mode: str,
    llm_doc_style: str,
    llm_cache_dir: Path,
    svd_dim: int,
    predict_batch_size: int,
    split_root: Path | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """Evaluate ensemble on all val parts for one seed+combo. Returns result rows."""
    model_paths = find_base_models(model_dir, seed, combo)
    model_pkgs = [plf.load_model_pkg(p) for p in model_paths]
    weights_list = list(weights)

    datasets = [
        (Path(d) if Path(d).is_absolute() else (PROJECT_ROOT / d).resolve())
        for d in DEFAULT_DATASETS
    ]
    if split_root is None:
        split_root = output_dir / "splits"
    val_parts = build_val_parts(datasets, seed, combo, split_root)

    llm_args = plf._make_llm_args(
        llm_mode="embedding",
        llm_doc_style=llm_doc_style,
        llm_cache_dir=llm_cache_dir,
        svd_dim=svd_dim,
    )

    results: list[dict] = []
    weight_label = f"wL{weights[0]:.2f}_wG{weights[1]:.2f}_wM{weights[2]:.2f}"

    for part in val_parts:
        pred_path = (
            output_dir / "preds" /
            f"ensemble_seed{seed}_combo{combo:03b}_{part['dataset']}_{ensemble_mode}_{weight_label}.csv"
        )
        if dry_run:
            print(f"  [dry-run] would predict {part['dataset']} -> {pred_path}", file=sys.stderr)
            continue

        t0 = time.perf_counter()
        try:
            features, _bundle = plf.build_llm_case_features(
                part["input"], svd_dim=svd_dim, llm_args=llm_args
            )
            prob = plf.predict_probability_matrix_ensemble(
                model_pkgs, weights_list, features,
                ensemble_mode=ensemble_mode, batch_size=predict_batch_size,
            )
            labels = plf.cluster_from_probability(prob, part["k"])
        except Exception as exc:
            print(f"ERROR ensemble {part['dataset']}: {exc}", file=sys.stderr)
            continue
        runtime = time.perf_counter() - t0

        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["bucket"])
            for label in labels:
                writer.writerow([f"bucket_{label:03d}"])

        gold = read_gold(part["gold"])
        pred = read_pred(pred_path)
        ba, tpr, tnr = pairwise_scores(gold, pred)
        results.append({
            "seed": seed,
            "combo": f"{combo:03b}",
            "dataset": part["dataset"],
            "method": f"ensemble_{ensemble_mode}",
            "doc_style": llm_doc_style,
            "model_type": "ensemble",
            "weights": weight_label,
            "ensemble_mode": ensemble_mode,
            "BA": ba,
            "TPR": tpr,
            "TNR": tnr,
            "num_cases": part["num_cases"],
            "k": part["k"],
            "num_pred_clusters": len(set(pred)),
            "runtime_sec": runtime,
            "model_path": ",".join(str(p) for p in model_paths),
            "pred_path": str(pred_path),
        })
        print(
            f"done seed={seed} combo={combo:03b} dataset={part['dataset']} "
            f"ensemble={ensemble_mode} {weight_label} BA={ba:.6f}",
            file=sys.stderr,
        )

    return results


def run_weight_search(
    model_dir: Path,
    output_dir: Path,
    combo: int,
    llm_doc_style: str,
    llm_cache_dir: Path,
    svd_dim: int,
    predict_batch_size: int,
    split_root: Path | None = None,
) -> list[dict]:
    """Run seed=0 weight search: pre-compute prob matrices, then sweep weights."""
    all_results: list[dict] = []
    total = len(WEIGHT_CONFIGS) * len(ENSEMBLE_MODES)

    datasets = [
        (Path(d) if Path(d).is_absolute() else (PROJECT_ROOT / d).resolve())
        for d in DEFAULT_DATASETS
    ]
    if split_root is None:
        split_root = output_dir / "splits"

    llm_args = plf._make_llm_args(
        llm_mode="embedding",
        llm_doc_style=llm_doc_style,
        llm_cache_dir=llm_cache_dir,
        svd_dim=svd_dim,
    )
    model_paths = find_base_models(model_dir, seed=0, combo=combo)
    model_pkgs = [plf.load_model_pkg(p) for p in model_paths]

    val_parts = build_val_parts(datasets, seed=0, combo=combo, split_root=split_root)

    # Pre-compute probability matrices per dataset per model
    # cached_probs[dataset_name] = [P_logistic, P_gbdt, P_mlp]
    cached_probs: dict[str, list[np.ndarray]] = {}
    cached_features: dict[str, list[plf.LLMCaseFeature]] = {}
    cached_ks: dict[str, int] = {}
    cached_golds: dict[str, list[str]] = {}

    for part in val_parts:
        ds = part["dataset"]
        print(f"\n[precompute] building features for {ds}", file=sys.stderr)
        features, _bundle = plf.build_llm_case_features(
            part["input"], svd_dim=svd_dim, llm_args=llm_args
        )
        cached_features[ds] = features
        cached_ks[ds] = part["k"]
        cached_golds[ds] = read_gold(part["gold"])

        probs: list[np.ndarray] = []
        for mi, mt in enumerate(BASE_MODEL_TYPES):
            t0 = time.perf_counter()
            prob = plf.predict_probability_matrix_sklearn(
                model_pkgs[mi], features, batch_size=predict_batch_size
            )
            elapsed = time.perf_counter() - t0
            print(f"  {mt}: prob matrix {prob.shape}, {elapsed:.1f}s", file=sys.stderr)
            probs.append(prob)
        cached_probs[ds] = probs

    # Sweep weight configs
    for idx, weights in enumerate(WEIGHT_CONFIGS):
        for mode in ENSEMBLE_MODES:
            i = idx * len(ENSEMBLE_MODES) + ENSEMBLE_MODES.index(mode)
            wlabel = f"wL{weights[0]:.2f}_wG{weights[1]:.2f}_wM{weights[2]:.2f}"
            print(
                f"\n[{i+1}/{total}] ensemble={mode} weights=({weights[0]:.2f},{weights[1]:.2f},{weights[2]:.2f})",
                file=sys.stderr,
            )

            for part in val_parts:
                ds = part["dataset"]
                probs = cached_probs[ds]  # [P_logistic, P_gbdt, P_mlp]
                features = cached_features[ds]
                n = len(features)

                t0 = time.perf_counter()
                # Fuse cached probability matrices
                weights_list = list(weights)
                if n <= 1:
                    fused = np.eye(n, dtype=np.float32)
                elif mode == "prob_average":
                    fused = np.zeros((n, n), dtype=np.float64)
                    for w, P in zip(weights_list, probs):
                        fused += w * P.astype(np.float64)
                    fused = fused.astype(np.float32)
                else:  # logit_average
                    fused = np.zeros((n, n), dtype=np.float64)
                    eps = 1e-9
                    for w, P in zip(weights_list, probs):
                        Pc = np.clip(P.astype(np.float64), eps, 1.0 - eps)
                        fused += w * np.log(Pc / (1.0 - Pc))
                    fused = (1.0 / (1.0 + np.exp(-fused))).astype(np.float32)

                labels = plf.cluster_from_probability(fused, cached_ks[ds])
                runtime = time.perf_counter() - t0

                pred_path = (
                    output_dir / "preds" /
                    f"ensemble_seed0_combo{combo:03b}_{ds}_{mode}_{wlabel}.csv"
                )
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                with open(pred_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["bucket"])
                    for label in labels:
                        writer.writerow([f"bucket_{label:03d}"])

                gold = cached_golds[ds]
                pred = read_pred(pred_path)
                ba, tpr, tnr = pairwise_scores(gold, pred)
                all_results.append({
                    "seed": 0,
                    "combo": f"{combo:03b}",
                    "dataset": ds,
                    "method": f"ensemble_{mode}",
                    "doc_style": llm_doc_style,
                    "model_type": "ensemble",
                    "weights": wlabel,
                    "ensemble_mode": mode,
                    "BA": ba, "TPR": tpr, "TNR": tnr,
                    "num_cases": part["num_cases"],
                    "k": part["k"],
                    "num_pred_clusters": len(set(pred)),
                    "runtime_sec": runtime,
                    "model_path": ",".join(str(p) for p in model_paths),
                    "pred_path": str(pred_path),
                })
                print(
                    f"  {ds}: BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f}",
                    file=sys.stderr,
                )

    return all_results


def _weights_from_label(label: str) -> tuple[float, float, float]:
    parts = label.replace("wL", "").replace("wG", "").replace("wM", "").split("_")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def summarize(
    results: list[dict],
    output_dir: Path,
    print_delta_baseline: bool = True,
) -> list[dict]:
    # Per-dataset summary
    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in results:
        key = (row["method"], row["dataset"], row.get("weights", ""), row.get("ensemble_mode", ""))
        groups[key].append(row)

    summary_rows: list[dict] = []
    for (method, dataset, weights, mode), items in sorted(groups.items()):
        summary_rows.append({
            "method": method,
            "dataset": dataset,
            "weights": weights,
            "ensemble_mode": mode,
            "mean_BA": statistics.mean(float(r["BA"]) for r in items),
            "std_BA": statistics.stdev(float(r["BA"]) for r in items) if len(items) > 1 else 0.0,
            "mean_TPR": statistics.mean(float(r["TPR"]) for r in items),
            "mean_TNR": statistics.mean(float(r["TNR"]) for r in items),
            "num_runs": len(items),
        })

    summary_path = output_dir / "summary.csv"
    summary_header = ["method", "dataset", "weights", "ensemble_mode",
                      "mean_BA", "std_BA", "mean_TPR", "mean_TNR", "num_runs"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_header)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Aggregate across datasets per config
    config_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in summary_rows:
        key = (row["weights"], row["ensemble_mode"])
        config_groups[key].append(float(row["mean_BA"]))

    print("\n=== Ensemble Weight Search Results (seed=0) ===")
    print(f"{'rank':<5} {'mode':<16} {'w_logistic':>8} {'w_gbdt':>8} {'w_mlp':>8} {'mean_BA':>10} {'first_batch':>12} {'stage2':>12} {'stage3':>12}")
    print("-" * 95)

    ranked: list[tuple[float, str, str, tuple[float, float, float], dict[str, float]]] = []
    for (weights_label, mode), bas in config_groups.items():
        mean_ba = statistics.mean(bas)
        w = _weights_from_label(weights_label)
        ds_scores: dict[str, float] = {}
        for row in summary_rows:
            if row["weights"] == weights_label and row["ensemble_mode"] == mode:
                ds_scores[row["dataset"]] = float(row["mean_BA"])
        ranked.append((mean_ba, weights_label, mode, w, ds_scores))
    ranked.sort(key=lambda x: x[0], reverse=True)

    for rank, (mean_ba, wlabel, mode, w, ds_scores) in enumerate(ranked, 1):
        fb = ds_scores.get("first_batch_dataset", 0)
        s2 = ds_scores.get("stage2_dataset_working", 0)
        s3 = ds_scores.get("stage3_dataset_32bugs_640cases", 0)
        print(
            f"{rank:<5} {mode:<16} {w[0]:8.2f} {w[1]:8.2f} {w[2]:8.2f} "
            f"{mean_ba:10.6f} {fb:12.6f} {s2:12.6f} {s3:12.6f}"
        )

    print(f"\nFull results: {output_dir / 'results.csv'}")
    print(f"Summary:      {summary_path}")
    return summary_rows


def main() -> int:
    p = argparse.ArgumentParser(description="Soft voting ensemble for pairwise same-bug models.")
    p.add_argument("--model-dir", type=Path, default=Path("/tmp/pairwise_llm_exp_full/models"))
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/pairwise_llm_exp_ensemble"))
    p.add_argument("--split-root", type=Path, default=Path("/tmp/pairwise_llm_exp_full/splits"))
    p.add_argument("--combo", type=int, default=0)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--search", action="store_true",
                   help="Run seed=0 weight search across all predefined configs")
    p.add_argument("--weights", nargs=3, type=float, default=[1/3, 1/3, 1/3],
                   help="Weights for (logistic, gbdt, mlp), sum will be normalized to 1")
    p.add_argument("--ensemble-mode", choices=("prob_average", "logit_average"), default="prob_average")
    p.add_argument("--llm-doc-style", choices=("features", "summary"), default="features")
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--predict-batch-size", type=int, default=100000)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-save-results", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    if args.search:
        print("=== Ensemble Weight Search (seed=0) ===", file=sys.stderr)
        print(f"Configs: {len(WEIGHT_CONFIGS)} weights x {len(ENSEMBLE_MODES)} modes = "
              f"{len(WEIGHT_CONFIGS) * len(ENSEMBLE_MODES)} evals", file=sys.stderr)
        all_results = run_weight_search(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            combo=args.combo,
            llm_doc_style=args.llm_doc_style,
            llm_cache_dir=args.llm_cache_dir,
            svd_dim=args.svd_dim,
            predict_batch_size=args.predict_batch_size,
            split_root=args.split_root,
        )
    else:
        # Normalize weights
        w = np.asarray(args.weights, dtype=np.float64)
        w = w / w.sum()
        weights_tuple = (float(w[0]), float(w[1]), float(w[2]))
        print(
            f"=== Ensemble {args.ensemble_mode} weights=(logistic={w[0]:.3f}, gbdt={w[1]:.3f}, mlp={w[2]:.3f}) ===",
            file=sys.stderr,
        )
        for seed in args.seeds:
            rows = evaluate_ensemble(
                model_dir=args.model_dir,
                output_dir=args.output_dir,
                seed=seed,
                combo=args.combo,
                weights=weights_tuple,
                ensemble_mode=args.ensemble_mode,
                llm_doc_style=args.llm_doc_style,
                llm_cache_dir=args.llm_cache_dir,
                svd_dim=args.svd_dim,
                predict_batch_size=args.predict_batch_size,
                split_root=args.split_root,
                dry_run=args.dry_run,
            )
            all_results.extend(rows)

    # Write results
    if all_results and not args.skip_save_results:
        result_header = [
            "seed", "combo", "dataset", "method", "doc_style", "model_type",
            "weights", "ensemble_mode",
            "BA", "TPR", "TNR", "num_cases", "k", "num_pred_clusters",
            "runtime_sec", "model_path", "pred_path",
        ]
        results_path = args.output_dir / "results.csv"
        with open(results_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=result_header)
            writer.writeheader()
            writer.writerows(all_results)

        summarize(all_results, args.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
