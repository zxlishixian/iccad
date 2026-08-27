#!/usr/bin/env python3
"""Run selective trace-assisted pair refinement experiments.

Experimental only: training reads gold.csv on half splits; prediction/refinement
uses P_base plus trace.log-derived features and does not read gold/meta.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import pairwise_llm_features as plf
import trace_features as tf
import trace_refiner as tr
from run_experiments import pairwise_scores, read_gold
from run_half_split_experiments import DEFAULT_DATASETS, opposite_part, part_for_bit, stratified_half_split
from run_input_signal_calibration import _temperature
from run_input_signal_experiments import ENSEMBLE_MODEL_TYPES, ENSEMBLE_WEIGHTS, _model_ext, print_wide

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class PartData:
    dataset: str
    input: Path
    gold_path: Path
    gold: list[str]
    k: int
    num_cases: int
    base_prob: np.ndarray
    base_labels: list[int]
    case_features: list[plf.LLMCaseFeature]


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_band(text: str) -> tuple[float, float]:
    left, right = str(text).split(",", 1)
    lower = float(left)
    upper = float(right)
    if lower > upper:
        raise ValueError(f"invalid band: {text}")
    return lower, upper


def _find_rich_model(model_root: Path, model_tag: str, seed: int, combo: int) -> Path:
    path = model_root / f"model_seed{seed}_combo{combo:03b}_{model_tag}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing rich model: {path}")
    return path


def _find_ensemble_models(model_dir: Path, seed: int, combo: int) -> list[Path]:
    paths = []
    for model_type in ENSEMBLE_MODEL_TYPES:
        path = model_dir / f"model_seed{seed}_combo{combo:03b}_{model_type}.{_model_ext(model_type)}"
        if not path.exists():
            raise FileNotFoundError(f"missing ensemble model: {path}")
        paths.append(path)
    return paths


def _split_parts(datasets: Sequence[Path], seed: int, combo: int, split_root: Path) -> tuple[list[dict], list[dict]]:
    train_parts = []
    val_parts = []
    for idx, ds in enumerate(datasets):
        splits = stratified_half_split(ds, seed, split_root)
        train_part = part_for_bit((combo >> idx) & 1)
        val_part = opposite_part(train_part)
        trn = dict(splits[train_part])
        val = dict(splits[val_part])
        trn["dataset"] = ds.name
        val["dataset"] = ds.name
        train_parts.append(trn)
        val_parts.append(val)
    return train_parts, val_parts


def _base_probability_for_part(
    part: dict,
    rich_model: dict,
    ensemble_pkgs: list[dict],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[plf.LLMCaseFeature], float]:
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
    t0 = time.perf_counter()
    rich_features, _ = plf.build_llm_case_features(part["input"], svd_dim=args.svd_dim, llm_args=rich_args)
    p_rich = plf.predict_probability_matrix_sklearn(rich_model, rich_features, batch_size=args.predict_batch_size)
    ensemble_features, _ = plf.build_llm_case_features(part["input"], svd_dim=args.svd_dim, llm_args=ensemble_args)
    p_ensemble = plf.predict_probability_matrix_ensemble(
        ensemble_pkgs,
        list(ENSEMBLE_WEIGHTS),
        ensemble_features,
        ensemble_mode="prob_average",
        batch_size=args.predict_batch_size,
    )
    p_base = float(args.alpha) * _temperature(p_rich, args.rich_temperature) + (1.0 - float(args.alpha)) * _temperature(p_ensemble, args.ensemble_temperature)
    np.fill_diagonal(p_base, 1.0)
    return p_base.astype(np.float32), rich_features, time.perf_counter() - t0


def _prepare_parts(
    parts: Sequence[dict],
    rich_model: dict,
    ensemble_pkgs: list[dict],
    args: argparse.Namespace,
) -> tuple[list[PartData], float]:
    out: list[PartData] = []
    runtime = 0.0
    for part in parts:
        print(f"[base] seed={args.seed_current} dataset={part['dataset']} input={part['input']}", file=sys.stderr)
        prob, case_features, sec = _base_probability_for_part(part, rich_model, ensemble_pkgs, args)
        runtime += sec
        labels = plf.cluster_from_probability(prob, part["k"])
        gold = read_gold(part["gold"])
        out.append(PartData(
            dataset=part["dataset"],
            input=Path(part["input"]),
            gold_path=Path(part["gold"]),
            gold=gold,
            k=int(part["k"]),
            num_cases=int(part["num_cases"]),
            base_prob=prob,
            base_labels=labels,
            case_features=case_features,
        ))
    return out, runtime


def _all_pairs(n: int):
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j


def _pair_error_delta(gold: Sequence[str], base_labels: Sequence[int], final_labels: Sequence[int]) -> dict:
    fixed_fn = new_fn = fixed_fp = new_fp = 0
    for i, j in _all_pairs(len(gold)):
        same = gold[i] == gold[j]
        b_same = base_labels[i] == base_labels[j]
        f_same = final_labels[i] == final_labels[j]
        if same and not b_same and f_same:
            fixed_fn += 1
        elif same and b_same and not f_same:
            new_fn += 1
        elif (not same) and b_same and not f_same:
            fixed_fp += 1
        elif (not same) and (not b_same) and f_same:
            new_fp += 1
    return {"fixed_fn_pairs": fixed_fn, "new_fn_pairs": new_fn, "fixed_fp_pairs": fixed_fp, "new_fp_pairs": new_fp}


def _build_training_matrix(
    train_parts: Sequence[PartData],
    trace_by_input: dict[tuple[str, int], list[tf.TraceCaseFeature]],
    tail_lines: int,
    lower: float,
    upper: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict]:
    Xs = []
    ys = []
    total_pairs = 0
    trace_missing = 0
    rng = np.random.default_rng(int(args.random_state))
    for part in train_parts:
        pairs = tr.uncertain_pairs_from_probability(part.base_prob, lower, upper)
        if not pairs:
            continue
        trace_feats = trace_by_input[(str(part.input), tail_lines)]
        usable = [(i, j) for i, j in pairs if not (trace_feats[i].missing or trace_feats[j].missing)]
        trace_missing += len(pairs) - len(usable)
        total_pairs += len(pairs)
        if not usable:
            continue
        X = tr.build_refiner_feature_matrix(part.base_prob, trace_feats, usable, case_features=part.case_features, include_summary=True)
        y = tr.pair_labels_from_gold(part.gold, usable)
        Xs.append(X)
        ys.append(y)
    if not Xs:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.float32), {"train_uncertain_pairs": total_pairs, "train_trace_missing_pairs": trace_missing}
    X_all = np.vstack(Xs).astype(np.float32)
    y_all = np.concatenate(ys).astype(np.float32)
    if args.max_train_pairs > 0 and len(y_all) > args.max_train_pairs:
        idx = rng.choice(np.arange(len(y_all)), size=int(args.max_train_pairs), replace=False)
        X_all = X_all[idx]
        y_all = y_all[idx]
    stats = {
        "train_uncertain_pairs": total_pairs,
        "train_trace_missing_pairs": trace_missing,
        "train_pairs_used": len(y_all),
        "train_positive_rate": float(y_all.mean()) if len(y_all) else 0.0,
    }
    return X_all, y_all, stats


def _evaluate_config(
    train_parts: Sequence[PartData],
    val_parts: Sequence[PartData],
    trace_by_input: dict[tuple[str, int], list[tf.TraceCaseFeature]],
    seed: int,
    tail_lines: int,
    lower: float,
    upper: float,
    refiner_model: str,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    X, y, train_stats = _build_training_matrix(train_parts, trace_by_input, tail_lines, lower, upper, args)
    refiner = tr.train_trace_refiner(X, y, model_type=refiner_model, random_state=int(args.random_state) + seed)
    result_rows = []
    error_rows = []
    for part in val_parts:
        trace_feats = trace_by_input[(str(part.input), tail_lines)]
        t0 = time.perf_counter()
        final_prob, stats = tr.refine_probability_matrix(
            part.base_prob,
            trace_feats,
            refiner,
            lower,
            upper,
            case_features=part.case_features,
            skip_missing_trace=True,
        )
        refine_sec = time.perf_counter() - t0
        final_labels = plf.cluster_from_probability(final_prob, part.k)
        pred = [f"bucket_{label:03d}" for label in final_labels]
        ba, tpr, tnr = pairwise_scores(part.gold, pred)
        base_pred = [f"bucket_{label:03d}" for label in part.base_labels]
        base_ba, base_tpr, base_tnr = pairwise_scores(part.gold, base_pred)
        err = _pair_error_delta(part.gold, part.base_labels, final_labels)
        result_rows.append({
            "seed": seed,
            "combo": f"{args.combo:03b}",
            "dataset": part.dataset,
            "method": "trace_refine",
            "tail_lines": tail_lines,
            "band_lower": lower,
            "band_upper": upper,
            "refiner_model": refiner_model,
            "BA": ba,
            "TPR": tpr,
            "TNR": tnr,
            "base_BA": base_ba,
            "base_TPR": base_tpr,
            "base_TNR": base_tnr,
            "delta_BA": ba - base_ba,
            "num_cases": part.num_cases,
            "k": part.k,
            "num_pred_clusters": len(set(pred)),
            "uncertain_pairs": stats.uncertain_pairs,
            "refined_pairs": stats.refined_pairs,
            "trace_missing_pairs": stats.trace_missing_pairs,
            "runtime_sec": refine_sec,
            "trace_runtime_sec": 0.0,
            **train_stats,
        })
        error_rows.append({
            "seed": seed,
            "combo": f"{args.combo:03b}",
            "dataset": part.dataset,
            "method": "trace_refine",
            "tail_lines": tail_lines,
            "band_lower": lower,
            "band_upper": upper,
            "refiner_model": refiner_model,
            "base_BA": base_ba,
            "BA": ba,
            "delta_BA": ba - base_ba,
            **err,
        })
        print(
            f"[eval] seed={seed} dataset={part.dataset} tail={tail_lines} band={lower:.2f},{upper:.2f} "
            f"BA={ba:.6f} base={base_ba:.6f} refined={stats.refined_pairs}/{stats.uncertain_pairs}",
            file=sys.stderr,
        )
    return result_rows, error_rows


def _summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["method"], row["tail_lines"], row["band_lower"], row["band_upper"], row["refiner_model"], row["dataset"])
        groups[key].append(row)
    out = []
    for key, items in sorted(groups.items()):
        method, tail, lower, upper, model, dataset = key
        bas = [float(r["BA"]) for r in items]
        tprs = [float(r["TPR"]) for r in items]
        tnrs = [float(r["TNR"]) for r in items]
        deltas = [float(r["delta_BA"]) for r in items]
        out.append({
            "method": method,
            "tail_lines": tail,
            "band_lower": lower,
            "band_upper": upper,
            "refiner_model": model,
            "dataset": dataset,
            "mean_BA": statistics.mean(bas),
            "std_BA": statistics.stdev(bas) if len(bas) > 1 else 0.0,
            "mean_TPR": statistics.mean(tprs),
            "mean_TNR": statistics.mean(tnrs),
            "mean_delta_BA": statistics.mean(deltas),
            "num_runs": len(items),
            "mean_uncertain_pairs": statistics.mean(float(r["uncertain_pairs"]) for r in items),
            "mean_refined_pairs": statistics.mean(float(r["refined_pairs"]) for r in items),
            "mean_trace_missing_pairs": statistics.mean(float(r["trace_missing_pairs"]) for r in items),
        })
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Selective trace-assisted pair refinement experiments.")
    p.add_argument("--python", default="/home/lishixian/miniforge3/envs/collab-overcooked/bin/python", help="compatibility no-op; this runner executes in-process")
    p.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/trace_refine_exp"))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--combo", type=int, default=0)
    p.add_argument("--tail-lines", nargs="+", type=int, default=[500])
    p.add_argument("--uncertain-bands", nargs="+", default=["0.35,0.65"])
    p.add_argument("--refiner-model", choices=("gbdt", "logistic"), default="gbdt")
    p.add_argument("--max-train-pairs", type=int, default=300000)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--model-root", type=Path, default=Path("/tmp/input_signal_5seed_top/models/llm_dual_struct_det_summary_dim64"))
    p.add_argument("--model-tag", default="llm_dual_struct_det_summary_dim64")
    p.add_argument("--split-root", type=Path, default=Path("/tmp/input_signal_5seed_top/models/llm_dual_struct_det_summary_dim64/splits"))
    p.add_argument("--ensemble-model-dir", type=Path, default=Path("/tmp/pairwise_llm_exp_full/models"))
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--predict-batch-size", type=int, default=100000)
    p.add_argument("--alpha", type=float, default=0.85)
    p.add_argument("--rich-temperature", type=float, default=0.75)
    p.add_argument("--ensemble-temperature", type=float, default=1.25)
    p.add_argument("--keep-temp", action="store_true", help="compatibility flag; temporary split/model artifacts are preserved by default")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [(d if d.is_absolute() else (PROJECT_ROOT / d).resolve()) for d in args.datasets]
    bands = [_parse_band(text) for text in args.uncertain_bands]
    result_rows = []
    error_rows = []
    trace_cache: dict[tuple[str, int], list[tf.TraceCaseFeature]] = {}

    for seed in args.seeds:
        args.seed_current = seed
        rich_model = plf.load_model_pkg(_find_rich_model(args.model_root, args.model_tag, seed, args.combo))
        ensemble_pkgs = [plf.load_model_pkg(path) for path in _find_ensemble_models(args.ensemble_model_dir, seed, args.combo)]
        split_root = args.split_root / f"seed_{seed}"
        train_info, val_info = _split_parts(datasets, seed, args.combo, split_root)
        train_parts, base_train_sec = _prepare_parts(train_info, rich_model, ensemble_pkgs, args)
        val_parts, base_val_sec = _prepare_parts(val_info, rich_model, ensemble_pkgs, args)

        for tail in args.tail_lines:
            trace_sec = 0.0
            for part in list(train_parts) + list(val_parts):
                key = (str(part.input), int(tail))
                if key not in trace_cache:
                    t0 = time.perf_counter()
                    trace_cache[key] = tf.build_trace_case_features(part.input, tail_lines=int(tail))
                    trace_sec += time.perf_counter() - t0
            for lower, upper in bands:
                rows, errs = _evaluate_config(
                    train_parts, val_parts, trace_cache, seed, int(tail), lower, upper, args.refiner_model, args
                )
                for row in rows:
                    row["base_runtime_sec"] = base_train_sec + base_val_sec
                    row["trace_runtime_sec"] = trace_sec
                result_rows.extend(rows)
                error_rows.extend(errs)

    result_header = [
        "seed", "combo", "dataset", "method", "tail_lines", "band_lower", "band_upper", "refiner_model",
        "BA", "TPR", "TNR", "base_BA", "base_TPR", "base_TNR", "delta_BA", "num_cases", "k",
        "num_pred_clusters", "uncertain_pairs", "refined_pairs", "trace_missing_pairs", "runtime_sec",
        "trace_runtime_sec", "base_runtime_sec", "train_uncertain_pairs", "train_trace_missing_pairs",
        "train_pairs_used", "train_positive_rate",
    ]
    _write_csv(args.output_dir / "results.csv", result_rows, result_header)
    summary_rows = _summarize(result_rows)
    summary_header = [
        "method", "tail_lines", "band_lower", "band_upper", "refiner_model", "dataset", "mean_BA", "std_BA",
        "mean_TPR", "mean_TNR", "mean_delta_BA", "num_runs", "mean_uncertain_pairs", "mean_refined_pairs",
        "mean_trace_missing_pairs",
    ]
    _write_csv(args.output_dir / "summary.csv", summary_rows, summary_header)
    error_header = [
        "seed", "combo", "dataset", "method", "tail_lines", "band_lower", "band_upper", "refiner_model",
        "base_BA", "BA", "delta_BA", "fixed_fn_pairs", "new_fn_pairs", "fixed_fp_pairs", "new_fp_pairs",
    ]
    _write_csv(args.output_dir / "error_analysis.csv", error_rows, error_header)
    print_wide([
        {
            "run_name": f"trace_t{r['tail_lines']}_b{float(r['band_lower']):.2f}-{float(r['band_upper']):.2f}_{r['refiner_model']}",
            "method": r["method"],
            "feature_mode": "selective_trace",
            "reduce_dim": "",
            "blend_alpha": "",
            "dataset": r["dataset"],
            "mean_BA": r["mean_BA"],
            "mean_TPR": r["mean_TPR"],
            "mean_TNR": r["mean_TNR"],
        }
        for r in summary_rows
    ])
    print(f"\nResults: {args.output_dir / 'results.csv'}")
    print(f"Summary: {args.output_dir / 'summary.csv'}")
    print(f"Error analysis: {args.output_dir / 'error_analysis.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
