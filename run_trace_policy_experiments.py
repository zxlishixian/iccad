#!/usr/bin/env python3
"""Run conservative trace veto/boost policy experiments."""

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

import pairwise_llm_features as plf
import trace_features as tf
import trace_policy as tp
from run_experiments import pairwise_scores
from run_half_split_experiments import DEFAULT_DATASETS
from run_input_signal_experiments import print_wide
from run_trace_refinement_experiments import (
    PROJECT_ROOT,
    PartData,
    _find_ensemble_models,
    _find_rich_model,
    _pair_error_delta,
    _prepare_parts,
    _split_parts,
    _write_csv,
)


def _param_string(params: tp.TracePolicyParams) -> str:
    if params.trace_policy == "none":
        return "none"
    parts = [f"policy={params.trace_policy}"]
    if params.trace_policy in {"veto", "veto_boost"}:
        parts.extend([
            f"vbase={params.veto_base_min:.2f}",
            f"vtrace={params.veto_trace_max:.2f}",
            f"vcap={params.veto_cap:.2f}",
        ])
    if params.trace_policy in {"boost", "veto_boost"}:
        parts.extend([
            f"blow={params.boost_base_low:.2f}",
            f"bhigh={params.boost_base_high:.2f}",
            f"btrace={params.boost_trace_min:.2f}",
            f"bfloor={params.boost_floor:.2f}",
        ])
    return ";".join(parts)


def _parse_range(text: str) -> tuple[float, float]:
    left, right = str(text).split(",", 1)
    return float(left), float(right)


def _policy_grid(args: argparse.Namespace) -> list[tp.TracePolicyParams]:
    out: list[tp.TracePolicyParams] = []
    for policy in args.trace_policy:
        if policy == "none":
            out.append(tp.TracePolicyParams(trace_policy="none"))
        elif policy == "veto":
            for base, trace, cap in itertools.product(args.veto_base_min, args.veto_trace_max, args.veto_cap):
                out.append(tp.TracePolicyParams(trace_policy="veto", veto_base_min=base, veto_trace_max=trace, veto_cap=cap))
        elif policy == "boost":
            for (low, high), trace, floor in itertools.product(args.boost_base_ranges, args.boost_trace_min, args.boost_floor):
                out.append(tp.TracePolicyParams(trace_policy="boost", boost_base_low=low, boost_base_high=high, boost_trace_min=trace, boost_floor=floor))
        elif policy == "veto_boost":
            for base, trace_max, cap, (low, high), trace_min, floor in itertools.product(
                args.veto_base_min, args.veto_trace_max, args.veto_cap,
                args.boost_base_ranges, args.boost_trace_min, args.boost_floor,
            ):
                out.append(tp.TracePolicyParams(
                    trace_policy="veto_boost",
                    veto_base_min=base,
                    veto_trace_max=trace_max,
                    veto_cap=cap,
                    boost_base_low=low,
                    boost_base_high=high,
                    boost_trace_min=trace_min,
                    boost_floor=floor,
                ))
        else:
            raise ValueError(f"unknown trace policy: {policy}")
    return out


def _evaluate_policy(
    part: PartData,
    trace_feats: list[tf.TraceCaseFeature],
    params: tp.TracePolicyParams,
) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    final_prob, stats = tp.apply_trace_policy(part.base_prob, trace_feats, part.case_features, params)
    runtime = time.perf_counter() - t0
    labels = plf.cluster_from_probability(final_prob, part.k)
    pred = [f"bucket_{label:03d}" for label in labels]
    ba, tpr, tnr = pairwise_scores(part.gold, pred)
    base_pred = [f"bucket_{label:03d}" for label in part.base_labels]
    base_ba, base_tpr, base_tnr = pairwise_scores(part.gold, base_pred)
    err = _pair_error_delta(part.gold, part.base_labels, labels)
    row = {
        "dataset": part.dataset,
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
        "pairs_vetoed": stats.pairs_vetoed,
        "pairs_boosted": stats.pairs_boosted,
        "trace_missing_pairs": stats.trace_missing_pairs,
        "runtime_sec": runtime,
    }
    return row, err


def _summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["run_name"]), str(row["dataset"]))].append(row)
    out = []
    for (run_name, dataset), items in sorted(groups.items()):
        out.append({
            "run_name": run_name,
            "method": items[0]["method"],
            "trace_policy": items[0]["trace_policy"],
            "params": items[0]["params"],
            "dataset": dataset,
            "mean_BA": statistics.mean(float(r["BA"]) for r in items),
            "std_BA": statistics.stdev(float(r["BA"]) for r in items) if len(items) > 1 else 0.0,
            "mean_TPR": statistics.mean(float(r["TPR"]) for r in items),
            "mean_TNR": statistics.mean(float(r["TNR"]) for r in items),
            "mean_delta_BA": statistics.mean(float(r["delta_BA"]) for r in items),
            "num_runs": len(items),
            "mean_pairs_vetoed": statistics.mean(float(r["pairs_vetoed"]) for r in items),
            "mean_pairs_boosted": statistics.mean(float(r["pairs_boosted"]) for r in items),
        })
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conservative trace veto/boost policy experiments.")
    p.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/trace_policy_exp"))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--combo", type=int, default=0)
    p.add_argument("--trace-policy", nargs="+", choices=("none", "veto", "boost", "veto_boost"), default=["none"])
    p.add_argument("--tail-lines", type=int, default=500)
    p.add_argument("--veto-base-min", nargs="+", type=float, default=[0.65])
    p.add_argument("--veto-trace-max", nargs="+", type=float, default=[0.10])
    p.add_argument("--veto-cap", nargs="+", type=float, default=[0.35])
    p.add_argument("--boost-base-ranges", nargs="+", type=_parse_range, default=[(0.30, 0.65)])
    p.add_argument("--boost-trace-min", nargs="+", type=float, default=[0.75])
    p.add_argument("--boost-floor", nargs="+", type=float, default=[0.65])
    p.add_argument("--alpha", type=float, default=0.85)
    p.add_argument("--rich-temperature", type=float, default=0.75)
    p.add_argument("--ensemble-temperature", type=float, default=1.25)
    p.add_argument("--model-root", type=Path, default=Path("/tmp/input_signal_5seed_top/models/llm_dual_struct_det_summary_dim64"))
    p.add_argument("--model-tag", default="llm_dual_struct_det_summary_dim64")
    p.add_argument("--split-root", type=Path, default=Path("/tmp/input_signal_5seed_top/models/llm_dual_struct_det_summary_dim64/splits"))
    p.add_argument("--ensemble-model-dir", type=Path, default=Path("/tmp/pairwise_llm_exp_full/models"))
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--predict-batch-size", type=int, default=100000)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [(d if d.is_absolute() else (PROJECT_ROOT / d).resolve()) for d in args.datasets]
    policies = _policy_grid(args)
    result_rows = []
    delta_rows = []
    trace_cache: dict[str, list[tf.TraceCaseFeature]] = {}
    for seed in args.seeds:
        args.seed_current = seed
        rich_model = plf.load_model_pkg(_find_rich_model(args.model_root, args.model_tag, seed, args.combo))
        ensemble_pkgs = [plf.load_model_pkg(path) for path in _find_ensemble_models(args.ensemble_model_dir, seed, args.combo)]
        _train_info, val_info = _split_parts(datasets, seed, args.combo, args.split_root / f"seed_{seed}")
        val_parts, base_sec = _prepare_parts(val_info, rich_model, ensemble_pkgs, args)
        for part in val_parts:
            key = str(part.input)
            if key not in trace_cache:
                t0 = time.perf_counter()
                trace_cache[key] = tf.build_trace_case_features(part.input, tail_lines=args.tail_lines)
                trace_sec = time.perf_counter() - t0
            else:
                trace_sec = 0.0
            for params in policies:
                row, err = _evaluate_policy(part, trace_cache[key], params)
                params_text = _param_string(params)
                run_name = f"{params.trace_policy}_{params_text}"
                row.update({
                    "run_name": run_name,
                    "seed": seed,
                    "combo": f"{args.combo:03b}",
                    "method": "trace_policy",
                    "alpha": args.alpha,
                    "rich_temp": args.rich_temperature,
                    "ensemble_temp": args.ensemble_temperature,
                    "trace_policy": params.trace_policy,
                    "params": params_text,
                    "trace_runtime_sec": trace_sec,
                    "base_runtime_sec": base_sec,
                })
                err.update({
                    "run_name": run_name,
                    "seed": seed,
                    "combo": f"{args.combo:03b}",
                    "dataset": part.dataset,
                    "trace_policy": params.trace_policy,
                    "params": params_text,
                    "net_FP_delta": int(err["new_fp_pairs"]) - int(err["fixed_fp_pairs"]),
                    "net_FN_delta": int(err["new_fn_pairs"]) - int(err["fixed_fn_pairs"]),
                })
                result_rows.append(row)
                delta_rows.append(err)
                print(
                    f"[eval] seed={seed} dataset={part.dataset} policy={params_text} "
                    f"BA={row['BA']:.6f} base={row['base_BA']:.6f} veto={row['pairs_vetoed']} boost={row['pairs_boosted']}",
                    file=sys.stderr,
                )
    result_header = [
        "run_name", "seed", "combo", "dataset", "method", "alpha", "rich_temp", "ensemble_temp",
        "trace_policy", "params", "BA", "TPR", "TNR", "base_BA", "base_TPR", "base_TNR", "delta_BA",
        "num_cases", "k", "num_pred_clusters", "pairs_vetoed", "pairs_boosted", "trace_missing_pairs",
        "runtime_sec", "trace_runtime_sec", "base_runtime_sec",
    ]
    _write_csv(args.output_dir / "results.csv", result_rows, result_header)
    summary_rows = _summarize(result_rows)
    summary_header = [
        "run_name", "method", "trace_policy", "params", "dataset", "mean_BA", "std_BA", "mean_TPR",
        "mean_TNR", "mean_delta_BA", "num_runs", "mean_pairs_vetoed", "mean_pairs_boosted",
    ]
    _write_csv(args.output_dir / "summary.csv", summary_rows, summary_header)
    delta_header = [
        "run_name", "seed", "combo", "dataset", "trace_policy", "params", "fixed_fn_pairs", "new_fn_pairs",
        "fixed_fp_pairs", "new_fp_pairs", "net_FP_delta", "net_FN_delta",
    ]
    _write_csv(args.output_dir / "deltas.csv", delta_rows, delta_header)
    print_wide([
        {
            "run_name": r["run_name"],
            "method": r["method"],
            "feature_mode": r["trace_policy"],
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
    print(f"Deltas:  {args.output_dir / 'deltas.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
