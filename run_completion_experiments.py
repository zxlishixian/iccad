#!/usr/bin/env python3
"""Completion JSON semantic features experiments for pairwise Gated MLP.

Uses NVIDIA NIM completion endpoint (qwen3-coder-480b) via OpenAI-compatible API
to extract structured failure semantics, then adds them as extra pairwise features.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import pickle
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import completion_case_features as ccf
import pairwise_features as pf
import pairwise_llm_features as plf
import run_unified_multidataset_experiments as unified
from oof_bridge_mining import fragmentation_rows
from run_experiments import pairwise_scores

PROJECT_ROOT = Path(__file__).resolve().parent


def _build_features(
    datasets: Sequence[Path],
    args: argparse.Namespace,
    seed: int,
) -> tuple[list[plf.LLMCaseFeature], unified.PairData, list[unified.DatasetSlice],
           unified.DatasetSlice, argparse.Namespace, int, object, object]:
    """Build features and optionally attach completion features."""
    llm_args = plf._make_llm_args(
        llm_mode="embedding", llm_doc_style="features",
        llm_cache_dir=args.llm_cache_dir, svd_dim=args.svd_dim, llm_dual=True,
    )
    features, _ = plf.build_llm_case_features_for_inputs(
        [d / "input.csv" for d in datasets],
        svd_dim=args.svd_dim, llm_args=llm_args,
    )
    slices = unified.build_slices(datasets)
    train_slices = slices[:-1]
    test_slice = slices[-1]
    train_stop = test_slice.start
    train_features = features[:train_stop]
    test_features = features[train_stop:]

    # Attach completion features
    completion_valid = True
    completion_ok_rate = 1.0
    completion_unknown_ratio = 0.0
    if args.completion_mode != "none":
        all_inputs = [sl.path / "input.csv" for sl in slices]
        completion_features, completion_debug = ccf.build_completion_case_features(
            all_inputs, cache_dir=args.completion_cache_dir,
            strict=args.strict_completion,
            cache_ignore_errors=(not args.completion_cache_keep_errors),
        )
        # Map by case_id for robust matching
        completion_by_case = {cf.case_id: cf for cf in completion_features}
        for feat in features:
            if feat.case_id in completion_by_case:
                feat.completion_feature = completion_by_case[feat.case_id]

        # Compute validity metrics
        total = len(completion_features)
        ok_count = sum(1 for cf in completion_features if cf.status == "ok")
        parse_fail = sum(1 for cf in completion_features if cf.status == "invalid_json")
        req_error = sum(1 for cf in completion_features if cf.status.startswith("request_error"))
        unknown_count = sum(1 for cf in completion_features if cf.status != "ok")
        completion_ok_rate = ok_count / max(1, total)
        parse_failure_rate = parse_fail / max(1, total)
        req_error_rate = req_error / max(1, total)
        completion_unknown_ratio = unknown_count / max(1, total)

        # Validity gate
        completion_valid = (
            completion_ok_rate >= args.min_completion_ok_rate
            and completion_unknown_ratio <= args.max_completion_unknown_ratio
        )

        print(f"[completion] total={total} ok={ok_count} ({completion_ok_rate:.1%}) "
              f"parse_fail={parse_fail} req_error={req_error} "
              f"unknown_ratio={completion_unknown_ratio:.1%} "
              f"valid={completion_valid}", file=sys.stderr)

        if not completion_valid and not args.allow_invalid_completion:
            print("[completion] ERROR: completion validity gate FAILED. "
                  "Skipping training. Use --allow-invalid-completion to override.",
                  file=sys.stderr)
            # Return a dummy result row marking as invalid
            return {
                "test_dataset": "", "seed": seed,
                "completion_mode": args.completion_mode,
                "feature_mode": "INVALID_COMPLETION",
                "input_dim": 0, "BA": float("nan"), "TPR": float("nan"), "TNR": float("nan"),
                "k": 0, "num_cases": 0,
                "top_fragmented_bug": "", "top_fragment_count": 0,
                "top_largest_fragment_ratio": 0.0, "top_intra_bug_TPR": 0.0,
                "runtime_sec": 0.0, "model_path": "", "pred_path": "",
                "completion_ok_rate": completion_ok_rate,
                "completion_unknown_ratio": completion_unknown_ratio,
                "completion_valid": False,
            }

        # Save completion examples
        examples_path = args.output_dir / "completion_examples.jsonl"
        examples_path.parent.mkdir(parents=True, exist_ok=True)
        with examples_path.open("w") as fh:
            for cf in completion_features[:20]:
                fh.write(json.dumps(cf.__dict__, default=str) + "\n")

    # Fit reducers
    llm_reducer = plf.fit_llm_reducer(train_features, args.llm_reduce_dim, random_state=seed)
    summary_reducer = plf.fit_llm_summary_reducer(train_features, args.llm_reduce_dim, random_state=seed)
    plf.apply_llm_reducer(test_features, llm_reducer, args.llm_reduce_dim)
    plf.apply_llm_summary_reducer(test_features, summary_reducer, args.llm_reduce_dim)

    train_args = argparse.Namespace(
        output_dir=args.output_dir, configs="balanced",
        model_arch="gated_mlp", device=args.device,
        epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size, max_pairs_per_dataset=args.max_pairs_per_dataset,
        llm_reduce_dim=args.llm_reduce_dim, llm_cache_dir=args.llm_cache_dir,
        early_stop_patience=args.early_stop_patience,
        graph_gammas=0.0, validation_fraction=0.08,
        svd_dim=args.svd_dim,
        negative_ratio=2.0, hard_negative_ratio=0.5, hard_positive_ratio=0.5,
        positive_mass=0.5, hard_positive_fraction=0.4,
        lr=0.001, weight_decay=0.0001,
        focal_gamma=2.0, width=256, representation_dim=128, dropout=0.2,
        max_aux_per_dataset=50, aux_batch_size=256,
        connectivity_weight=0.0, connectivity_top_m=5, connectivity_margin=0.5,
        connectivity_edges=100, connectivity_batch_size=256,
        prototype_weight=0.0, prototype_margin=0.5,
        prototype_edges=100, prototype_batch_size=256,
        ranking_weight=0.0, ranking_margin=0.2,
        domain_weight=0.0, domain_grl_scale=0.0, grad_clip=1.0,
        model_arches=["gated_mlp"], seeds=[0], datasets=[],
        holdouts=[], configs_list=["balanced"],
        bridge_weight=0.0, bridge_batch_ratio=0.0,
        model_type="mlp",
    )
    pair_data = unified.build_pair_data(train_features, train_slices, train_args, seed)
    k = len(set(test_slice.labels))

    return (train_features, test_features, pair_data, train_slices, test_slice,
            train_args, k, llm_reducer, summary_reducer)


def run_one_fold(
    datasets: Sequence[Path], target: Path, args: argparse.Namespace,
    seed: int, completion_mode: str,
) -> dict:
    t0 = time.perf_counter()
    (train_features, test_features, pair_data, train_slices, test_slice,
     train_args, k, llm_reducer, summary_reducer) = _build_features(
        datasets, args, seed
    )

    feature_mode = ("llm_dual_struct_det_summary_completion" if completion_mode != "none"
                    else "llm_dual_struct_det_summary")
    X = plf.build_rich_pair_feature_matrix(
        train_features, pair_data.pairs, feature_mode=feature_mode,
    )

    pkg = unified.train_unified_model(X, pair_data, train_args, "balanced", seed)
    pkg["feature_mode"] = feature_mode
    prob = unified.predict_probability(pkg, test_features, args.predict_batch_size)
    labels = plf.cluster_from_probability(prob, k)
    pred_buckets = [f"bucket_{l:03d}" for l in labels]
    ba, tpr, tnr = pairwise_scores(test_slice.labels, pred_buckets)

    frag = fragmentation_rows(test_slice.cases, test_slice.labels, labels)
    top = frag[0] if frag else {}

    # Save model
    model_path = args.output_dir / "models" / f"{target.name}_comp_{completion_mode}_seed{seed}.pt"
    pred_path = args.output_dir / "preds" / f"{target.name}_comp_{completion_mode}_seed{seed}.csv"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in pkg["model"].state_dict().items()},
        "input_dim": pkg["input_dim"], "width": pkg["width"],
        "representation_dim": pkg["representation_dim"],
        "dropout": pkg["dropout"], "model_arch": "gated_mlp",
        "feature_mode": feature_mode, "completion_mode": completion_mode,
    }, model_path)
    with model_path.with_suffix(".preproc.pkl").open("wb") as fh:
        pickle.dump({
            "scaler": pkg["scaler"], "llm_reducer": llm_reducer,
            "llm_summary_reducer": summary_reducer,
            "llm_reduce_dim": args.llm_reduce_dim,
        }, fh)
    with pred_path.open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["bucket"])
        for b in pred_buckets: w.writerow([b])

    runtime = time.perf_counter() - t0
    return {
        "test_dataset": target.name, "seed": seed,
        "completion_mode": completion_mode, "feature_mode": feature_mode,
        "input_dim": pkg["input_dim"],
        "BA": ba, "TPR": tpr, "TNR": tnr,
        "k": k, "num_cases": len(test_slice.cases),
        "top_fragmented_bug": top.get("bug_id", ""),
        "top_fragment_count": top.get("num_pred_fragments", 0),
        "top_largest_fragment_ratio": top.get("largest_fragment_ratio", 0.0),
        "top_intra_bug_TPR": top.get("intra_bug_TPR", 0.0),
        "runtime_sec": runtime,
        "model_path": str(model_path), "pred_path": str(pred_path),
        "completion_ok_rate": completion_ok_rate,
        "completion_unknown_ratio": completion_unknown_ratio,
        "completion_valid": completion_valid,
    }


def summarize(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["test_dataset"], row["completion_mode"])].append(row)
    out = []
    for (ds, mode), vals in grouped.items():
        bas = [float(r["BA"]) for r in vals]
        tprs = [float(r["TPR"]) for r in vals]
        tnrs = [float(r["TNR"]) for r in vals]
        out.append({
            "test_dataset": ds, "completion_mode": mode,
            "mean_BA": statistics.mean(bas),
            "std_BA": statistics.stdev(bas) if len(bas) > 1 else 0.0,
            "worst_BA": min(bas), "best_BA": max(bas),
            "mean_TPR": statistics.mean(tprs),
            "mean_TNR": statistics.mean(tnrs),
            "mean_top_fragments": statistics.mean([float(r["top_fragment_count"]) for r in vals]),
            "runs": len(vals),
        })
    return sorted(out, key=lambda r: (r["test_dataset"], -r["mean_BA"]))


def print_summary_table(rows: Sequence[dict]) -> None:
    print("\n| dataset | mode | mean BA | std | worst | best | TPR | TNR | top frag |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(f"| {row['test_dataset']} | {row['completion_mode']} | {row['mean_BA']:.4f} "
              f"| {row['std_BA']:.4f} | {row['worst_BA']:.4f} | {row['best_BA']:.4f} "
              f"| {row['mean_TPR']:.4f} | {row['mean_TNR']:.4f} "
              f"| {row['mean_top_fragments']:.1f} |")


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Completion JSON features experiments")
    p.add_argument("--datasets", nargs="+", type=Path, default=unified.DEFAULT_DATASETS)
    p.add_argument("--test-datasets", nargs="+", default=["set2"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--completion-mode", choices=["none", "summary_json"], default="summary_json")
    p.add_argument("--completion-cache-dir", type=Path, default=Path("/tmp/regr_fail_completion_cache"))
    p.add_argument("--strict-completion", action="store_true",
                   help="Exit with error on any completion failure")
    p.add_argument("--min-completion-ok-rate", type=float, default=0.8,
                   help="Minimum completion ok rate to allow training (default 0.8)")
    p.add_argument("--max-completion-unknown-ratio", type=float, default=0.3,
                   help="Maximum completion unknown ratio to allow training (default 0.3)")
    p.add_argument("--allow-invalid-completion", action="store_true",
                   help="Allow training even if completion validity gate fails")
    p.add_argument("--completion-cache-keep-errors", action="store_true",
                   help="Keep error caches instead of re-fetching")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--llm-reduce-dim", type=int, default=64)
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--max-pairs-per-dataset", type=int, default=30000)
    p.add_argument("--early-stop-patience", type=int, default=6)
    p.add_argument("--predict-batch-size", type=int, default=100000)
    return p.parse_args(argv)


def resolve_test_datasets(datasets, requested):
    aliases = {
        "set2": "benchmark_set_2", "stable": "stable_official_like_multitest_v1",
        "vcs": "official_vcs_stage1_dataset_v1", "set1": "benchmark_set_1",
        "first": "first_batch_dataset", "stage2": "stage2_dataset_working",
        "stage3": "stage3_dataset_32bugs_640cases",
    }
    by_name = {d.name: d for d in datasets}
    targets = []
    for name in requested:
        n = aliases.get(name, name)
        if n not in by_name:
            raise KeyError(f"Unknown: {name} (available: {sorted(by_name)})")
        targets.append(by_name[n])
    return targets


def main(argv=None):
    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [unified.resolve(p) for p in args.datasets]
    targets = resolve_test_datasets(datasets, args.test_datasets)

    print(f"Test datasets: {[t.name for t in targets]}", file=sys.stderr)
    print(f"Completion mode: {args.completion_mode}", file=sys.stderr)
    print(f"Seeds: {args.seeds}", file=sys.stderr)

    # Run both baseline and completion for comparison
    all_rows = []
    for target in targets:
        outer = [d for d in datasets if d != target] + [target]
        for seed in args.seeds:
            # Baseline (no completion)
            for mode in ["none", args.completion_mode]:
                print(f"\n=== target={target.name} seed={seed} completion={mode} ===", file=sys.stderr)
                args.completion_mode = mode  # hack to reuse for both modes
                row = run_one_fold(outer, target, args, seed, mode)
                all_rows.append(row)
                print(f"  BA={row['BA']:.4f} TPR={row['TPR']:.4f} TNR={row['TNR']:.4f} "
                      f"dim={row['input_dim']} top_frag={row['top_fragmented_bug']} "
                      f"fragments={row['top_fragment_count']}",
                      file=sys.stderr)

    write_csv(args.output_dir / "results.csv", all_rows, list(all_rows[0]))
    summary = summarize(all_rows)
    write_csv(args.output_dir / "summary.csv", summary, list(summary[0]))
    print_summary_table(summary)

    # Per-seed details
    print("\n## Per-seed details")
    for target in sorted(set(r["test_dataset"] for r in all_rows)):
        print(f"\n### {target}")
        for mode in ["none", "summary_json"]:
            sub = [r for r in all_rows if r["test_dataset"] == target and r["completion_mode"] == mode]
            if sub:
                bas = [r["BA"] for r in sub]
                print(f"  {mode}: BA {[f'{b:.4f}' for b in bas]} "
                      f"mean={statistics.mean(bas):.4f} worst={min(bas):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
