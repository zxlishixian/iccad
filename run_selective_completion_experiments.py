#!/usr/bin/env python3
"""Selective Completion Post-hoc Refinement experiments.

Loads existing Gated MLP checkpoints, runs selective completion refinement
on the hardest cases only, and compares base vs refined clustering.
No model training.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import json
import pickle
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import pairwise_llm_features as plf
import run_unified_multidataset_experiments as unified
from oof_bridge_mining import fragmentation_rows
from run_experiments import pairwise_scores, read_gold
from selective_completion_refinement import (
    RefinementResult, refine, load_completion_config,
    CompletionUsageStats,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def _save_model_dir() -> Path:
    """Best available model directory."""
    # Prefer OOF bridge models (most recent gated MLP checkpoints)
    for d in [
        Path("/tmp/oof_bridge_set2/models"),
        Path("/tmp/trace_struct_exp/models"),
    ]:
        if d.is_dir() and list(d.glob("*.pt")):
            return d
    # Fallback: train a quick model
    return Path("/tmp/trace_struct_exp/models")


def _find_best_model(target_name: str, seed: int) -> Path | None:
    """Find the best existing Gated MLP model for a target dataset and seed."""
    model_dir = _save_model_dir()
    patterns = [
        f"*{target_name}*seed{seed}*.pt",
        f"*{target_name}*_seed{seed}.pt",
        f"benchmark_set_2_raw_gated_seed{seed}.pt",
    ]
    for pat in patterns:
        candidates = list(model_dir.glob(pat))
        if candidates:
            return candidates[0]
    return None


def load_model_and_predict(
    model_path: Path, features: list[plf.LLMCaseFeature],
    predict_batch_size: int = 100000,
) -> np.ndarray:
    """Load a saved Gated MLP model and compute probability matrix.

    Handles two save formats:
    1. plf.save_model_pkg format (pickle)
    2. torch.save + .preproc.pkl format (bridge/trace experiments)
    """
    suffix = model_path.suffix
    if suffix == ".pkl":
        pkg = plf.load_model_pkg(model_path)
        pkg["feature_mode"] = pkg.get("feature_mode", "llm_dual_struct_det_summary")
        return plf.predict_probability_matrix_sklearn(pkg, features, batch_size=predict_batch_size)

    # Torch save format
    import torch
    from run_unified_multidataset_experiments import make_model

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    preproc_path = model_path.with_suffix(".preproc.pkl")
    if preproc_path.exists():
        with preproc_path.open("rb") as fh:
            preproc = pickle.load(fh)
    else:
        preproc = {}

    input_dim = checkpoint.get("input_dim", 294)
    width = checkpoint.get("width", 256)
    rep_dim = checkpoint.get("representation_dim", 128)
    dropout = checkpoint.get("dropout", 0.2)
    arch = checkpoint.get("model_arch", "gated_mlp")

    # Build args namespace for make_model
    model_args = argparse.Namespace(
        width=width, representation_dim=rep_dim, dropout=dropout,
        model_arch=arch, device="cuda",
    )
    model = make_model(input_dim, 7, width, rep_dim, dropout, arch, model_args)
    model.load_state_dict({k: v for k, v in checkpoint["state_dict"].items()})
    model.eval()

    # Build model_pkg
    pkg = {
        "model": model,
        "scaler": preproc.get("scaler"),
        "llm_reducer": preproc.get("llm_reducer"),
        "llm_summary_reducer": preproc.get("llm_summary_reducer"),
        "llm_reduce_dim": preproc.get("llm_reduce_dim", 64),
        "feature_mode": checkpoint.get("feature_mode", "llm_dual_struct_det_summary"),
        "model_type": "mlp",
        "model_arch": arch,
        "device": "cuda",
        "trace_reduce_dim": preproc.get("trace_reduce_dim", 0),
        "trace_reducer": preproc.get("trace_reducer"),
    }
    return plf.predict_probability_matrix_sklearn(pkg, features, batch_size=predict_batch_size)


def run_one_experiment(
    dataset: Path, seed: int, args: argparse.Namespace,
) -> dict | None:
    """Run refinement experiment for one dataset+seed combination.

    If no saved model exists, train a quick one first.
    """
    t0 = time.perf_counter()
    datasets = [unified.resolve(p) for p in args.datasets]
    target = unified.resolve(dataset) if not dataset.is_absolute() else dataset
    outer = [d for d in datasets if d != target] + [target]

    # Build features
    llm_args = plf._make_llm_args(
        llm_mode="embedding", llm_doc_style="features",
        llm_cache_dir=args.llm_cache_dir, svd_dim=args.svd_dim, llm_dual=True,
    )
    features, _ = plf.build_llm_case_features_for_inputs(
        [d / "input.csv" for d in outer],
        svd_dim=args.svd_dim, llm_args=llm_args,
    )
    slices = unified.build_slices(outer)
    test_slice = slices[-1]
    test_features = features[test_slice.start:test_slice.stop]

    # Fit reducers on train features
    train_features = features[:test_slice.start]
    llm_reducer = plf.fit_llm_reducer(train_features, args.llm_reduce_dim, random_state=seed)
    summary_reducer = plf.fit_llm_summary_reducer(train_features, args.llm_reduce_dim, random_state=seed)
    plf.apply_llm_reducer(test_features, llm_reducer, args.llm_reduce_dim)
    plf.apply_llm_summary_reducer(test_features, summary_reducer, args.llm_reduce_dim)
    k = len(set(test_slice.labels))
    n = len(test_features)

    # Runtime guard
    if n > 100 and not args.force_large:
        max_cases = min(args.completion_max_cases, 3)
        print(f"[runtime_guard] n={n} > 100, limiting to max_cases={max_cases}", file=sys.stderr)
    elif n > 300 and not args.force_large:
        print(f"[runtime_guard] n={n} > 300, completion disabled", file=sys.stderr)
        max_cases = 0
    else:
        max_cases = args.completion_max_cases

    # Load or train model
    model_path = _find_best_model(target.name, seed)
    if model_path is None:
        print(f"[model] no saved model for {target.name} seed={seed}, training quick model...", file=sys.stderr)
        train_args = argparse.Namespace(
            model_arch="gated_mlp", device=args.device, epochs=20,
            steps_per_epoch=100, batch_size=4096, max_pairs_per_dataset=30000,
            llm_reduce_dim=args.llm_reduce_dim, llm_cache_dir=args.llm_cache_dir,
            early_stop_patience=5, graph_gammas=0.0, validation_fraction=0.08,
            svd_dim=args.svd_dim, width=256, representation_dim=128, dropout=0.2,
            lr=0.001, weight_decay=0.0001, focal_gamma=2.0,
            negative_ratio=2.0, hard_negative_ratio=0.5, hard_positive_ratio=0.5,
            positive_mass=0.5, hard_positive_fraction=0.4,
            max_aux_per_dataset=50, aux_batch_size=256,
            connectivity_weight=0.0, connectivity_top_m=5, connectivity_margin=0.5,
            connectivity_edges=100, connectivity_batch_size=256,
            prototype_weight=0.0, prototype_margin=0.5,
            prototype_edges=100, prototype_batch_size=256,
            ranking_weight=0.0, ranking_margin=0.2,
            domain_weight=0.0, domain_grl_scale=0.0, grad_clip=1.0,
            model_arches=["gated_mlp"], seeds=[0], datasets=[],
            holdouts=[], configs_list=["balanced"], bridge_weight=0.0,
            bridge_batch_ratio=0.0, model_type="mlp", configs="balanced",
            output_dir=args.output_dir,
        )
        pair_data = unified.build_pair_data(train_features, slices[:-1], train_args, seed)
        X = plf.build_rich_pair_feature_matrix(
            train_features, pair_data.pairs,
            feature_mode="llm_dual_struct_det_summary",
        )
        pkg = unified.train_unified_model(X, pair_data, train_args, "balanced", seed)
        pkg["feature_mode"] = "llm_dual_struct_det_summary"
        prob_base = unified.predict_probability(pkg, test_features, args.predict_batch_size)
    else:
        print(f"[model] loaded {model_path}", file=sys.stderr)
        prob_base = load_model_and_predict(model_path, test_features, args.predict_batch_size)

    # Base clustering
    labels_base = plf.cluster_from_probability(prob_base, k)
    pred_base = [f"bucket_{l:03d}" for l in labels_base]
    ba_base, tpr_base, tnr_base = pairwise_scores(test_slice.labels, pred_base)
    frag_base = fragmentation_rows(test_slice.cases, test_slice.labels, labels_base)
    print(f"[base] BA={ba_base:.4f} TPR={tpr_base:.4f} TNR={tnr_base:.4f}", file=sys.stderr)

    # Selective completion refinement
    if max_cases > 0 and args.completion_mode != "none":
        case_ids = [f.case_id for f in test_features]
        infos = [f.info for f in test_features]
        config = load_completion_config()

        result = refine(
            prob_base=prob_base.astype(np.float64),
            k=k,
            infos=infos,
            case_ids=case_ids,
            cache_dir=args.completion_cache_dir,
            selection_mode=args.completion_select,
            max_cases=max_cases,
            boost=args.completion_boost,
            veto=args.completion_veto,
            adjust_scope=args.completion_adjust_scope,
            neighbor_top_m=args.completion_neighbor_top_m,
            config=config,
        )

        labels_refined = result.labels_refined
        pred_refined = [f"bucket_{l:03d}" for l in labels_refined]
        ba_ref, tpr_ref, tnr_ref = pairwise_scores(test_slice.labels, pred_refined)
        frag_ref = fragmentation_rows(test_slice.cases, test_slice.labels, labels_refined)
        print(result.completion_stats.log, file=sys.stderr)
        print(f"[refined] BA={ba_ref:.4f} TPR={tpr_ref:.4f} TNR={tnr_ref:.4f} "
              f"calls={result.completion_stats.requests} "
              f"adjustments={len(result.adjustments)}", file=sys.stderr)

        # Save completion examples
        examples_path = args.output_dir / f"completion_examples_{target.name}_seed{seed}.jsonl"
        examples_path.parent.mkdir(parents=True, exist_ok=True)
        with examples_path.open("w") as fh:
            for idx, data in list(result.completion_results.items())[:10]:
                fh.write(json.dumps({"case_index": int(idx), **{k: v for k, v in data.items() if k != "raw_response"}}, default=str) + "\n")

        # Save adjusted pairs
        adj_path = args.output_dir / f"adjusted_pairs_{target.name}_seed{seed}.csv"
        if result.adjustments:
            with adj_path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(result.adjustments[0].keys()))
                w.writeheader(); w.writerows(result.adjustments)

        num_calls = result.completion_stats.requests
        runtime = result.runtime_sec
        delta_ba = ba_ref - ba_base
        delta_tpr = tpr_ref - tpr_base
        delta_tnr = tnr_ref - tnr_base
        num_adj = len(result.adjustments)
    else:
        ba_ref, tpr_ref, tnr_ref = ba_base, tpr_base, tnr_base
        delta_ba, delta_tpr, delta_tnr = 0.0, 0.0, 0.0
        num_calls, runtime, num_adj = 0, 0.0, 0
        frag_ref = frag_base
        labels_refined = labels_base

    top_base = frag_base[0] if frag_base else {}
    top_ref = frag_ref[0] if frag_ref else {}

    return {
        "test_dataset": target.name, "seed": seed,
        "config": f"completion_{args.completion_select}_k{max_cases}",
        "BA_base": ba_base, "TPR_base": tpr_base, "TNR_base": tnr_base,
        "BA_refined": ba_ref, "TPR_refined": tpr_ref, "TNR_refined": tnr_ref,
        "delta_BA": delta_ba, "delta_TPR": delta_tpr, "delta_TNR": delta_tnr,
        "num_completion_calls": num_calls,
        "completion_runtime_sec": round(runtime, 1),
        "num_adjusted_pairs": num_adj,
        "k": k, "num_cases": n,
        "top_frag_base": top_base.get("bug_id", ""),
        "top_frag_base_count": top_base.get("num_pred_fragments", 0),
        "top_frag_ref": top_ref.get("bug_id", ""),
        "top_frag_ref_count": top_ref.get("num_pred_fragments", 0),
        "runtime_sec": time.perf_counter() - t0,
    }


def summarize(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["test_dataset"], r["config"])].append(r)
    out = []
    for (ds, cfg), vals in grouped.items():
        bas_base = [float(r["BA_base"]) for r in vals]
        bas_ref = [float(r["BA_refined"]) for r in vals]
        deltas = [float(r["delta_BA"]) for r in vals]
        out.append({
            "test_dataset": ds, "config": cfg,
            "mean_BA_base": statistics.mean(bas_base),
            "mean_BA_refined": statistics.mean(bas_ref),
            "mean_delta_BA": statistics.mean(deltas),
            "worst_BA_base": min(bas_base),
            "worst_BA_refined": min(bas_ref),
            "mean_calls": statistics.mean([float(r["num_completion_calls"]) for r in vals]),
            "mean_adjustments": statistics.mean([float(r["num_adjusted_pairs"]) for r in vals]),
            "runs": len(vals),
        })
    return sorted(out, key=lambda r: (r["test_dataset"], -r["mean_delta_BA"]))


def print_table(summary):
    print("\n| dataset | config | BA base | BA refined | delta | worst base | worst refined | calls | adj |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary:
        print(f"| {r['test_dataset']} | {r['config']} | {r['mean_BA_base']:.4f} "
              f"| {r['mean_BA_refined']:.4f} | {r['mean_delta_BA']:+.4f} "
              f"| {r['worst_BA_base']:.4f} | {r['worst_BA_refined']:.4f} "
              f"| {r['mean_calls']:.1f} | {r['mean_adjustments']:.1f} |")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Selective completion post-hoc refinement")
    p.add_argument("--datasets", nargs="+", type=Path, default=unified.DEFAULT_DATASETS)
    p.add_argument("--test-datasets", nargs="+", default=["set2"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--completion-mode", choices=["none", "summary_json"], default="summary_json")
    p.add_argument("--completion-select", choices=["entropy_top", "margin_top", "conflict_top", "hybrid"],
                   default="hybrid")
    p.add_argument("--completion-max-cases", type=int, default=5)
    p.add_argument("--completion-boost", type=float, default=0.05)
    p.add_argument("--completion-veto", type=float, default=0.10)
    p.add_argument("--completion-adjust-scope", choices=["selected_selected", "selected_uncertain_neighbors"],
                   default="selected_uncertain_neighbors")
    p.add_argument("--completion-neighbor-top-m", type=int, default=15)
    p.add_argument("--completion-cache-dir", type=Path, default=Path("/tmp/regr_fail_completion_cache"))
    p.add_argument("--force-large", action="store_true", dest="force_large")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--llm-reduce-dim", type=int, default=64)
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--predict-batch-size", type=int, default=100000)
    return p.parse_args(argv)


def resolve_targets(datasets, requested):
    aliases = {"set2": "benchmark_set_2", "stable": "stable_official_like_multitest_v1",
               "vcs": "official_vcs_stage1_dataset_v1", "set1": "benchmark_set_1",
               "first": "first_batch_dataset", "stage2": "stage2_dataset_working",
               "stage3": "stage3_dataset_32bugs_640cases"}
    by_name = {d.name: d for d in datasets}
    return [by_name[aliases.get(n, n)] for n in requested]


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [unified.resolve(p) for p in args.datasets]
    targets = resolve_targets(datasets, args.test_datasets)

    print(f"Targets: {[t.name for t in targets]}, seeds={args.seeds}, "
          f"select={args.completion_select}, max_cases={args.completion_max_cases}",
          file=sys.stderr)

    rows = []
    for target in targets:
        for seed in args.seeds:
            print(f"\n=== {target.name} seed={seed} ===", file=sys.stderr)
            row = run_one_experiment(target, seed, args)
            if row:
                rows.append(row)

    if rows:
        write_csv(args.output_dir / "results.csv", rows, list(rows[0]))
        summary = summarize(rows)
        write_csv(args.output_dir / "summary.csv", summary, list(summary[0]))
        print_table(summary)

        print("\n## Per-seed delta BA")
        for target in sorted(set(r["test_dataset"] for r in rows)):
            sub = [r for r in rows if r["test_dataset"] == target]
            deltas = [f"{r['delta_BA']:+.4f}" for r in sub]
            print(f"  {target}: {deltas}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
