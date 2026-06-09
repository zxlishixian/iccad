#!/usr/bin/env python3
"""Dataset-aware OOF bridge-edge experiments for systematic fragmentation.

Experimental only. Gold/golden labels are used for OOF mining and evaluation.
The formal regr_fail_bucketing.py prediction path is not modified.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import pairwise_llm_features as plf
from oof_bridge_mining import BridgeEdge, fragmentation_rows, mine_oof_bridge_edges
from run_experiments import pairwise_scores
from run_official_full_retrain_experiments import write_csv, write_pred
import run_unified_multidataset_experiments as unified

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ALIASES = {
    "first": "first_batch_dataset",
    "stage2": "stage2_dataset_working",
    "stage3": "stage3_dataset_32bugs_640cases",
    "vcs": "official_vcs_stage1_dataset_v1",
    "stable": "stable_official_like_multitest_v1",
    "set1": "benchmark_set_1",
    "set2": "benchmark_set_2",
}


def make_training_args(args: argparse.Namespace, output_dir: Path, oof: bool = False) -> argparse.Namespace:
    epochs = args.oof_epochs if oof else args.epochs
    steps = args.oof_steps_per_epoch if oof else args.steps_per_epoch
    ns = unified.parse_args([
        "--output-dir", str(output_dir),
        "--configs", "balanced",
        "--model-arches", "gated_mlp",
        "--device", args.device,
        "--epochs", str(epochs),
        "--steps-per-epoch", str(steps),
        "--batch-size", str(args.batch_size),
        "--max-pairs-per-dataset", str(args.max_pairs_per_dataset),
        "--llm-reduce-dim", str(args.llm_reduce_dim),
        "--llm-cache-dir", str(args.llm_cache_dir),
        "--early-stop-patience", str(args.early_stop_patience),
        "--graph-gammas", "0.0",
    ])
    ns.model_arch = "gated_mlp"
    return ns


def prepare_fold_features(
    train_datasets: Sequence[Path],
    validation_dataset: Path,
    args: argparse.Namespace,
    seed: int,
) -> tuple[list[plf.LLMCaseFeature], list[plf.LLMCaseFeature], list[unified.DatasetSlice], unified.DatasetSlice, object, object]:
    ordered = list(train_datasets) + [validation_dataset]
    slices = unified.build_slices(ordered)
    llm_args = plf._make_llm_args(
        llm_mode="embedding", llm_doc_style="features",
        llm_cache_dir=args.llm_cache_dir, svd_dim=args.svd_dim, llm_dual=True,
    )
    features, _ = plf.build_llm_case_features_for_inputs(
        [dataset / "input.csv" for dataset in ordered],
        svd_dim=args.svd_dim, llm_args=llm_args,
    )
    train_stop = slices[-1].start
    train_features = features[:train_stop]
    validation_features = features[train_stop:]
    if not train_features or train_features[0].llm_vec.size != 768 or train_features[0].llm_summary_vec.size != 768:
        feature_dim = train_features[0].llm_vec.size if train_features else 0
        summary_dim = train_features[0].llm_summary_vec.size if train_features else 0
        raise RuntimeError(
            f"LLM embedding fallback/dimension mismatch: features={feature_dim} summary={summary_dim}; expected 768/768"
        )
    llm_reducer = plf.fit_llm_reducer(train_features, args.llm_reduce_dim, random_state=seed)
    summary_reducer = plf.fit_llm_summary_reducer(train_features, args.llm_reduce_dim, random_state=seed)
    plf.apply_llm_reducer(validation_features, llm_reducer, args.llm_reduce_dim)
    plf.apply_llm_summary_reducer(validation_features, summary_reducer, args.llm_reduce_dim)
    return train_features, validation_features, slices[:-1], slices[-1], llm_reducer, summary_reducer


def oof_predictions_for_outer_train(
    outer_train: Sequence[Path], args: argparse.Namespace, seed: int,
) -> dict[str, dict]:
    output: dict[str, dict] = {}
    train_args = make_training_args(args, args.output_dir / "oof_models", oof=True)
    llm_args = plf._make_llm_args(
        llm_mode="embedding", llm_doc_style="features",
        llm_cache_dir=args.llm_cache_dir, svd_dim=args.svd_dim, llm_dual=True,
    )
    base_features, _ = plf.build_llm_case_features_for_inputs(
        [dataset / "input.csv" for dataset in outer_train],
        svd_dim=args.svd_dim, llm_args=llm_args,
    )
    base_slices = unified.build_slices(outer_train)
    if not base_features or base_features[0].llm_vec.size != 768 or base_features[0].llm_summary_vec.size != 768:
        feature_dim = base_features[0].llm_vec.size if base_features else 0
        summary_dim = base_features[0].llm_summary_vec.size if base_features else 0
        raise RuntimeError(f"OOF LLM embedding mismatch: features={feature_dim} summary={summary_dim}")

    for validation_index, validation_dataset in enumerate(outer_train):
        cache_path = args.output_dir / "oof_cache" / f"seed{seed}_{validation_dataset.name}.pkl"
        if cache_path.is_file() and not args.no_oof_cache:
            with cache_path.open("rb") as handle:
                output[str(validation_dataset.resolve())] = pickle.load(handle)
            print(f"[oof-cache] seed={seed} validation={validation_dataset.name}", flush=True)
            continue
        inner_indices = [idx for idx in range(len(outer_train)) if idx != validation_index]
        inner_train = [outer_train[idx] for idx in inner_indices]
        fold_seed = seed * 1009 + validation_index * 97 + 31
        train_features: list[plf.LLMCaseFeature] = []
        for idx in inner_indices:
            ds = base_slices[idx]
            train_features.extend(copy.deepcopy(base_features[ds.start:ds.stop]))
        validation_ds = base_slices[validation_index]
        validation_features = copy.deepcopy(base_features[validation_ds.start:validation_ds.stop])
        train_slices = unified.build_slices(inner_train)
        llm_reducer = plf.fit_llm_reducer(train_features, args.llm_reduce_dim, random_state=fold_seed)
        summary_reducer = plf.fit_llm_summary_reducer(train_features, args.llm_reduce_dim, random_state=fold_seed)
        plf.apply_llm_reducer(validation_features, llm_reducer, args.llm_reduce_dim)
        plf.apply_llm_summary_reducer(validation_features, summary_reducer, args.llm_reduce_dim)
        pair_data = unified.build_pair_data(train_features, train_slices, train_args, fold_seed)
        X = plf.build_rich_pair_feature_matrix(
            train_features, pair_data.pairs,
            feature_mode="llm_dual_struct_det_summary",
        )
        print(
            f"[oof] seed={seed} validation={validation_dataset.name} "
            f"train_domains={len(inner_train)} pairs={len(X)} input_dim={X.shape[1]}",
            flush=True,
        )
        pkg = unified.train_unified_model(X, pair_data, train_args, "balanced", fold_seed)
        prob = unified.predict_probability(pkg, validation_features, args.predict_batch_size)
        labels = plf.cluster_from_probability(prob, len(set(validation_ds.labels)))
        data = {
            "prob": prob,
            "pred": np.asarray(labels, dtype=np.int64),
            "cases": validation_ds.cases,
            "gold": validation_ds.labels,
            "infos": [feature.info for feature in validation_features],
        }
        output[str(validation_dataset.resolve())] = data
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(data, handle)
    return output


def mine_strategy_edges(
    oof: dict[str, dict], args: argparse.Namespace,
    bridge_select: str, conflict_filter: bool,
    quality_weighted: bool = False,
    quality_min: float = 0.2,
    top_quality_ratio: float | None = None,
    max_edges_total: int | None = None,
    hardest_fragments_only: bool = False,
) -> dict[str, list[BridgeEdge]]:
    result: dict[str, list[BridgeEdge]] = {}
    for dataset_key, data in oof.items():
        # Compute OOF BA for quality scoring
        from run_experiments import pairwise_scores
        pred_buckets = [f"bucket_{l:03d}" for l in data["pred"]]
        oof_ba, _, _ = pairwise_scores(data["gold"], pred_buckets)
        result[dataset_key] = mine_oof_bridge_edges(
            data["cases"], data["gold"], data["prob"], data["pred"], data["infos"],
            bridge_select=bridge_select,
            bridge_threshold=args.bridge_threshold,
            bridge_quantile=args.bridge_quantile,
            max_edges_per_bug=args.max_edges_per_bug,
            max_edges_per_fragment_pair=args.max_edges_per_fragment_pair,
            conflict_filter=conflict_filter,
            quality_weighted=quality_weighted,
            quality_min=quality_min,
            top_quality_ratio=top_quality_ratio,
            max_edges_total=max_edges_total,
            hardest_fragments_only=hardest_fragments_only,
            oof_ba=oof_ba,
        )
    return result


def bridge_pairs_for_final_slices(
    train_slices: Sequence[unified.DatasetSlice], edge_map: dict[str, list[BridgeEdge]],
) -> tuple[list[tuple[int, int]], list[float], list[dict]]:
    pairs: list[tuple[int, int]] = []
    weights: list[float] = []
    rows: list[dict] = []
    for dataset_slice in train_slices:
        key = str(dataset_slice.path.resolve())
        for edge in edge_map.get(key, []):
            pairs.append((dataset_slice.start + edge.i, dataset_slice.start + edge.j))
            weights.append(float(edge.weight))
            rows.append({
                "dataset": dataset_slice.name,
                "case_i": dataset_slice.cases[edge.i],
                "case_j": dataset_slice.cases[edge.j],
                "bug_id": edge.bug_id,
                "fragment_i": edge.fragment_i,
                "fragment_j": edge.fragment_j,
                "oof_probability": edge.oof_probability,
                "weight": edge.weight,
                "quality": edge.quality,
            })
    return pairs, weights, rows


def save_model(
    path: Path, pkg: dict, llm_reducer: object, summary_reducer: object,
    args: argparse.Namespace, config_name: str,
) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in pkg["model"].state_dict().items()},
        "input_dim": pkg["input_dim"], "width": pkg["width"],
        "representation_dim": pkg["representation_dim"], "dropout": pkg["dropout"],
        "model_arch": "gated_mlp", "config": config_name,
        "bridge_weight": pkg["bridge_weight"], "num_bridge_edges": pkg["num_bridge_edges"],
    }, path)
    with path.with_suffix(".preproc.pkl").open("wb") as handle:
        pickle.dump({
            "scaler": pkg["scaler"], "llm_reducer": llm_reducer,
            "llm_summary_reducer": summary_reducer,
            "llm_reduce_dim": args.llm_reduce_dim,
        }, handle)


def strategy_specs(args: argparse.Namespace) -> list[dict]:
    specs = [{"name": "raw_gated", "weight": 0.0, "select": "none", "conflict": False,
              "quality_weighted": False, "quality_min": 0.2, "top_quality_ratio": None,
              "max_edges_total": None, "hardest_fragments_only": False}]
    # Baseline bridge configs
    for weight in args.bridge_weights:
        specs.append({"name": f"bridge_abs_w{weight:g}", "weight": weight, "select": "abs_threshold",
                      "conflict": False, "quality_weighted": False, "quality_min": 0.2,
                      "top_quality_ratio": None, "max_edges_total": None,
                      "hardest_fragments_only": False})
    # Quality-weighted bridge
    if args.bridge_quality_weighted:
        specs.append({"name": "bridge_quality_w0.2", "weight": 0.2, "select": "abs_threshold",
                      "conflict": False, "quality_weighted": True, "quality_min": args.bridge_quality_min,
                      "top_quality_ratio": None, "max_edges_total": None,
                      "hardest_fragments_only": False})
    # Low-budget bridge
    if args.bridge_max_edges_total:
        for budget in args.bridge_max_edges_total:
            specs.append({"name": f"bridge_budget{int(budget)}_w0.2", "weight": 0.2,
                          "select": "abs_threshold", "conflict": False, "quality_weighted": False,
                          "quality_min": 0.2, "top_quality_ratio": None,
                          "max_edges_total": int(budget), "hardest_fragments_only": False})
    # Quality + low-budget
    if args.bridge_quality_weighted and args.bridge_max_edges_total:
        for budget in args.bridge_max_edges_total:
            specs.append({"name": f"bridge_quality_budget{int(budget)}_w0.2", "weight": 0.2,
                          "select": "abs_threshold", "conflict": False,
                          "quality_weighted": True, "quality_min": args.bridge_quality_min,
                          "top_quality_ratio": None, "max_edges_total": int(budget),
                          "hardest_fragments_only": False})
    # Hardest fragments only
    if args.bridge_hardest_fragments:
        specs.append({"name": "bridge_hardest_w0.2", "weight": 0.2, "select": "abs_threshold",
                      "conflict": False, "quality_weighted": False, "quality_min": 0.2,
                      "top_quality_ratio": None, "max_edges_total": None,
                      "hardest_fragments_only": True})
    # Deduplicate
    seen = set()
    unique = []
    for spec in specs:
        if spec["name"] not in seen:
            unique.append(spec); seen.add(spec["name"])
    return unique


def run_outer_fold(args: argparse.Namespace, target: Path, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    started = time.perf_counter()
    outer_train = [dataset for dataset in args.datasets if dataset != target]
    oof = oof_predictions_for_outer_train(outer_train, args, seed)

    # Cache edge maps by (select, conflict, quality_weighted, hardest, max_total)
    edge_map_cache: dict[tuple, dict[str, list[BridgeEdge]]] = {}

    def get_edges(select: str, conflict: bool, quality_weighted: bool,
                  quality_min: float, top_quality_ratio: float | None,
                  max_edges_total: int | None, hardest_fragments_only: bool) -> dict[str, list[BridgeEdge]]:
        cache_key = (select, conflict, quality_weighted, quality_min,
                     top_quality_ratio, max_edges_total, hardest_fragments_only)
        if cache_key not in edge_map_cache:
            edge_map_cache[cache_key] = mine_strategy_edges(
                oof, args, select, conflict,
                quality_weighted=quality_weighted, quality_min=quality_min,
                top_quality_ratio=top_quality_ratio,
                max_edges_total=max_edges_total,
                hardest_fragments_only=hardest_fragments_only,
            )
        return edge_map_cache[cache_key]

    train_features, test_features, train_slices, test_slice, llm_reducer, summary_reducer = prepare_fold_features(
        outer_train, target, args, seed
    )
    train_args = make_training_args(args, args.output_dir / "models", oof=False)
    pair_data = unified.build_pair_data(train_features, train_slices, train_args, seed)
    X = plf.build_rich_pair_feature_matrix(
        train_features, pair_data.pairs, feature_mode="llm_dual_struct_det_summary"
    )
    k = len(set(test_slice.labels))
    result_rows: list[dict] = []
    fragmentation_output: list[dict] = []
    bridge_output: list[dict] = []
    for spec in strategy_specs(args):
        config_started = time.perf_counter()
        if spec["select"] == "none":
            bridge_pairs, bridge_weights, bridge_rows = [], [], []
        else:
            edge_map = get_edges(
                spec["select"], spec["conflict"],
                spec["quality_weighted"], spec["quality_min"],
                spec.get("top_quality_ratio"), spec.get("max_edges_total"),
                spec["hardest_fragments_only"],
            )
            bridge_pairs, bridge_weights, bridge_rows = bridge_pairs_for_final_slices(
                train_slices, edge_map
            )
        bridge_X = None
        if bridge_pairs:
            bridge_X = plf.build_rich_pair_feature_matrix(
                train_features, bridge_pairs, feature_mode="llm_dual_struct_det_summary"
            )
        pkg = unified.train_unified_model(
            X, pair_data, train_args, "balanced", seed,
            bridge_X=bridge_X, bridge_weight=spec["weight"],
            bridge_batch_ratio=args.bridge_batch_ratio,
            bridge_weights=bridge_weights if spec["quality_weighted"] and bridge_weights else None,
        )
        prob = unified.predict_probability(pkg, test_features, args.predict_batch_size)
        labels = plf.cluster_from_probability(prob, k)
        pred_path = args.output_dir / "preds" / f"{target.name}_{spec['name']}_seed{seed}.csv"
        prob_path = args.output_dir / "probs" / f"{target.name}_{spec['name']}_seed{seed}.npy"
        model_path = args.output_dir / "models" / f"{target.name}_{spec['name']}_seed{seed}.pt"
        pred = write_pred(pred_path, test_slice.cases, labels)
        prob_path.parent.mkdir(parents=True, exist_ok=True); np.save(prob_path, prob)
        save_model(model_path, pkg, llm_reducer, summary_reducer, args, spec["name"])
        ba, tpr, tnr = pairwise_scores(test_slice.labels, pred)
        frag_rows = fragmentation_rows(test_slice.cases, test_slice.labels, labels)
        for row in frag_rows:
            full = {"test_dataset": target.name, "seed": seed, "config_name": spec["name"], **row}
            fragmentation_output.append(full)
        frag_path = args.output_dir / f"fragmentation_seed{seed}_{spec['name']}.csv"
        write_csv(frag_path, frag_rows, list(frag_rows[0]))
        top = frag_rows[0]
        focus = next((row for row in frag_rows if row["bug_id"] == "bug_107"), top)
        for row in bridge_rows:
            bridge_output.append({"test_dataset": target.name, "seed": seed, "config_name": spec["name"], **row})
        if bridge_rows:
            edge_path = args.output_dir / f"bridge_edges_seed{seed}_{spec['name']}.csv"
            write_csv(edge_path, bridge_rows, list(bridge_rows[0]))
        result_rows.append({
            "test_dataset": target.name, "seed": seed, "config_name": spec["name"],
            "bridge_loss": "none" if spec["weight"] == 0 else "oof",
            "bridge_weight": spec["weight"], "bridge_select": spec["select"],
            "bridge_threshold": args.bridge_threshold, "bridge_quantile": args.bridge_quantile,
            "conflict_filter": spec["conflict"],
            "quality_weighted": spec["quality_weighted"],
            "hardest_fragments_only": spec["hardest_fragments_only"],
            "num_bridge_edges": len(bridge_pairs),
            "BA": ba, "TPR": tpr, "TNR": tnr,
            "top_fragmented_bug": top["bug_id"], "top_fragment_count": top["num_pred_fragments"],
            "top_largest_fragment_ratio": top["largest_fragment_ratio"],
            "top_intra_bug_TPR": top["intra_bug_TPR"],
            "focus_bug": focus["bug_id"], "focus_fragments": focus["num_pred_fragments"],
            "focus_largest_fragment_ratio": focus["largest_fragment_ratio"],
            "focus_intra_bug_TPR": focus["intra_bug_TPR"],
            "runtime_sec": time.perf_counter() - config_started,
            "model_path": str(model_path), "pred_path": str(pred_path), "prob_path": str(prob_path),
        })
        print(
            f"[eval] target={target.name} seed={seed} config={spec['name']} edges={len(bridge_pairs)} "
            f"BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f} "
            f"focus_fragments={focus['num_pred_fragments']} focus_TPR={focus['intra_bug_TPR']:.4f}",
            flush=True,
        )
    print(f"[outer] target={target.name} seed={seed} runtime={time.perf_counter()-started:.1f}s", flush=True)
    return result_rows, fragmentation_output, bridge_output


def summarize(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows: grouped[(row["test_dataset"], row["config_name"])].append(row)
    output = []
    for (dataset, config), values in grouped.items():
        bas = np.asarray([float(row["BA"]) for row in values])
        output.append({
            "test_dataset": dataset, "config_name": config,
            "mean_BA": float(np.mean(bas)), "std_BA": float(np.std(bas)),
            "mean_TPR": float(np.mean([float(row["TPR"]) for row in values])),
            "mean_TNR": float(np.mean([float(row["TNR"]) for row in values])),
            "worst_BA": float(np.min(bas)),
            "mean_num_bridge_edges": float(np.mean([float(row["num_bridge_edges"]) for row in values])),
            "mean_focus_fragments": float(np.mean([float(row["focus_fragments"]) for row in values])),
            "mean_focus_largest_fragment_ratio": float(np.mean([float(row["focus_largest_fragment_ratio"]) for row in values])),
            "mean_focus_intra_bug_TPR": float(np.mean([float(row["focus_intra_bug_TPR"]) for row in values])),
            "runs": len(values),
        })
    return sorted(output, key=lambda row: (row["test_dataset"], -row["mean_BA"]))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental dataset-aware OOF bridge-edge mining")
    parser.add_argument("--datasets", nargs="+", type=Path, default=unified.DEFAULT_DATASETS)
    parser.add_argument("--test-datasets", nargs="+", default=["set2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 5])
    parser.add_argument("--bridge-weights", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--bridge-select", nargs="+", choices=("abs_threshold", "bug_quantile"), default=["abs_threshold", "bug_quantile"])
    parser.add_argument("--bridge-threshold", type=float, default=0.35)
    parser.add_argument("--bridge-quantile", type=float, default=0.20)
    parser.add_argument("--bridge-batch-ratio", type=float, default=0.25)
    parser.add_argument("--max-edges-per-bug", type=int, default=200)
    parser.add_argument("--max-edges-per-fragment-pair", type=int, default=20)
    parser.add_argument("--bridge-quality-weighted", action="store_true",
                        help="Use quality scores to weight bridge edges in loss")
    parser.add_argument("--bridge-quality-min", type=float, default=0.2,
                        help="Minimum quality score for bridge edges")
    parser.add_argument("--bridge-max-edges-total", nargs="+", type=float, default=None,
                        help="Global budget cap on bridge edges (e.g. 100 200 300)")
    parser.add_argument("--bridge-hardest-fragments", action="store_true",
                        help="Only bridge hardest (most separated) fragment pairs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--svd-dim", type=int, default=64)
    parser.add_argument("--llm-reduce-dim", type=int, default=64)
    parser.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--oof-epochs", type=int, default=15)
    parser.add_argument("--oof-steps-per-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-pairs-per-dataset", type=int, default=30000)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--predict-batch-size", type=int, default=100000)
    parser.add_argument("--no-oof-cache", action="store_true")
    return parser.parse_args(argv)


def resolve_targets(datasets: Sequence[Path], requested: Sequence[str]) -> list[Path]:
    by_name = {dataset.name: dataset for dataset in datasets}
    targets = []
    for name in requested:
        resolved_name = DATASET_ALIASES.get(name, name)
        if resolved_name not in by_name:
            raise KeyError(f"unknown test dataset {name}; available={sorted(by_name)}")
        targets.append(by_name[resolved_name])
    return targets


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.datasets = [unified.resolve(path) for path in args.datasets]
    targets = resolve_targets(args.datasets, args.test_datasets)
    rows: list[dict] = []; fragments: list[dict] = []; edges: list[dict] = []
    for target in targets:
        for seed in args.seeds:
            fold_rows, fold_fragments, fold_edges = run_outer_fold(args, target, seed)
            rows.extend(fold_rows); fragments.extend(fold_fragments); edges.extend(fold_edges)
            write_csv(args.output_dir / "results.partial.csv", rows, list(rows[0]))
    write_csv(args.output_dir / "results.csv", rows, list(rows[0]))
    summary = summarize(rows)
    write_csv(args.output_dir / "summary.csv", summary, list(summary[0]))
    if fragments: write_csv(args.output_dir / "fragmentation.csv", fragments, list(fragments[0]))
    if edges: write_csv(args.output_dir / "bridge_edges.csv", edges, list(edges[0]))
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "datasets": [str(path) for path in args.datasets], "test_datasets": [path.name for path in targets],
        "seeds": args.seeds, "feature_mode": "llm_dual_struct_det_summary",
        "oof_mode": "leave_one_training_dataset_out", "formal_predictor_modified": False,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "datasets"},
    }, indent=2) + "\n")
    print("\n| dataset | config | mean BA | std | worst | TPR | TNR | edges | focus fragments | focus TPR |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary:
        print(f"| {row['test_dataset']} | {row['config_name']} | {row['mean_BA']:.4f} | {row['std_BA']:.4f} | {row['worst_BA']:.4f} | {row['mean_TPR']:.4f} | {row['mean_TNR']:.4f} | {row['mean_num_bridge_edges']:.1f} | {row['mean_focus_fragments']:.2f} | {row['mean_focus_intra_bug_TPR']:.4f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
