#!/usr/bin/env python3
"""Evaluate seed consensus and recall guardrails for multi-view probabilities.

Experimental only.  The runner consumes cached LODO P_same matrices and never
changes the formal predictor.  Gold labels are used only after clustering for
evaluation; none of the probability corrections inspect gold labels.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import graph_clustering as gc
import official_style_features as osf
from run_experiments import pairwise_scores, read_gold
from run_graph_multiview_experiments import conflict_matrix_from_records
from run_official_full_retrain_experiments import write_csv, write_pred


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    Path("old_fake_dataset/first_batch_dataset"),
    Path("old_fake_dataset/stage2_dataset_working"),
    Path("old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("official_format_fake_dataset/stable_official_like_multitest_v1"),
    Path("official_format_fake_dataset/directed_cross_v2"),
    Path("test_case/problem/benchmark_set_1"),
    Path("test_case/problem/benchmark_set_2"),
]


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _find_prob(prob_dirs: Sequence[Path], filename: str) -> Path:
    for directory in prob_dirs:
        candidate = directory / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing cached probability {filename} in {list(map(str, prob_dirs))}")


def _load_seed_blends(
    prob_dirs: Sequence[Path],
    dataset: str,
    seeds: Sequence[int],
    beta: float,
    base_view: str,
    expert_view: str,
    model_type: str,
    source_graph_method: str,
) -> np.ndarray:
    matrices: list[np.ndarray] = []
    for seed in seeds:
        suffix = f"{model_type}_{source_graph_method}_seed{seed}.npy"
        if beta <= 1e-9:
            prob = np.load(
                _find_prob(prob_dirs, f"{dataset}_{base_view}_{suffix}")
            ).astype(np.float32)
        elif beta >= 1.0 - 1e-9:
            prob = np.load(
                _find_prob(prob_dirs, f"{dataset}_{expert_view}_{suffix}")
            ).astype(np.float32)
        else:
            p_base = np.load(
                _find_prob(prob_dirs, f"{dataset}_{base_view}_{suffix}")
            ).astype(np.float32)
            p_expert = np.load(
                _find_prob(prob_dirs, f"{dataset}_{expert_view}_{suffix}")
            ).astype(np.float32)
            if p_base.shape != p_expert.shape:
                raise ValueError(
                    f"shape mismatch for {dataset} seed={seed}: "
                    f"{p_base.shape} vs {p_expert.shape}"
                )
            prob = ((1.0 - beta) * p_base + beta * p_expert).astype(np.float32)
        prob = (prob + prob.T) * 0.5
        np.fill_diagonal(prob, 1.0)
        matrices.append(prob)
    if not matrices:
        raise ValueError("at least one seed is required")
    return np.stack(matrices, axis=0)


def _coassociation(seed_probs: np.ndarray, k: int, method: str) -> np.ndarray:
    n = int(seed_probs.shape[1])
    coassoc = np.zeros((n, n), dtype=np.float32)
    for prob in seed_probs:
        labels = np.asarray(gc.cluster_probability_graph(prob, k, method).labels, dtype=np.int32)
        coassoc += (labels[:, None] == labels[None, :]).astype(np.float32)
    coassoc /= float(seed_probs.shape[0])
    np.fill_diagonal(coassoc, 1.0)
    return coassoc


def _symmetrize(prob: np.ndarray) -> np.ndarray:
    out = np.clip((np.asarray(prob, dtype=np.float32) + np.asarray(prob, dtype=np.float32).T) * 0.5, 0.0, 1.0)
    np.fill_diagonal(out, 1.0)
    return out


def build_candidates(
    primary_stack: np.ndarray,
    safe_stack: np.ndarray | None,
    k: int,
    source_clusterer: str,
    consensus_weights: Sequence[float],
    safe_weights: Sequence[float],
    guard_consensus: Sequence[float],
    guard_strengths: Sequence[float],
) -> Iterable[tuple[str, np.ndarray, dict]]:
    primary = _symmetrize(np.mean(primary_stack, axis=0))
    coassoc = _coassociation(primary_stack, k, source_clusterer)
    yield "seed_mean", primary, {"prob_adjusted_pairs": 0}

    for weight in consensus_weights:
        prob = _symmetrize((1.0 - weight) * primary + weight * coassoc)
        yield f"consensus_w{weight:.2f}", prob, {"prob_adjusted_pairs": int(np.sum(np.triu(np.abs(prob - primary) > 1e-7, 1)))}

    if safe_stack is None:
        return
    safe = _symmetrize(np.mean(safe_stack, axis=0))
    positive_delta = np.maximum(safe - primary, 0.0)
    for weight in safe_weights:
        prob = _symmetrize(primary + weight * positive_delta)
        yield f"safe_recall_w{weight:.2f}", prob, {
            "prob_adjusted_pairs": int(np.sum(np.triu(positive_delta > 1e-7, 1))),
            "mean_positive_delta": float(np.mean(positive_delta[positive_delta > 0])) if np.any(positive_delta > 0) else 0.0,
        }

    # The guarded variant only accepts recall corrections when the primary
    # seeds repeatedly place the pair in the same cluster.  It cannot lower a
    # P_same value and therefore cannot directly create a split.
    for threshold in guard_consensus:
        confidence = np.clip((coassoc - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
        for strength in guard_strengths:
            delta = strength * confidence * positive_delta
            prob = _symmetrize(primary + delta)
            yield f"guard_c{threshold:.2f}_s{strength:.2f}", prob, {
                "prob_adjusted_pairs": int(np.sum(np.triu(delta > 1e-7, 1))),
                "mean_positive_delta": float(np.mean(delta[delta > 0])) if np.any(delta > 0) else 0.0,
                "consensus_threshold": threshold,
                "guard_strength": strength,
            }


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["policy"]), str(row["clusterer"]))].append(row)
    output: list[dict] = []
    for (policy, clusterer), values in groups.items():
        bas = [float(row["BA"]) for row in values]
        output.append({
            "policy": policy,
            "clusterer": clusterer,
            "mean_BA": float(np.mean(bas)),
            "worst_BA": float(np.min(bas)),
            "std_dataset_BA": float(np.std(bas)),
            "mean_TPR": float(np.mean([float(row["TPR"]) for row in values])),
            "mean_TNR": float(np.mean([float(row["TNR"]) for row in values])),
            "runtime_sec": float(np.sum([float(row["runtime_sec"]) for row in values])),
            "datasets": len(values),
            "dataset_BA": json.dumps({str(row["dataset"]): float(row["BA"]) for row in values}, sort_keys=True),
        })
    return sorted(output, key=lambda row: (row["worst_BA"], row["mean_BA"]), reverse=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental multi-view worst-case guardrail evaluation")
    parser.add_argument("--primary-prob-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--safe-prob-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--primary-beta", type=float, default=0.50)
    parser.add_argument("--safe-beta", type=float, default=0.50)
    parser.add_argument("--base-view", default="dual")
    parser.add_argument("--expert-view", default="quad_event_object_context")
    parser.add_argument("--model-type", default="gbdt")
    parser.add_argument("--source-graph-method", default="agglomerative_complete")
    parser.add_argument("--clusterers", nargs="+", default=["agglomerative_complete", "agglomerative_avg"])
    parser.add_argument("--consensus-weights", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    parser.add_argument("--safe-weights", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    parser.add_argument("--guard-consensus", nargs="+", type=float, default=[0.60, 0.70, 0.80, 0.90])
    parser.add_argument("--guard-strengths", nargs="+", type=float, default=[0.25, 0.50, 1.00])
    parser.add_argument("--signed-conflict-penalty", type=float, default=1.0)
    parser.add_argument("--signed-max-iter", type=int, default=20)
    parser.add_argument("--signed-move-margin", type=float, default=0.0)
    parser.add_argument("--selector-balance-weight", type=float, default=0.2)
    parser.add_argument("--selector-conflict-weight", type=float, default=0.0)
    parser.add_argument("--signed-keep-k", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge-top-m", type=int, default=5)
    parser.add_argument("--merge-threshold", type=float, default=0.75)
    parser.add_argument("--merge-conflict-threshold", type=float, default=0.20)
    parser.add_argument("--merge-internal-threshold", type=float, default=0.55)
    parser.add_argument("--merge-max-merges", type=int, default=2)
    parser.add_argument("--mknn-k", type=int, default=5)
    parser.add_argument("--mknn-threshold", type=float, default=0.65)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    primary_dirs = [resolve(path) for path in args.primary_prob_dirs]
    safe_dirs = [resolve(path) for path in args.safe_prob_dirs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preds").mkdir(exist_ok=True)
    (args.output_dir / "probs").mkdir(exist_ok=True)
    rows: list[dict] = []

    for dataset_arg in args.datasets:
        dataset = resolve(dataset_arg)
        name = dataset.name
        cases = osf.read_cases(dataset / "input.csv")
        gold = read_gold(osf.gold_path(dataset))
        k = len(set(gold))
        conflict = conflict_matrix_from_records(
            osf.build_case_records(name, dataset / "input.csv", gold_csv=None)
        )
        load_started = time.perf_counter()
        primary_stack = _load_seed_blends(
            primary_dirs, name, args.seeds, args.primary_beta, args.base_view,
            args.expert_view, args.model_type, args.source_graph_method,
        )
        safe_stack = None
        if safe_dirs:
            safe_stack = _load_seed_blends(
                safe_dirs, name, args.seeds, args.safe_beta, args.base_view,
                args.expert_view, args.model_type, args.source_graph_method,
            )
        load_sec = time.perf_counter() - load_started
        candidates = list(build_candidates(
            primary_stack, safe_stack, k, args.source_graph_method,
            args.consensus_weights, args.safe_weights, args.guard_consensus,
            args.guard_strengths,
        ))
        for policy, prob, diagnostics in candidates:
            for clusterer in args.clusterers:
                started = time.perf_counter()
                result = gc.cluster_probability_graph(
                    prob,
                    k,
                    clusterer,
                    conflict_matrix=conflict,
                    signed_conflict_penalty=args.signed_conflict_penalty,
                    signed_max_iter=args.signed_max_iter,
                    signed_keep_k=args.signed_keep_k,
                    signed_move_margin=args.signed_move_margin,
                    selector_balance_weight=args.selector_balance_weight,
                    selector_conflict_weight=args.selector_conflict_weight,
                    merge_top_m=args.merge_top_m,
                    merge_threshold=args.merge_threshold,
                    merge_conflict_threshold=args.merge_conflict_threshold,
                    merge_internal_threshold=args.merge_internal_threshold,
                    merge_max_merges=args.merge_max_merges,
                    mknn_k=args.mknn_k,
                    mknn_threshold=args.mknn_threshold,
                )
                cluster_sec = time.perf_counter() - started
                pred_path = args.output_dir / "preds" / f"{name}_{policy}_{clusterer}.csv"
                pred = write_pred(pred_path, cases, result.labels)
                prob_path = args.output_dir / "probs" / f"{name}_{policy}.npy"
                if not prob_path.exists():
                    np.save(prob_path, prob)
                ba, tpr, tnr = pairwise_scores(gold, pred)
                selection_debug = {}
                if result.trajectory:
                    selected = next(
                        (
                            item for item in reversed(result.trajectory)
                            if item.get("action") == "quality_selected"
                        ),
                        None,
                    )
                    if selected is not None:
                        selection_debug = {
                            "selected_clusterer": selected.get("candidate", ""),
                            "selector_quality": selected.get("quality_score", ""),
                            "selector_pair_ll": selected.get("pair_log_likelihood", ""),
                            "selector_entropy": selected.get("cluster_entropy", ""),
                            "selector_max_cluster_fraction": selected.get("max_cluster_fraction", ""),
                        }
                row = {
                    "dataset": name,
                    "policy": policy,
                    "clusterer": clusterer,
                    "BA": ba,
                    "TPR": tpr,
                    "TNR": tnr,
                    "k": k,
                    "cases": len(gold),
                    "num_clusters": len(set(pred)),
                    "seeds": len(args.seeds),
                    "load_sec": load_sec,
                    "runtime_sec": cluster_sec,
                    "pred_path": str(pred_path),
                    "prob_path": str(prob_path),
                    **selection_debug,
                    **diagnostics,
                }
                rows.append(row)
                print(
                    f"[guardrail] dataset={name} policy={policy} clusterer={clusterer} "
                    f"BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f}",
                    flush=True,
                )

    result_fields = sorted({key for row in rows for key in row})
    write_csv(args.output_dir / "results.csv", rows, result_fields)
    summary = summarize(rows)
    summary_fields = [
        "policy", "clusterer", "mean_BA", "worst_BA", "std_dataset_BA",
        "mean_TPR", "mean_TNR", "runtime_sec", "datasets", "dataset_BA",
    ]
    write_csv(args.output_dir / "summary.csv", summary, summary_fields)
    print("\n| rank | policy | clusterer | mean BA | worst BA | TPR | TNR |")
    print("|---:|---|---|---:|---:|---:|---:|")
    for rank, row in enumerate(summary[:20], 1):
        print(
            f"| {rank} | {row['policy']} | {row['clusterer']} | "
            f"{row['mean_BA']:.4f} | {row['worst_BA']:.4f} | "
            f"{row['mean_TPR']:.4f} | {row['mean_TNR']:.4f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
