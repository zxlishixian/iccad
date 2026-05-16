#!/usr/bin/env python3
"""Train experimental pairwise same-bug models with LLM-augmented features.

Supports three backends:
  A. logistic  – sklearn LogisticRegression with StandardScaler
  B. gbdt      – sklearn HistGradientBoostingClassifier
  C. mlp       – small PyTorch MLP (128→64→1)

Training reads gold.csv for labels; inference does not.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import pairwise_llm_features as plf
import regr_fail_bucketing as rfb
from run_experiments import pairwise_scores, read_gold
from run_half_split_experiments import DEFAULT_DATASETS, opposite_part, part_for_bit, stratified_half_split


def _normalized_matrix(vectors: Sequence[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    mat = np.vstack(vectors).astype(np.float32)
    if mat.ndim != 2 or mat.shape[1] == 0:
        return np.zeros((len(vectors), len(vectors)), dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.maximum(norms, 1e-12)
    return mat @ mat.T


def _case_info(features: Sequence[plf.LLMCaseFeature], idx: int, key: str) -> str:
    value = (features[idx].info or {}).get(key, "")
    return str(value) if value is not None else ""


def _same_nonempty(features: Sequence[plf.LLMCaseFeature], i: int, j: int, key: str) -> bool:
    vi = _case_info(features, i, key)
    vj = _case_info(features, j, key)
    return bool(vi and vj and vi == vj)


def _build_similarity_matrix(features: Sequence[plf.LLMCaseFeature]) -> tuple[np.ndarray, np.ndarray]:
    n = len(features)
    if n == 0:
        empty = np.zeros((0, 0), dtype=np.float32)
        return empty, empty
    sims = []
    det_sim = _normalized_matrix([f.det_vec for f in features])
    if det_sim.size:
        sims.append(det_sim)
    if any(f.has_llm for f in features):
        llm_sim = _normalized_matrix([f.llm_vec for f in features])
        if llm_sim.size:
            sims.append(llm_sim)
    if any(f.has_llm_summary for f in features):
        summary_sim = _normalized_matrix([f.llm_summary_vec for f in features])
        if summary_sim.size:
            sims.append(summary_sim)
    if not sims:
        combined = np.zeros((n, n), dtype=np.float32)
    else:
        combined = np.mean(np.stack(sims, axis=0), axis=0).astype(np.float32)
    return det_sim if det_sim.size else combined, combined


def _positive_hard_score(features: Sequence[plf.LLMCaseFeature], sim: np.ndarray, i: int, j: int, mode: str) -> float:
    base = float(sim[i, j]) if sim.size else 0.0
    if mode != "diverse":
        return base
    # Lower score is harder: low vector similarity and different surface failure descriptions.
    score = base
    if _same_nonempty(features, i, j, "primary_signature"):
        score += 0.30
    if _same_nonempty(features, i, j, "mismatch_type"):
        score += 0.15
    if _same_nonempty(features, i, j, "fatal_file"):
        score += 0.10
    if _same_nonempty(features, i, j, "register_name"):
        score += 0.05
    return score


def _negative_hard_score(features: Sequence[plf.LLMCaseFeature], sim: np.ndarray, i: int, j: int, mode: str) -> float:
    base = float(sim[i, j]) if sim.size else 0.0
    if mode != "confusable":
        return base
    # Higher score is harder: high semantic similarity and same structured failure hints.
    score = base
    if _same_nonempty(features, i, j, "primary_signature"):
        score += 0.45
    if _same_nonempty(features, i, j, "mismatch_type"):
        score += 0.25
    if _same_nonempty(features, i, j, "fatal_file"):
        score += 0.20
    if _same_nonempty(features, i, j, "register_name"):
        score += 0.10
    if _same_nonempty(features, i, j, "pc_region"):
        score += 0.10
    return score


def sample_pairs(
    features: list[plf.LLMCaseFeature],
    labels: Sequence[str],
    negative_ratio: float,
    hard_negative_ratio: float,
    hard_positive_ratio: float,
    max_train_pairs: int,
    random_state: int,
    positive_sampling: str = "det_low",
    negative_sampling: str = "det_high",
) -> tuple[list[tuple[int, int]], np.ndarray, dict]:
    rng = random.Random(random_state)
    np_rng = np.random.default_rng(random_state)

    by_bug: dict[str, list[int]] = defaultdict(list)
    for idx, bug in enumerate(labels):
        by_bug[str(bug)].append(idx)

    det_sim, combined_sim = _build_similarity_matrix(features)
    pos_sim = combined_sim if positive_sampling == "diverse" else det_sim
    neg_sim = combined_sim if negative_sampling == "confusable" else det_sim

    positives: list[tuple[int, int]] = []
    for indices in by_bug.values():
        for pos, i in enumerate(indices):
            for j in indices[pos + 1:]:
                positives.append((i, j))

    hard_positive_added = 0
    if positives and hard_positive_ratio > 0 and features:
        pos_set_base = {tuple(sorted(p)) for p in positives}
        hard_candidates = sorted(
            [
                (_positive_hard_score(features, pos_sim, i, j, positive_sampling), (i, j))
                for (i, j) in pos_set_base
            ],
            key=lambda x: x[0],
        )
        target_hard_pos = min(
            int(round(len(positives) * hard_positive_ratio)),
            max(0, max_train_pairs - len(positives)),
        )
        if hard_candidates and target_hard_pos > 0:
            extras = []
            while len(extras) < target_hard_pos:
                for _, pair in hard_candidates:
                    extras.append(pair)
                    if len(extras) >= target_hard_pos:
                        break
            positives.extend(extras)
            hard_positive_added = len(extras)

    pos_set = {tuple(sorted(p)) for p in positives}
    target_neg = min(
        int(round(len(positives) * negative_ratio)),
        max(0, max_train_pairs - len(positives)),
    )
    target_hard = int(round(target_neg * hard_negative_ratio))

    negatives: list[tuple[int, int]] = []
    neg_set: set[tuple[int, int]] = set()
    n = len(features)

    if target_hard > 0 and n > 1:
        hard_candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    continue
                pair = (i, j)
                if pair in pos_set:
                    continue
                hard_candidates.append((_negative_hard_score(features, neg_sim, i, j, negative_sampling), pair))
        hard_candidates.sort(key=lambda x: x[0], reverse=True)
        for _, pair in hard_candidates:
            if pair in neg_set:
                continue
            neg_set.add(pair)
            negatives.append(pair)
            if len(negatives) >= target_hard:
                break

    attempts = 0
    while len(negatives) < target_neg and attempts < target_neg * 100 + 1000:
        attempts += 1
        i = np_rng.integers(0, n)
        j = np_rng.integers(0, n)
        if i == j or labels[i] == labels[j]:
            continue
        pair = tuple(sorted((int(i), int(j))))
        if pair in neg_set or pair in pos_set:
            continue
        neg_set.add(pair)
        negatives.append(pair)

    pairs = positives + negatives
    y = np.asarray([1.0] * len(positives) + [0.0] * len(negatives), dtype=np.float32)

    if len(pairs) > max_train_pairs:
        pos_idx = np.flatnonzero(y == 1.0)
        neg_idx = np.flatnonzero(y == 0.0)
        keep_pos = min(len(pos_idx), max(1, max_train_pairs // (1 + max(1, int(negative_ratio)))))
        keep_neg = min(len(neg_idx), max_train_pairs - keep_pos)
        selected = np.concatenate(
            [
                np_rng.choice(pos_idx, size=keep_pos, replace=False),
                np_rng.choice(neg_idx, size=keep_neg, replace=False),
            ]
        )
        np_rng.shuffle(selected)
        pairs = [pairs[int(idx)] for idx in selected]
        y = y[selected]
    else:
        order = list(range(len(pairs)))
        rng.shuffle(order)
        pairs = [pairs[idx] for idx in order]
        y = y[order]

    stats = {
        "positive_pairs": int((y == 1.0).sum()),
        "negative_pairs": int((y == 0.0).sum()),
        "hard_positive_oversampled": hard_positive_added,
        "hard_negative_pairs": min(len(negatives), target_hard),
        "positive_sampling": positive_sampling,
        "negative_sampling": negative_sampling,
    }
    return pairs, y.astype(np.float32, copy=False), stats


def train(args: argparse.Namespace) -> dict:
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    t0 = time.perf_counter()

    # Resolve datasets
    datasets = [
        (Path(d) if Path(d).is_absolute() else (Path.cwd() / d).resolve())
        for d in args.datasets
    ]

    # Build half-split
    split_root = args.output_dir / "splits" / f"seed_{args.seed}"
    train_inputs: list[Path] = []
    train_golds: list[Path] = []
    val_parts: list[dict] = []
    for idx, ds in enumerate(datasets):
        splits = stratified_half_split(ds, args.seed, split_root)
        train_part = part_for_bit((args.combo >> idx) & 1)
        val_part = opposite_part(train_part)
        train_inputs.append(splits[train_part]["input"])
        train_golds.append(splits[train_part]["gold"])
        val_info = dict(splits[val_part])
        val_info["dataset"] = ds.name
        val_parts.append(val_info)

    # Build LLM-augmented features for training
    llm_args = plf._make_llm_args(
        llm_mode="embedding" if args.use_llm else "none",
        llm_doc_style=args.llm_doc_style,
        llm_cache_dir=args.llm_cache_dir,
        llm_dual=args.feature_mode in plf.DUAL_FEATURE_MODES,
    )
    all_features: list[plf.LLMCaseFeature] = []
    all_labels: list[str] = []
    all_features, _bundle = plf.build_llm_case_features_for_inputs(
        train_inputs, svd_dim=args.svd_dim, llm_args=llm_args
    )
    offset = 0
    for inp, gold in zip(train_inputs, train_golds):
        gold_labels = read_gold(gold)
        next_offset = offset + len(gold_labels)
        if next_offset > len(all_features):
            raise RuntimeError(f"feature/label mismatch in {inp}")
        all_labels.extend(gold_labels)
        offset = next_offset
    if len(all_features) != len(all_labels):
        raise RuntimeError(f"feature/label mismatch: {len(all_features)} vs {len(all_labels)}")

    llm_reducer = None
    llm_summary_reducer = None
    if args.feature_mode in ({"rich", "rich_no_det"} | plf.DUAL_FEATURE_MODES) and args.llm_reduce_dim > 0:
        llm_reducer = plf.fit_llm_reducer(
            all_features, args.llm_reduce_dim, random_state=args.random_state
        )
        if args.feature_mode in plf.DUAL_FEATURE_MODES:
            llm_summary_reducer = plf.fit_llm_summary_reducer(
                all_features, args.llm_reduce_dim, random_state=args.random_state
            )
        print(
            f"[features] llm_reduce_dim={args.llm_reduce_dim} "
            f"features_reducer={'none' if llm_reducer is None else type(llm_reducer).__name__} "
            f"summary_reducer={'none' if llm_summary_reducer is None else type(llm_summary_reducer).__name__}",
            file=sys.stderr,
        )

    # Sample pairs
    pairs, y, pair_stats = sample_pairs(
        all_features,
        all_labels,
        negative_ratio=args.negative_ratio,
        hard_negative_ratio=args.hard_negative_ratio,
        hard_positive_ratio=args.hard_positive_ratio,
        max_train_pairs=args.max_train_pairs,
        random_state=args.random_state,
        positive_sampling=args.positive_sampling,
        negative_sampling=args.negative_sampling,
    )
    print(
        f"[pairs] total={len(pairs)} pos={pair_stats['positive_pairs']} "
        f"neg={pair_stats['negative_pairs']} hard_pos_extra={pair_stats['hard_positive_oversampled']} "
        f"hard_neg={pair_stats['hard_negative_pairs']}",
        file=sys.stderr,
    )

    # Build pairwise feature matrix
    X = plf.build_rich_pair_feature_matrix(all_features, pairs, feature_mode=args.feature_mode)
    input_dim = X.shape[1]
    print(
        f"[features] mode={args.feature_mode} input_dim={input_dim} pairs={len(pairs)}",
        file=sys.stderr,
    )

    # Train selected backend
    train_time = time.perf_counter()
    if args.model_type == "logistic":
        model_pkg = plf.train_logistic_model(X, y, random_state=args.random_state)
    elif args.model_type == "gbdt":
        model_pkg = plf.train_gbdt_model(X, y, random_state=args.random_state)
    elif args.model_type == "mlp":
        model_pkg = plf.train_mlp_model(
            X, y,
            input_dim=input_dim,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            random_state=args.random_state,
            mlp_arch=args.mlp_arch,
            loss=args.loss,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
            early_stop_patience=args.early_stop_patience,
            layernorm=args.layernorm,
            batchnorm=args.batchnorm,
        )
    else:
        raise ValueError(f"unknown model_type: {args.model_type}")
    model_pkg.update({
        "feature_mode": args.feature_mode,
        "llm_reduce_dim": args.llm_reduce_dim if args.feature_mode in ({"rich", "rich_no_det"} | plf.DUAL_FEATURE_MODES) else 0,
        "llm_reducer": llm_reducer,
        "llm_summary_reducer": llm_summary_reducer,
        "svd_dim": args.svd_dim,
    })
    train_time = time.perf_counter() - train_time

    # Evaluate on each validation part
    val_results: list[dict] = []
    for part in val_parts:
        val_feats, _bundle = plf.build_llm_case_features(
            part["input"], svd_dim=args.svd_dim, llm_args=llm_args
        )
        prob = plf.predict_probability_matrix_sklearn(
            model_pkg, val_feats, batch_size=args.predict_batch_size
        )
        pred = plf.cluster_from_probability(prob, part["k"])
        gold = read_gold(part["gold"])
        ba, tpr, tnr = pairwise_scores(gold, [f"bucket_{l:03d}" for l in pred])
        val_results.append({
            "dataset": part["dataset"],
            "BA": ba, "TPR": tpr, "TNR": tnr,
            "num_cases": part["num_cases"], "k": part["k"],
        })
        print(
            f"[val] dataset={part['dataset']} BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f} "
            f"k={part['k']} cases={part['num_cases']}",
            file=sys.stderr,
        )

    mean_ba = float(np.mean([r["BA"] for r in val_results]))
    mean_tpr = float(np.mean([r["TPR"] for r in val_results]))
    mean_tnr = float(np.mean([r["TNR"] for r in val_results]))

    # Save model
    ext = "pt" if args.model_type == "mlp" else "pkl"
    model_tag = args.model_tag or args.model_type
    model_path = args.output_dir / f"model_seed{args.seed}_combo{args.combo:03b}_{model_tag}.{ext}"
    plf.save_model_pkg(model_pkg, model_path)

    total_time = time.perf_counter() - t0
    config = {
        "model_type": args.model_type,
        "model_tag": model_tag,
        "feature_mode": args.feature_mode,
        "mlp_arch": args.mlp_arch,
        "loss": args.loss,
        "llm_reduce_dim": args.llm_reduce_dim,
        "svd_dim": args.svd_dim,
        "use_llm": args.use_llm,
        "llm_doc_style": args.llm_doc_style,
        "positive_sampling": args.positive_sampling,
        "negative_sampling": args.negative_sampling,
        "input_dim": input_dim,
        "seed": args.seed,
        "combo": args.combo,
        "val_mean_BA": mean_ba,
        "val_mean_TPR": mean_tpr,
        "val_mean_TNR": mean_tnr,
        "val_details": val_results,
        "num_train_pairs": len(pairs),
        "pair_stats": pair_stats,
        "train_time_sec": train_time,
        "total_time_sec": total_time,
        "model_path": str(model_path),
    }
    config_path = args.output_dir / f"config_seed{args.seed}_combo{args.combo:03b}_{model_tag}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train pairwise LLM same-bug model.")
    p.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-type", choices=("logistic", "gbdt", "mlp"), default="logistic")
    p.add_argument("--model-tag", default="", help="optional filename tag for saved model/config")
    p.add_argument(
        "--feature-mode",
        choices=(
            "summary21", "rich", "rich_no_llm", "rich_no_det",
            "llm_dual", "llm_dual_struct", "llm_dual_struct_det_summary",
            "llm_dual_struct_det_summary_cross",
        ),
        default="summary21",
    )
    p.add_argument("--llm-reduce-dim", type=int, default=128)
    p.add_argument("--mlp-arch", choices=("shallow", "deep", "residual"), default="shallow")
    p.add_argument("--loss", choices=("bce", "focal"), default="bce")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", default="auto")
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--layernorm", action="store_true", default=True)
    p.add_argument("--no-layernorm", action="store_true", default=False)
    p.add_argument("--batchnorm", action="store_true", default=False)
    p.add_argument("--use-llm", action="store_true", default=True)
    p.add_argument("--no-llm", action="store_true", default=False)
    p.add_argument("--llm-doc-style", choices=("features", "summary"), default="features")
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dims", nargs="+", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--negative-ratio", type=float, default=2.0)
    p.add_argument("--hard-negative-ratio", type=float, default=0.5)
    p.add_argument("--hard-positive-ratio", type=float, default=0.5)
    p.add_argument("--positive-sampling", choices=("det_low", "diverse"), default="det_low")
    p.add_argument("--negative-sampling", choices=("det_high", "confusable"), default="det_high")
    p.add_argument("--max-train-pairs", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--combo", type=int, default=0)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--predict-batch-size", type=int, default=100000)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_llm:
        args.use_llm = False
    if args.no_layernorm:
        args.layernorm = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        config = train(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(config, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
