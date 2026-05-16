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


def sample_pairs(
    features: list[plf.LLMCaseFeature],
    labels: Sequence[str],
    negative_ratio: float,
    hard_negative_ratio: float,
    hard_positive_ratio: float,
    max_train_pairs: int,
    random_state: int,
) -> tuple[list[tuple[int, int]], np.ndarray, dict]:
    rng = random.Random(random_state)
    np_rng = np.random.default_rng(random_state)

    by_bug: dict[str, list[int]] = defaultdict(list)
    for idx, bug in enumerate(labels):
        by_bug[str(bug)].append(idx)

    # Build det cosine matrix for hard mining
    if features:
        det_mat = np.vstack([f.det_vec for f in features]).astype(np.float32)
        norms = np.linalg.norm(det_mat, axis=1, keepdims=True)
        det_mat = det_mat / np.maximum(norms, 1e-12)
        det_sim = det_mat @ det_mat.T
    else:
        det_sim = np.zeros((0, 0), dtype=np.float32)

    positives: list[tuple[int, int]] = []
    for indices in by_bug.values():
        for pos, i in enumerate(indices):
            for j in indices[pos + 1:]:
                positives.append((i, j))

    hard_positive_added = 0
    if positives and hard_positive_ratio > 0 and features:
        pos_set = {tuple(sorted(p)) for p in positives}
        hard_candidates = sorted(
            [(float(det_sim[i, j]), (i, j)) for (i, j) in pos_set if det_sim.size],
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

    # Hard negatives: high det cosine but different bugs
    if target_hard > 0 and n > 1 and det_sim.size:
        sim_copy = det_sim.copy()
        np.fill_diagonal(sim_copy, -np.inf)
        top_k = min(16, n - 1)
        for i in range(n):
            cands = np.argpartition(-sim_copy[i], min(top_k, n - 1))[:top_k]
            cands = sorted(cands, key=lambda j: sim_copy[i, j], reverse=True)
            for j in cands:
                j = int(j)
                if labels[i] == labels[j]:
                    continue
                pair = tuple(sorted((i, j)))
                if pair in neg_set or pair in pos_set:
                    continue
                neg_set.add(pair)
                negatives.append(pair)
                if len(negatives) >= target_hard:
                    break
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
    )
    all_features: list[plf.LLMCaseFeature] = []
    all_labels: list[str] = []
    for inp, gold in zip(train_inputs, train_golds):
        feats, _bundle = plf.build_llm_case_features(inp, svd_dim=args.svd_dim, llm_args=llm_args)
        gold_labels = read_gold(gold)
        if len(feats) != len(gold_labels):
            raise RuntimeError(f"feature/label mismatch: {len(feats)} vs {len(gold_labels)} in {inp}")
        all_features.extend(feats)
        all_labels.extend(gold_labels)

    # Sample pairs
    pairs, y, pair_stats = sample_pairs(
        all_features,
        all_labels,
        negative_ratio=args.negative_ratio,
        hard_negative_ratio=args.hard_negative_ratio,
        hard_positive_ratio=args.hard_positive_ratio,
        max_train_pairs=args.max_train_pairs,
        random_state=args.random_state,
    )
    print(
        f"[pairs] total={len(pairs)} pos={pair_stats['positive_pairs']} "
        f"neg={pair_stats['negative_pairs']} hard_pos_extra={pair_stats['hard_positive_oversampled']} "
        f"hard_neg={pair_stats['hard_negative_pairs']}",
        file=sys.stderr,
    )

    # Build pairwise feature matrix
    X = plf.build_llm_pair_feature_matrix(all_features, pairs)
    input_dim = X.shape[1]
    print(f"[features] input_dim={input_dim} pairs={len(pairs)}", file=sys.stderr)

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
        )
    else:
        raise ValueError(f"unknown model_type: {args.model_type}")
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
    model_path = args.output_dir / f"model_seed{args.seed}_combo{args.combo:03b}_{args.model_type}.{ext}"
    plf.save_model_pkg(model_pkg, model_path)

    total_time = time.perf_counter() - t0
    config = {
        "model_type": args.model_type,
        "svd_dim": args.svd_dim,
        "use_llm": args.use_llm,
        "llm_doc_style": args.llm_doc_style,
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
    config_path = args.output_dir / f"config_seed{args.seed}_combo{args.combo:03b}_{args.model_type}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train pairwise LLM same-bug model.")
    p.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-type", choices=("logistic", "gbdt", "mlp"), default="logistic")
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
    p.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 64])
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--negative-ratio", type=float, default=2.0)
    p.add_argument("--hard-negative-ratio", type=float, default=0.5)
    p.add_argument("--hard-positive-ratio", type=float, default=0.5)
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
