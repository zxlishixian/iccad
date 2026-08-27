#!/usr/bin/env python3
"""Train the experimental Pairwise Same-Bug MLP.

Training reads local gold.csv files. The trained model is optional and is only
used when regr_fail_bucketing.py is explicitly run with --cluster pairwise_mlp.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import pairwise_features as pf
from run_experiments import pairwise_scores, read_gold
from run_half_split_experiments import DEFAULT_DATASETS, opposite_part, part_for_bit, stratified_half_split


def resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def cluster_from_probability(prob: np.ndarray, k: int) -> list[int]:
    from sklearn.cluster import AgglomerativeClustering

    n = prob.shape[0]
    if n == 0:
        return []
    k = max(1, min(k, n))
    if k == n:
        return list(range(n))
    distance = 1.0 - prob
    np.fill_diagonal(distance, 0.0)
    try:
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
    return model.fit_predict(distance).tolist()


def labels_for_inputs(input_paths: Sequence[Path], gold_paths: Sequence[Path]) -> list[str]:
    labels: list[str] = []
    for input_path, gold_path in zip(input_paths, gold_paths):
        del input_path
        labels.extend(read_gold(gold_path))
    return labels


def sample_pairs(
    features: list[pf.CaseFeature],
    labels: Sequence[str],
    negative_ratio: float,
    hard_negative_ratio: float,
    hard_positive_ratio: float,
    max_positive_pairs: int,
    max_positive_pairs_per_bug: int,
    max_train_pairs: int,
    random_state: int,
) -> tuple[list[tuple[int, int]], np.ndarray, dict]:
    rng = random.Random(random_state)
    np_rng = np.random.default_rng(random_state)
    by_bug: dict[str, list[int]] = defaultdict(list)
    for idx, bug in enumerate(labels):
        by_bug[str(bug)].append(idx)

    dense = np.vstack([feature.dense_vec for feature in features]).astype(np.float32) if features else np.zeros((0, 1), dtype=np.float32)
    if len(dense):
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        dense = dense / np.maximum(norms, 1e-12)
        sim = dense @ dense.T
    else:
        sim = np.zeros((0, 0), dtype=np.float32)

    positives: list[tuple[int, int]] = []
    hard_positive_pool: list[tuple[float, tuple[int, int]]] = []
    positive_pairs_by_bug: dict[str, int] = {}
    for indices in by_bug.values():
        bug_pairs: list[tuple[int, int]] = []
        for pos, i in enumerate(indices):
            for j in indices[pos + 1 :]:
                bug_pairs.append((i, j))
                hard_positive_pool.append((float(sim[i, j]) if sim.size else 0.0, (i, j)))
        if max_positive_pairs_per_bug > 0 and len(bug_pairs) > max_positive_pairs_per_bug:
            bug_pairs = rng.sample(bug_pairs, max_positive_pairs_per_bug)
        positives.extend(bug_pairs)
        if indices:
            positive_pairs_by_bug[str(labels[indices[0]])] = len(bug_pairs)
    if len(positives) > max_positive_pairs:
        positives = rng.sample(positives, max_positive_pairs)

    # Oversample hard positives: same-bug pairs that are far apart in dense
    # text space. These are exactly the pairs that low-TPR models tend to miss.
    hard_positive_added = 0
    if positives and hard_positive_ratio > 0:
        pos_set_for_hard = {tuple(sorted(pair)) for pair in positives}
        hard_positive_pool.sort(key=lambda item: item[0])
        target_hard_pos = min(
            int(round(len(positives) * hard_positive_ratio)),
            max(0, max_train_pairs - len(positives)),
        )
        hard_candidates = [pair for _, pair in hard_positive_pool if tuple(sorted(pair)) in pos_set_for_hard]
        if hard_candidates and target_hard_pos > 0:
            repeats = []
            while len(repeats) < target_hard_pos:
                for pair in hard_candidates:
                    repeats.append(pair)
                    if len(repeats) >= target_hard_pos:
                        break
            positives.extend(repeats)
            hard_positive_added = len(repeats)

    pos_set = {tuple(sorted(pair)) for pair in positives}
    target_neg = min(int(round(len(positives) * negative_ratio)), max(1, max_train_pairs - len(positives)))
    target_hard = int(round(target_neg * hard_negative_ratio))
    negatives: list[tuple[int, int]] = []
    neg_set: set[tuple[int, int]] = set()

    if target_hard > 0 and features:
        sim_for_neg = sim.copy()
        np.fill_diagonal(sim_for_neg, -np.inf)
        top_k = min(32, max(1, len(features) - 1))
        for i in range(len(features)):
            candidates = np.argpartition(-sim_for_neg[i], min(top_k, len(features) - 1))[:top_k]
            candidates = sorted(candidates, key=lambda j: sim_for_neg[i, j], reverse=True)
            for j in candidates:
                if labels[i] == labels[j]:
                    continue
                pair = tuple(sorted((i, int(j))))
                if pair in neg_set or pair in pos_set:
                    continue
                neg_set.add(pair)
                negatives.append(pair)
                if len(negatives) >= target_hard:
                    break
            if len(negatives) >= target_hard:
                break

    n = len(features)
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
        "positive_pairs_selected": int((y == 1.0).sum()),
        "negative_pairs_selected": int((y == 0.0).sum()),
        "hard_positive_oversampled": hard_positive_added,
        "hard_negative_pairs": min(len(negatives), target_hard),
        "positive_pairs_by_bug": positive_pairs_by_bug,
    }
    return pairs, y.astype(np.float32, copy=False), stats


def evaluate_validation_parts(model, val_parts: Sequence[dict], args: argparse.Namespace, device: str) -> dict:
    bas: list[float] = []
    tprs: list[float] = []
    tnrs: list[float] = []
    details: list[dict] = []
    for part in val_parts:
        features, _ = pf.build_case_features(part["input"], parser="drain", svd_dim=args.svd_dim)
        prob = pf.predict_probability_matrix(
            model,
            features,
            device=device,
            batch_size=args.predict_batch_size,
            prob_bias=args.prob_bias,
            prob_temperature=args.prob_temperature,
        )
        prob = pf.calibrate_probability_matrix(
            prob,
            features,
            primary_floor=args.pairwise_primary_floor,
            op_pair_floor=args.pairwise_op_pair_floor,
            mismatch_floor=args.pairwise_mismatch_floor,
            conflict_penalty=args.pairwise_conflict_penalty,
            cosine_gate=args.pairwise_mismatch_cosine_gate,
        )
        pred = cluster_from_probability(prob, part["k"])
        gold = read_gold(part["gold"])
        ba, tpr, tnr = pairwise_scores(gold, [f"bucket_{label:03d}" for label in pred])
        bas.append(ba)
        tprs.append(tpr)
        tnrs.append(tnr)
        details.append({"dataset": part["dataset"], "BA": ba, "TPR": tpr, "TNR": tnr})
    return {
        "BA": float(np.mean(bas)) if bas else 0.0,
        "TPR": float(np.mean(tprs)) if tprs else 0.0,
        "TNR": float(np.mean(tnrs)) if tnrs else 0.0,
        "details": details,
    }


def build_splits(args: argparse.Namespace) -> tuple[list[Path], list[Path], list[dict]]:
    seed = args.seeds[0]
    split_root = args.output.parent / "pairwise_mlp_splits" / f"seed_{seed}"
    train_inputs: list[Path] = []
    train_golds: list[Path] = []
    val_parts: list[dict] = []
    for idx, dataset in enumerate(args.datasets):
        dataset = Path(dataset)
        splits = stratified_half_split(dataset, seed, split_root)
        train_part = part_for_bit((args.combo >> idx) & 1)
        val_part = opposite_part(train_part)
        train_inputs.append(splits[train_part]["input"])
        train_golds.append(splits[train_part]["gold"])
        val_info = dict(splits[val_part])
        val_info["dataset"] = dataset.name
        val_info["val_part"] = val_part
        val_parts.append(val_info)
    return train_inputs, train_golds, val_parts


def train(args: argparse.Namespace) -> dict:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("pairwise MLP training requires PyTorch; install torch to use train_pairwise_mlp.py") from exc

    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    device = resolve_device(args.device)

    train_inputs, train_golds, val_parts = build_splits(args)
    train_features, _ = pf.build_case_features_for_inputs(train_inputs, parser="drain", svd_dim=args.svd_dim)
    train_labels = labels_for_inputs(train_inputs, train_golds)
    pairs, y, pair_stats = sample_pairs(
        train_features,
        train_labels,
        negative_ratio=args.negative_ratio,
        hard_negative_ratio=args.hard_negative_ratio,
        hard_positive_ratio=args.hard_positive_ratio,
        max_positive_pairs=args.max_positive_pairs,
        max_positive_pairs_per_bug=args.max_positive_pairs_per_bug,
        max_train_pairs=args.max_train_pairs,
        random_state=args.random_state,
    )
    print(
        "pair_sampling "
        f"pairs={len(pairs)} pos={int((y == 1.0).sum())} neg={int((y == 0.0).sum())} "
        f"hard_pos_extra={pair_stats['hard_positive_oversampled']} "
        f"hard_neg={pair_stats['hard_negative_pairs']}",
        file=sys.stderr,
    )
    X = pf.build_pair_feature_matrix(train_features, pairs)
    y_tensor = torch.from_numpy(y)
    X_tensor = torch.from_numpy(X)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    input_dim = X.shape[1]
    model = pf.build_pairwise_mlp_model(
        input_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        architecture=args.architecture,
    ).to(device)
    pos = float((y == 1.0).sum())
    neg = float((y == 0.0).sum())
    pos_weight = torch.tensor([args.pos_weight_scale * neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {"BA": -1.0, "TPR": 0.0, "TNR": 0.0, "epoch": 0, "state_dict": None, "details": []}
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        if epoch % args.eval_every != 0:
            print(f"epoch={epoch} train_loss={np.mean(losses):.6f}", file=sys.stderr)
            continue

        metrics = evaluate_validation_parts(model, val_parts, args, device)
        improved = metrics["BA"] > best["BA"]
        print(
            f"epoch={epoch} train_loss={np.mean(losses):.6f} "
            f"val_BA={metrics['BA']:.6f} val_TPR={metrics['TPR']:.6f} val_TNR={metrics['TNR']:.6f}",
            file=sys.stderr,
        )
        if improved:
            best = {
                "BA": metrics["BA"],
                "TPR": metrics["TPR"],
                "TNR": metrics["TNR"],
                "epoch": epoch,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "details": metrics["details"],
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.early_stop_patience:
                print(f"early stopping at epoch={epoch}", file=sys.stderr)
                break

    state_dict = best["state_dict"] or {key: value.detach().cpu() for key, value in model.state_dict().items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state_dict,
            "input_dim": input_dim,
            "hidden_dims": list(args.hidden_dims),
            "dropout": args.dropout,
            "architecture": args.architecture,
            "svd_dim": args.svd_dim,
            "feature_schema_version": pf.FEATURE_SCHEMA_VERSION,
        },
        args.output,
    )
    config = {
        "model_type": "pairwise_mlp",
        "svd_dim": args.svd_dim,
        "hidden_dims": list(args.hidden_dims),
        "dropout": args.dropout,
        "architecture": args.architecture,
        "feature_schema_version": pf.FEATURE_SCHEMA_VERSION,
        "training_datasets": [str(Path(path)) for path in args.datasets],
        "validation_mode": args.validation_mode,
        "seed": args.seeds[0],
        "combo": args.combo,
        "best_epoch": best["epoch"],
        "best_val_BA": best["BA"],
        "best_val_TPR": best["TPR"],
        "best_val_TNR": best["TNR"],
        "validation_details": best["details"],
        "num_train_pairs": len(pairs),
        "num_positive_pairs": int((y == 1.0).sum()),
        "num_negative_pairs": int((y == 0.0).sum()),
        "pair_sampling": pair_stats,
        "pos_weight_scale": args.pos_weight_scale,
        "pairwise_primary_floor": args.pairwise_primary_floor,
        "pairwise_op_pair_floor": args.pairwise_op_pair_floor,
        "pairwise_mismatch_floor": args.pairwise_mismatch_floor,
        "pairwise_conflict_penalty": args.pairwise_conflict_penalty,
    }
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train experimental Pairwise Same-Bug MLP.")
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--svd-dim", type=int, default=128)
    parser.add_argument("--architecture", choices=("plain", "layernorm", "residual"), default="residual")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 512, 256, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--negative-ratio", type=float, default=1.5)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.5)
    parser.add_argument("--hard-positive-ratio", type=float, default=0.5)
    parser.add_argument("--max-positive-pairs", type=int, default=50000)
    parser.add_argument("--max-positive-pairs-per-bug", type=int, default=4000)
    parser.add_argument("--max-train-pairs", type=int, default=300000)
    parser.add_argument("--pos-weight-scale", type=float, default=1.2)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--validation-mode", choices=("half_split",), default="half_split")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--combo", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--predict-batch-size", type=int, default=100000)
    parser.add_argument("--prob-bias", type=float, default=0.0)
    parser.add_argument("--prob-temperature", type=float, default=1.0)
    parser.add_argument("--pairwise-primary-floor", type=float, default=0.70)
    parser.add_argument("--pairwise-op-pair-floor", type=float, default=0.65)
    parser.add_argument("--pairwise-mismatch-floor", type=float, default=0.55)
    parser.add_argument("--pairwise-conflict-penalty", type=float, default=0.05)
    parser.add_argument("--pairwise-mismatch-cosine-gate", type=float, default=0.20)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = train(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(config, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
