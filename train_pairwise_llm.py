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


def build_connectivity_positive_pairs(
    features: Sequence[plf.LLMCaseFeature],
    labels: Sequence[str],
    positive_sampling: str = "diverse",
) -> list[tuple[int, int]]:
    """Build per-bug positive backbones plus cross-view bridge edges.

    The maximum-similarity tree gives every case a learnable route to its bug
    component.  One low-similarity bridge per case then exposes surface-form
    changes that ordinary random positive sampling tends to miss.  Pairs are
    always local to the supplied episode.
    """
    by_bug: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_bug[str(label)].append(index)
    det_sim, combined_sim = _build_similarity_matrix(features)
    similarity = combined_sim if positive_sampling == "diverse" else det_sim
    grouped_pairs: list[list[tuple[int, int]]] = []
    for indices in by_bug.values():
        if len(indices) < 2:
            continue
        local: list[tuple[int, int]] = []
        local_matrix = similarity[np.ix_(indices, indices)] if similarity.size else None
        if local_matrix is None:
            center_position = 0
        else:
            center_position = int(np.argmax(np.mean(local_matrix, axis=1)))
        visited = {indices[center_position]}
        remaining = set(indices) - visited
        while remaining:
            best = max(
                (
                    float(similarity[left, right]) if similarity.size else 0.0,
                    -right,
                    left,
                    right,
                )
                for left in visited
                for right in remaining
            )
            left, right = int(best[2]), int(best[3])
            local.append(tuple(sorted((left, right))))
            visited.add(right)
            remaining.remove(right)
        # The hard bridge is deliberately different from the easy tree: it
        # teaches invariance between distinct surface manifestations.
        for left in indices:
            candidates = [right for right in indices if right != left]
            right = min(
                candidates,
                key=lambda value: (
                    float(similarity[left, value]) if similarity.size else 0.0,
                    value,
                ),
            )
            local.append(tuple(sorted((left, right))))
        grouped_pairs.append(list(dict.fromkeys(local)))

    # Round-robin across bugs so a large bug cannot consume the reserve before
    # small bugs receive any connectivity edge.
    output: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    position = 0
    while True:
        added = False
        for group in grouped_pairs:
            if position >= len(group):
                continue
            pair = group[position]
            if pair not in seen:
                seen.add(pair)
                output.append(pair)
            added = True
        if not added:
            break
        position += 1
    return output


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
    connectivity_positive_fraction: float = 0.0,
) -> tuple[list[tuple[int, int]], np.ndarray, dict]:
    rng = random.Random(random_state)
    np_rng = np.random.default_rng(random_state)

    by_bug: dict[str, list[int]] = defaultdict(list)
    for idx, bug in enumerate(labels):
        by_bug[str(bug)].append(idx)

    det_sim, combined_sim = _build_similarity_matrix(features)
    pos_sim = combined_sim if positive_sampling == "diverse" else det_sim
    neg_sim = combined_sim if negative_sampling == "confusable" else det_sim

    unique_positives: list[tuple[int, int]] = []
    for indices in by_bug.values():
        for pos, i in enumerate(indices):
            for j in indices[pos + 1:]:
                unique_positives.append((i, j))

    hard_positive_added = 0
    hard_positive_selected = 0
    ratio = max(0.0, float(negative_ratio))
    positive_budget = max_train_pairs if ratio <= 0.0 else max(1, int(max_train_pairs // (1.0 + ratio)))
    hard_candidates = sorted(
        [
            (_positive_hard_score(features, pos_sim, i, j, positive_sampling), (i, j))
            for i, j in unique_positives
        ],
        key=lambda item: item[0],
    )
    hard_fraction = (
        max(0.0, float(hard_positive_ratio)) / (1.0 + max(0.0, float(hard_positive_ratio)))
        if hard_positive_ratio > 0
        else 0.0
    )
    if len(unique_positives) > positive_budget:
        connectivity_candidates = build_connectivity_positive_pairs(
            features, labels, positive_sampling=positive_sampling
        )
        connectivity_keep = min(
            len(connectivity_candidates),
            int(round(positive_budget * max(0.0, min(1.0, connectivity_positive_fraction)))),
        )
        connectivity_pairs = connectivity_candidates[:connectivity_keep]
        reserved = set(connectivity_pairs)
        hard_keep = min(
            positive_budget - len(connectivity_pairs),
            int(round(positive_budget * hard_fraction)),
        )
        hard_pairs = [
            pair for _, pair in hard_candidates if pair not in reserved
        ][:hard_keep]
        reserved.update(hard_pairs)
        remaining = [pair for pair in unique_positives if pair not in reserved]
        random_keep = positive_budget - len(connectivity_pairs) - len(hard_pairs)
        random_pairs = rng.sample(remaining, random_keep) if random_keep < len(remaining) else remaining
        positives = connectivity_pairs + hard_pairs + random_pairs
        hard_positive_selected = len(hard_pairs)
        connectivity_selected = len(connectivity_pairs)
    else:
        positives = list(unique_positives)
        extra_budget = max(0, positive_budget - len(positives))
        target_extra = min(int(round(len(unique_positives) * max(0.0, hard_positive_ratio))), extra_budget)
        if hard_candidates and target_extra > 0:
            extras: list[tuple[int, int]] = []
            while len(extras) < target_extra:
                for _, pair in hard_candidates:
                    extras.append(pair)
                    if len(extras) >= target_extra:
                        break
            positives.extend(extras)
            hard_positive_added = len(extras)
            hard_positive_selected = len(extras)
        connectivity_selected = min(
            len(build_connectivity_positive_pairs(features, labels, positive_sampling)),
            len(unique_positives),
        ) if connectivity_positive_fraction > 0 else 0

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
        "hard_positive_selected": hard_positive_selected,
        "connectivity_positive_selected": connectivity_selected,
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

    # Encode trace files if trace encoder is provided
    trace_encoder = None
    if args.trace_encoder is not None and args.feature_mode == "llm_dual_struct_det_summary_trace":
        from trace_transformer_pretrain import load_pretrained
        trace_encoder = load_pretrained(args.trace_encoder, device=args.device)
        # Collect trace paths from all training inputs into one lookup
        trace_by_case: dict[str, tuple[Path | None, str]] = {}
        for inp in train_inputs:
            from trace_sequence import collect_trace_paths_from_input
            for case_id, path, status in collect_trace_paths_from_input(inp):
                trace_by_case[case_id] = (path, status)
        encoded_count = 0
        for feat in all_features:
            path, status = trace_by_case.get(feat.case_id, (None, "missing"))
            if status == "ok" and path is not None:
                feat.trace_vec = trace_encoder.encode_trace_tail(
                    str(path), tail_lines=args.trace_window_size,
                )
                encoded_count += 1
        print(f"[trace] encoder loaded, traces encoded for {encoded_count}/{len(all_features)} cases",
              file=sys.stderr)
        plf.normalize_trace_vectors(all_features)

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

    trace_reducer = None
    if args.trace_encoder is not None and args.feature_mode == "llm_dual_struct_det_summary_trace":
        if args.trace_reduce_dim > 0:
            trace_reducer = plf.fit_trace_reducer(
                all_features, args.trace_reduce_dim, random_state=args.random_state,
            )
        print(
            f"[features] trace_reduce_dim={args.trace_reduce_dim} "
            f"trace_reducer={'none' if trace_reducer is None else type(trace_reducer).__name__}",
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

    # Prepare the exact held-out half-split clustering episodes once. All
    # neural architectures use these same matrices for early stopping by BA.
    validation_clusters = []
    if args.model_type == "mlp" and args.feature_mode != "llm_dual_struct_det_summary_trace":
        for part in val_parts:
            val_feats, _ = plf.build_llm_case_features(
                part["input"], svd_dim=args.svd_dim, llm_args=llm_args
            )
            if args.llm_reduce_dim > 0:
                plf.apply_llm_reducer(val_feats, llm_reducer, args.llm_reduce_dim)
                if args.feature_mode in plf.DUAL_FEATURE_MODES:
                    plf.apply_llm_summary_reducer(val_feats, llm_summary_reducer, args.llm_reduce_dim)
            val_pairs = [(i, j) for i in range(len(val_feats)) for j in range(i + 1, len(val_feats))]
            validation_clusters.append({
                "dataset": part["dataset"],
                "X": plf.build_rich_pair_feature_matrix(val_feats, val_pairs, feature_mode=args.feature_mode),
                "pairs": val_pairs,
                "n": len(val_feats),
                "k": part["k"],
                "gold": read_gold(part["gold"]),
            })

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
            model_arch=args.model_arch,
            gate_reg=args.gate_reg,
            ft_d_token=args.ft_d_token,
            ft_layers=args.ft_layers,
            ft_heads=args.ft_heads,
            ft_dropout=args.ft_dropout,
            ft_attention_dropout=args.ft_attention_dropout,
            ft_ffn_mult=args.ft_ffn_mult,
            ft_max_tokens=args.ft_max_tokens,
            validation_clusters=validation_clusters,
        )
    else:
        raise ValueError(f"unknown model_type: {args.model_type}")
    model_pkg.update({
        "feature_mode": args.feature_mode,
        "feature_schema_version": 1,
        "llm_reduce_dim": args.llm_reduce_dim if args.feature_mode in ({"rich", "rich_no_det"} | plf.DUAL_FEATURE_MODES) else 0,
        "llm_reducer": llm_reducer,
        "llm_summary_reducer": llm_summary_reducer,
        "trace_reduce_dim": args.trace_reduce_dim if args.feature_mode == "llm_dual_struct_det_summary_trace" else 0,
        "trace_reducer": trace_reducer,
        "trace_encoder_dir": str(args.trace_encoder) if args.trace_encoder is not None else "",
        "svd_dim": args.svd_dim,
    })
    train_time = time.perf_counter() - train_time

    # Evaluate on each validation part
    val_results: list[dict] = []
    for part in val_parts:
        val_feats, _bundle = plf.build_llm_case_features(
            part["input"], svd_dim=args.svd_dim, llm_args=llm_args
        )
        # Encode traces for val features if trace mode
        if args.feature_mode == "llm_dual_struct_det_summary_trace" and trace_encoder is not None:
            from trace_sequence import collect_trace_paths_from_input
            trace_by_case = {}
            for case_id, path, status in collect_trace_paths_from_input(part["input"]):
                trace_by_case[case_id] = (path, status)
            for feat in val_feats:
                path, status = trace_by_case.get(feat.case_id, (None, "missing"))
                if status == "ok" and path is not None:
                    feat.trace_vec = trace_encoder.encode_trace_tail(
                        str(path), tail_lines=args.trace_window_size,
                    )
            plf.normalize_trace_vectors(val_feats)
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
        "model_arch": args.model_arch if args.model_type == "mlp" else "",
        "model_config": {
            "gate_reg": args.gate_reg,
            "ft_d_token": args.ft_d_token,
            "ft_layers": args.ft_layers,
            "ft_heads": args.ft_heads,
            "ft_dropout": args.ft_dropout,
            "ft_attention_dropout": args.ft_attention_dropout,
            "ft_ffn_mult": args.ft_ffn_mult,
            "ft_max_tokens": args.ft_max_tokens,
        } if args.model_type == "mlp" else {},
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
        "trace_reduce_dim": args.trace_reduce_dim if args.feature_mode == "llm_dual_struct_det_summary_trace" else 0,
        "trace_encoder_dir": str(args.trace_encoder) if args.trace_encoder is not None else "",
        "best_epoch": int(model_pkg.get("best_epoch", 0)),
        "best_val_BA": float(model_pkg.get("best_val_BA", -1.0)),
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
            "llm_dual_struct_det_summary_trace",
        ),
        default="summary21",
    )
    p.add_argument("--llm-reduce-dim", type=int, default=128)
    p.add_argument("--mlp-arch", choices=("shallow", "deep", "residual"), default="shallow", help="legacy residual MLP shape selector")
    p.add_argument(
        "--model-arch",
        choices=("auto", "res_mlp", "gated_mlp", "ft_transformer"),
        default="auto",
        help="auto preserves the legacy --mlp-arch behavior",
    )
    p.add_argument("--gate-reg", type=float, default=1e-4)
    p.add_argument("--ft-d-token", type=int, default=64)
    p.add_argument("--ft-layers", type=int, default=2)
    p.add_argument("--ft-heads", type=int, default=4)
    p.add_argument("--ft-dropout", type=float, default=0.1)
    p.add_argument("--ft-attention-dropout", type=float, default=0.1)
    p.add_argument("--ft-ffn-mult", type=int, default=2)
    p.add_argument("--ft-max-tokens", type=int, default=0)
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
    p.add_argument("--trace-encoder", type=Path, default=None,
                   help="path to pretrained trace encoder directory")
    p.add_argument("--trace-reduce-dim", type=int, default=32,
                   help="SVD reduction dim for trace embeddings (0 to disable)")
    p.add_argument("--trace-window-mode", choices=("tail", "random"), default="tail")
    p.add_argument("--trace-window-size", type=int, default=500)
    p.add_argument("--trace-max-seq-len", type=int, default=1024)
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
