#!/usr/bin/env python3
"""Unified multi-dataset pairwise training experiments.

Experimental only. Every dataset is treated as an independent clustering
episode: pairs and auxiliary constraints are built only within a dataset.
Gold/golden labels are used by this runner for training and evaluation. The
formal ``regr_fail_bucketing.py`` predictor is not modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import official_style_features as osf
import pairwise_llm_features as plf
import train_pairwise_llm as tpl
from run_experiments import pairwise_scores, read_gold
from run_official_full_retrain_experiments import write_csv, write_pred


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    Path("old_fake_dataset/first_batch_dataset"),
    Path("old_fake_dataset/stage2_dataset_working"),
    Path("old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("official_format_fake_dataset/stable_official_like_multitest_v1"),
    Path("test_case/problem/benchmark_set_1"),
    Path("test_case/problem/benchmark_set_2"),
]


@dataclass
class DatasetSlice:
    name: str
    path: Path
    start: int
    stop: int
    labels: list[str]
    cases: list[str]


@dataclass
class PairData:
    pairs: list[tuple[int, int]]
    labels: np.ndarray
    dataset_ids: np.ndarray
    dataset_names: list[str]
    triplets: np.ndarray
    triangles: np.ndarray
    hard_positive_mask: np.ndarray
    connectivity_groups: list[np.ndarray]
    prototype_groups: list[tuple[np.ndarray, np.ndarray]]
    stats: list[dict]


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_slices(datasets: Sequence[Path]) -> list[DatasetSlice]:
    slices: list[DatasetSlice] = []
    offset = 0
    for dataset in datasets:
        labels = read_gold(osf.gold_path(dataset))
        cases = osf.read_cases(dataset / "input.csv")
        if len(labels) != len(cases):
            raise RuntimeError(
                f"case/label mismatch for {dataset}: cases={len(cases)} labels={len(labels)}"
            )
        slices.append(DatasetSlice(dataset.name, dataset, offset, offset + len(labels), labels, cases))
        offset += len(labels)
    return slices


def _pair_key(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def _build_auxiliary_indices(
    local_pairs: Sequence[tuple[int, int]],
    labels: Sequence[str],
    max_items: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pair-row triplets (positive, negative) and triangle row triples."""
    rng = random.Random(seed)
    pair_to_row = {_pair_key(i, j): row for row, (i, j) in enumerate(local_pairs)}
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[str(label)].append(idx)

    triplets: list[tuple[int, int]] = []
    all_indices = list(range(len(labels)))
    for anchor in all_indices:
        positives = [x for x in by_label[str(labels[anchor])] if x != anchor]
        negatives = [x for x in all_indices if labels[x] != labels[anchor]]
        rng.shuffle(positives)
        rng.shuffle(negatives)
        for pos, neg in zip(positives[:8], negatives[:8]):
            pos_row = pair_to_row.get(_pair_key(anchor, pos))
            neg_row = pair_to_row.get(_pair_key(anchor, neg))
            if pos_row is not None and neg_row is not None:
                triplets.append((pos_row, neg_row))

    triangles: list[tuple[int, int, int]] = []
    attempts = 0
    target = max_items
    while len(triangles) < target and attempts < target * 50 + 1000 and len(labels) >= 3:
        attempts += 1
        i, j, k = rng.sample(all_indices, 3)
        rows = (
            pair_to_row.get(_pair_key(i, j)),
            pair_to_row.get(_pair_key(j, k)),
            pair_to_row.get(_pair_key(i, k)),
        )
        if all(row is not None for row in rows):
            triangles.append((int(rows[0]), int(rows[1]), int(rows[2])))

    rng.shuffle(triplets)
    rng.shuffle(triangles)
    return (
        np.asarray(triplets[:max_items], dtype=np.int64).reshape(-1, 2),
        np.asarray(triangles[:max_items], dtype=np.int64).reshape(-1, 3),
    )


def build_pair_data(
    features: list[plf.LLMCaseFeature],
    slices: Sequence[DatasetSlice],
    args: argparse.Namespace,
    seed: int,
) -> PairData:
    all_pairs: list[tuple[int, int]] = []
    all_y: list[np.ndarray] = []
    all_ds: list[np.ndarray] = []
    all_triplets: list[np.ndarray] = []
    all_triangles: list[np.ndarray] = []
    all_hard_positive: list[np.ndarray] = []
    connectivity_groups: list[np.ndarray] = []
    prototype_groups: list[tuple[np.ndarray, np.ndarray]] = []
    stats: list[dict] = []
    pair_offset = 0

    for dataset_id, ds in enumerate(slices):
        local_features = features[ds.start:ds.stop]
        local_pairs, y, pair_stats = tpl.sample_pairs(
            local_features,
            ds.labels,
            negative_ratio=args.negative_ratio,
            hard_negative_ratio=args.hard_negative_ratio,
            hard_positive_ratio=args.hard_positive_ratio,
            max_train_pairs=args.max_pairs_per_dataset,
            random_state=seed * 1009 + dataset_id * 97 + 17,
            positive_sampling="diverse",
            negative_sampling="confusable",
        )
        triplets, triangles = _build_auxiliary_indices(
            local_pairs, ds.labels, args.max_aux_per_dataset, seed * 2017 + dataset_id
        )
        # Mark the least-similar 40% of unique positive edges as hard. Surface
        # conflicts are included by the existing diverse hardness score.
        det_sim, combined_sim = tpl._build_similarity_matrix(local_features)
        positive_rows = [row for row, ((i, j), label) in enumerate(zip(local_pairs, y)) if label > 0.5]
        scored = sorted(
            (tpl._positive_hard_score(local_features, combined_sim, *local_pairs[row], "diverse"), row)
            for row in positive_rows
        )
        hard_count = int(math.ceil(len(scored) * args.hard_positive_fraction))
        local_hard = np.zeros(len(local_pairs), dtype=bool)
        for _, row in scored[:hard_count]: local_hard[row] = True
        all_hard_positive.append(local_hard)

        pair_to_row = {_pair_key(i, j): row + pair_offset for row, (i, j) in enumerate(local_pairs)}
        by_label: dict[str, list[int]] = defaultdict(list)
        for idx, label in enumerate(ds.labels): by_label[str(label)].append(idx)
        for anchor, label in enumerate(ds.labels):
            rows = [pair_to_row[_pair_key(anchor, other)] for other in by_label[str(label)] if other != anchor and _pair_key(anchor, other) in pair_to_row]
            if rows: connectivity_groups.append(np.asarray(rows, dtype=np.int64))
        for label, members in by_label.items():
            pos_rows = sorted({pair_to_row[_pair_key(i, j)] for x, i in enumerate(members) for j in members[x + 1:] if _pair_key(i, j) in pair_to_row})
            neg_rows = sorted({row + pair_offset for row, ((i, j), target) in enumerate(zip(local_pairs, y)) if target < 0.5 and (i in members or j in members)})
            if pos_rows and neg_rows: prototype_groups.append((np.asarray(pos_rows, dtype=np.int64), np.asarray(neg_rows, dtype=np.int64)))

        all_pairs.extend((i + ds.start, j + ds.start) for i, j in local_pairs)
        all_y.append(y)
        all_ds.append(np.full(len(y), dataset_id, dtype=np.int64))
        if len(triplets):
            all_triplets.append(triplets + pair_offset)
        if len(triangles):
            all_triangles.append(triangles + pair_offset)
        pair_stats = dict(pair_stats)
        pair_stats.update({
            "dataset": ds.name,
            "dataset_id": dataset_id,
            "pairs": len(y),
            "triplets": len(triplets),
            "triangles": len(triangles),
        })
        stats.append(pair_stats)
        pair_offset += len(y)

    return PairData(
        pairs=all_pairs,
        labels=np.concatenate(all_y).astype(np.float32),
        dataset_ids=np.concatenate(all_ds),
        dataset_names=[x.name for x in slices],
        triplets=np.vstack(all_triplets) if all_triplets else np.zeros((0, 2), dtype=np.int64),
        triangles=np.vstack(all_triangles) if all_triangles else np.zeros((0, 3), dtype=np.int64),
        hard_positive_mask=np.concatenate(all_hard_positive) if all_hard_positive else np.zeros(0, dtype=bool),
        connectivity_groups=connectivity_groups,
        prototype_groups=prototype_groups,
        stats=stats,
    )


def graph_refine_probability(prob: np.ndarray, gamma: float) -> np.ndarray:
    """Conservative two-hop max-min propagation for clustering consistency."""
    if gamma <= 0.0 or prob.shape[0] <= 2:
        return prob.astype(np.float32, copy=True)
    p = np.clip(prob.astype(np.float32), 0.0, 1.0)
    n = p.shape[0]
    path2 = np.zeros_like(p)
    for via in range(n):
        path2 = np.maximum(path2, np.minimum(p[:, via:via + 1], p[via:via + 1, :]))
    out = (1.0 - float(gamma)) * p + float(gamma) * path2
    out = (out + out.T) * 0.5
    np.fill_diagonal(out, 1.0)
    return out.astype(np.float32)


def _gradient_reverse(x, scale: float):  # type: ignore[no-untyped-def]
    import torch

    class GradientReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):  # type: ignore[no-untyped-def]
            ctx.scale = scale
            return value.view_as(value)

        @staticmethod
        def backward(ctx, grad):  # type: ignore[no-untyped-def]
            return -ctx.scale * grad

    return GradientReverse.apply(x)


def make_model(
    input_dim: int,
    num_domains: int,
    width: int,
    representation_dim: int,
    dropout: float,
    model_arch: str = "unified_residual",
    args: argparse.Namespace | None = None,
):
    import torch
    from torch import nn

    if model_arch != "unified_residual":
        from pairwise_neural_models import build_pairwise_neural_model
        return build_pairwise_neural_model(
            input_dim=input_dim,
            model_arch=model_arch,
            dropout=dropout,
            layernorm=True,
            batchnorm=False,
            gate_reg=float(getattr(args, "gate_reg", 1e-4)),
            ft_d_token=int(getattr(args, "ft_d_token", 64)),
            ft_layers=int(getattr(args, "ft_layers", 2)),
            ft_heads=int(getattr(args, "ft_heads", 4)),
            ft_dropout=float(getattr(args, "ft_dropout", 0.1)),
            ft_attention_dropout=float(getattr(args, "ft_attention_dropout", 0.1)),
            ft_ffn_mult=int(getattr(args, "ft_ffn_mult", 2)),
            ft_max_tokens=0,
        )

    class ResidualBlock(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            )
            self.activation = nn.GELU()

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.activation(x + self.net(x))

    class UnifiedPairModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(dropout),
                ResidualBlock(width),
                ResidualBlock(width),
                nn.Linear(width, width // 2),
                nn.LayerNorm(width // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                ResidualBlock(width // 2),
                nn.Linear(width // 2, representation_dim),
                nn.LayerNorm(representation_dim),
                nn.GELU(),
            )
            self.pair_head = nn.Linear(representation_dim, 1)
            self.domain_head = nn.Sequential(
                nn.Linear(representation_dim, max(32, representation_dim // 2)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(32, representation_dim // 2), num_domains),
            )

        def encode(self, x):  # type: ignore[no-untyped-def]
            return self.trunk(x)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.pair_head(self.encode(x)).squeeze(-1)

        def domain_logits(self, representation, grl_scale: float):  # type: ignore[no-untyped-def]
            return self.domain_head(_gradient_reverse(representation, grl_scale))

    return UnifiedPairModel()


def train_unified_model(
    X: np.ndarray,
    pair_data: PairData,
    args: argparse.Namespace,
    config: str,
    seed: int,
    teacher_probs: np.ndarray | None = None,
    easy_mask: np.ndarray | None = None,
    bridge_X: np.ndarray | None = None,
    bridge_weight: float = 0.0,
    bridge_batch_ratio: float = 0.25,
    bridge_weights: list[float] | None = None,
) -> dict:
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch import nn
    from torch.utils.data import DataLoader, WeightedRandomSampler

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    device = plf.pf.resolve_torch_device(args.device)

    train_mask = np.ones(len(X), dtype=bool)
    val_indices: list[int] = []
    for dataset_id in range(len(pair_data.dataset_names)):
        idx = np.flatnonzero(pair_data.dataset_ids == dataset_id)
        if len(idx) < 8:
            continue
        rng.shuffle(idx)
        take = max(1, int(round(len(idx) * args.validation_fraction)))
        val_indices.extend(idx[:take].tolist())
    train_mask[np.asarray(val_indices, dtype=np.int64)] = False
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.asarray(val_indices, dtype=np.int64)
    if len(val_idx) == 0:
        val_idx = train_idx[: min(len(train_idx), 128)]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_all = scaler.transform(X).astype(np.float32)
    X_all_t = torch.from_numpy(X_all)
    bridge_all_t = None
    bridge_weights_t = None
    if bridge_X is not None and len(bridge_X):
        bridge_scaled = scaler.transform(bridge_X).astype(np.float32)
        bridge_all_t = torch.from_numpy(bridge_scaled)
        if bridge_weights is not None and len(bridge_weights) == len(bridge_X):
            bridge_weights_t = torch.from_numpy(
                np.asarray(bridge_weights, dtype=np.float32)
            )
    y_all_t = torch.from_numpy(pair_data.labels.astype(np.float32))
    ds_all_t = torch.from_numpy(pair_data.dataset_ids.astype(np.int64))

    counts = np.bincount(pair_data.dataset_ids[train_idx], minlength=len(pair_data.dataset_names))
    if config in {"hard_pos", "hard_pos_connect", "hard_pos_prototype"}:
        sample_weights = np.zeros(len(train_idx), dtype=np.float64)
        for dataset_id in range(len(pair_data.dataset_names)):
            local = np.flatnonzero(pair_data.dataset_ids[train_idx] == dataset_id)
            if not len(local): continue
            global_rows = train_idx[local]
            hard = pair_data.hard_positive_mask[global_rows]
            easy = (pair_data.labels[global_rows] > 0.5) & ~hard
            negative = pair_data.labels[global_rows] < 0.5
            for mask, mass in ((hard, args.positive_mass * args.hard_positive_fraction), (easy, args.positive_mass * (1.0 - args.hard_positive_fraction)), (negative, 1.0 - args.positive_mass)):
                if np.any(mask): sample_weights[local[mask]] = mass / np.sum(mask)
        sample_weights = np.maximum(sample_weights, 1e-12)
    else:
        sample_weights = np.asarray(
            [1.0 / max(1, counts[pair_data.dataset_ids[idx]]) for idx in train_idx],
            dtype=np.float64,
        )
    micro_batch_size = min(args.batch_size, 128) if args.model_arch == "ft_transformer" else args.batch_size
    accumulation_steps = max(1, int(math.ceil(args.batch_size / micro_batch_size)))
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=max(len(train_idx), args.steps_per_epoch * args.batch_size),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        torch.from_numpy(train_idx.astype(np.int64)),
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=0,
    )

    model = make_model(
        X.shape[1], len(pair_data.dataset_names), args.width,
        args.representation_dim, args.dropout, args.model_arch, args,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    pos = float(np.sum(pair_data.labels[train_idx] == 1.0))
    neg = float(np.sum(pair_data.labels[train_idx] == 0.0))
    pos_weight = torch.tensor(neg / max(1.0, pos), dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    domain_ce = nn.CrossEntropyLoss()

    use_ranking = config in {"rank", "rank_trans", "rank_trans_domain"}
    use_transitivity = config in {"rank_trans", "rank_trans_domain"}
    use_domain = config == "rank_trans_domain"
    use_connectivity = config == "hard_pos_connect"
    use_prototype = config == "hard_pos_prototype"
    triplets = pair_data.triplets
    triangles = pair_data.triangles
    if len(triplets):
        triplets = triplets[np.all(train_mask[triplets], axis=1)]
    if len(triangles):
        triangles = triangles[np.all(train_mask[triangles], axis=1)]

    def focal_loss(logits, targets):  # type: ignore[no-untyped-def]
        raw = bce(logits, targets)
        prob = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, prob, 1.0 - prob)
        alpha_pos = neg / max(pos + neg, 1.0)
        alpha_t = torch.where(
            targets > 0.5,
            torch.full_like(targets, alpha_pos),
            torch.full_like(targets, 1.0 - alpha_pos),
        )
        return (alpha_t * torch.pow(1.0 - pt, args.focal_gamma) * raw).mean()

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history: list[dict] = []
    for epoch in range(args.epochs):
        model.train()
        totals = defaultdict(float)
        batches = 0
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps = 0
        for batch_number, batch_indices in enumerate(loader):
            idx = batch_indices.numpy()
            xb = X_all_t[idx].to(device)
            yb = y_all_t[idx].to(device)
            db = ds_all_t[idx].to(device)
            if args.model_arch == "unified_residual":
                representation = model.encode(xb)
                logits = model.pair_head(representation).squeeze(-1)
            else:
                representation = None
                logits = model(xb)
            cls_loss = focal_loss(logits, yb)
            loss = cls_loss
            totals["classification"] += float(cls_loss.detach().cpu())
            if bridge_all_t is not None and bridge_weight > 0.0:
                bridge_count = max(1, int(round(len(idx) * bridge_batch_ratio)))
                bridge_idx = rng.integers(0, len(bridge_all_t), size=bridge_count)
                bridge_logits = model(bridge_all_t[bridge_idx].to(device))
                bridge_targets = torch.ones_like(bridge_logits)
                if bridge_weights_t is not None:
                    # Quality-weighted bridge loss
                    sample_w = bridge_weights_t[torch.from_numpy(bridge_idx)]
                    bridge_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        bridge_logits, bridge_targets,
                        weight=sample_w.to(device),
                    )
                else:
                    bridge_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        bridge_logits, bridge_targets
                    )
                loss = loss + float(bridge_weight) * bridge_loss
                totals["bridge"] += float(bridge_loss.detach().cpu())
            if teacher_probs is not None and easy_mask is not None:
                easy = np.asarray(easy_mask[idx], dtype=bool)
                if np.any(easy):
                    easy_t = torch.from_numpy(easy).to(device)
                    teacher = torch.from_numpy(
                        np.asarray(teacher_probs[idx][easy], dtype=np.float32)
                    ).to(device)
                    student = torch.sigmoid(logits[easy_t])
                    distill_loss = torch.nn.functional.mse_loss(student, teacher)
                    weight = float(getattr(args, "distillation_weight", 0.30))
                    loss = loss + weight * distill_loss
                    totals["distillation"] += float(distill_loss.detach().cpu())

            aux_count = min(args.aux_batch_size, max(1, len(idx) // 2))
            if use_ranking and len(triplets):
                chosen = triplets[rng.integers(0, len(triplets), size=aux_count)]
                pos_logits = model(X_all_t[chosen[:, 0]].to(device))
                neg_logits = model(X_all_t[chosen[:, 1]].to(device))
                rank_loss = torch.nn.functional.softplus(
                    args.ranking_margin - pos_logits + neg_logits
                ).mean()
                loss = loss + args.ranking_weight * rank_loss
                totals["ranking"] += float(rank_loss.detach().cpu())

            if use_transitivity and len(triangles):
                chosen = triangles[rng.integers(0, len(triangles), size=aux_count)]
                tri_logits = model(X_all_t[chosen.reshape(-1)].to(device)).reshape(-1, 3)
                tri_prob = torch.sigmoid(tri_logits)
                a, b, c = tri_prob[:, 0], tri_prob[:, 1], tri_prob[:, 2]
                trans_loss = (
                    torch.relu(a + b - c - 1.0)
                    + torch.relu(a + c - b - 1.0)
                    + torch.relu(b + c - a - 1.0)
                ).mean()
                loss = loss + args.transitivity_weight * trans_loss
                totals["transitivity"] += float(trans_loss.detach().cpu())

            if use_connectivity and pair_data.connectivity_groups:
                chosen_groups = [pair_data.connectivity_groups[int(x)] for x in rng.integers(0, len(pair_data.connectivity_groups), size=min(args.connectivity_batch_size, len(pair_data.connectivity_groups)))]
                connect_terms = []
                for group in chosen_groups:
                    chosen = group if len(group) <= args.connectivity_edges else rng.choice(group, size=args.connectivity_edges, replace=False)
                    edge_prob = torch.sigmoid(model(X_all_t[chosen].to(device)))
                    top = torch.topk(edge_prob, k=min(args.connectivity_top_m, len(edge_prob))).values.mean()
                    connect_terms.append(torch.relu(torch.as_tensor(args.connectivity_margin, device=device) - top))
                if connect_terms:
                    connect_loss = torch.stack(connect_terms).mean()
                    loss = loss + args.connectivity_weight * connect_loss
                    totals["connectivity"] += float(connect_loss.detach().cpu())

            if use_prototype and pair_data.prototype_groups:
                groups = [pair_data.prototype_groups[int(x)] for x in rng.integers(0, len(pair_data.prototype_groups), size=min(args.prototype_batch_size, len(pair_data.prototype_groups)))]
                prototype_terms = []
                for positive_rows, negative_rows in groups:
                    pos_choice = positive_rows if len(positive_rows) <= args.prototype_edges else rng.choice(positive_rows, size=args.prototype_edges, replace=False)
                    neg_choice = negative_rows if len(negative_rows) <= args.prototype_edges else rng.choice(negative_rows, size=args.prototype_edges, replace=False)
                    pos_score = model(X_all_t[pos_choice].to(device)).mean()
                    neg_score = torch.logsumexp(model(X_all_t[neg_choice].to(device)), dim=0) - math.log(max(1, len(neg_choice)))
                    prototype_terms.append(torch.nn.functional.softplus(args.prototype_margin - pos_score + neg_score))
                if prototype_terms:
                    prototype_loss = torch.stack(prototype_terms).mean()
                    loss = loss + args.prototype_weight * prototype_loss
                    totals["prototype"] += float(prototype_loss.detach().cpu())

            if use_domain and representation is not None and len(pair_data.dataset_names) > 1:
                domain_loss = domain_ce(
                    model.domain_logits(representation, args.domain_grl_scale), db
                )
                loss = loss + args.domain_weight * domain_loss
                totals["domain"] += float(domain_loss.detach().cpu())

            (loss / accumulation_steps).backward()
            totals["total"] += float(loss.detach().cpu())
            batches += 1
            if (batch_number + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if optimizer_steps >= args.steps_per_epoch:
                break
        if batches % accumulation_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        model.eval()
        with torch.no_grad():
            val_chunks = []
            val_step = 128 if args.model_arch == "ft_transformer" else max(1, len(val_idx))
            for start in range(0, len(val_idx), val_step):
                val_chunks.append(model(X_all_t[val_idx[start:start + val_step]].to(device)))
            val_logits = torch.cat(val_chunks)
            val_loss = float(
                focal_loss(val_logits, y_all_t[val_idx].to(device)).detach().cpu()
            )
        row = {"epoch": epoch, "val_loss": val_loss}
        row.update({key: value / max(1, batches) for key, value in totals.items()})
        history.append(row)
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            bad_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1
            if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {
        "model": model,
        "scaler": scaler,
        "device": device,
        "best_val_loss": best_val,
        "history": history,
        "config": config,
        "model_arch": args.model_arch,
        "input_dim": X.shape[1],
        "width": args.width,
        "representation_dim": args.representation_dim,
        "dropout": args.dropout,
        "bridge_weight": float(bridge_weight),
        "bridge_batch_ratio": float(bridge_batch_ratio),
        "num_bridge_edges": 0 if bridge_X is None else int(len(bridge_X)),
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


def predict_probability(
    model_pkg: dict,
    features: list[plf.LLMCaseFeature],
    batch_size: int,
) -> np.ndarray:
    import torch

    pairs = [(i, j) for i in range(len(features)) for j in range(i + 1, len(features))]
    X = plf.build_rich_pair_feature_matrix(
        features, pairs, feature_mode="llm_dual_struct_det_summary"
    )
    X = model_pkg["scaler"].transform(X).astype(np.float32)
    model = model_pkg["model"]
    device = model_pkg["device"]
    values: list[np.ndarray] = []
    model.eval()
    effective_batch = min(batch_size, 128) if model_pkg.get("model_arch") == "ft_transformer" else batch_size
    with torch.no_grad():
        for start in range(0, len(X), effective_batch):
            logits = model(torch.from_numpy(X[start:start + effective_batch]).to(device))
            values.append(torch.sigmoid(logits).detach().cpu().numpy())
    scores = np.concatenate(values).astype(np.float32) if values else np.zeros(0, dtype=np.float32)
    prob = np.eye(len(features), dtype=np.float32)
    for (i, j), score in zip(pairs, scores):
        prob[i, j] = prob[j, i] = float(score)
    return prob


def run_fold(
    args: argparse.Namespace,
    datasets: list[Path],
    holdout_index: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    holdout = datasets[holdout_index]
    train_datasets = [ds for idx, ds in enumerate(datasets) if idx != holdout_index]
    ordered = train_datasets + [holdout]
    slices = build_slices(ordered)
    llm_args = plf._make_llm_args(
        llm_mode="embedding",
        llm_doc_style="features",
        llm_cache_dir=args.llm_cache_dir,
        svd_dim=args.svd_dim,
        llm_dual=True,
    )
    features, _bundle = plf.build_llm_case_features_for_inputs(
        [ds / "input.csv" for ds in ordered],
        svd_dim=args.svd_dim,
        llm_args=llm_args,
    )
    train_stop = slices[-1].start
    train_features = features[:train_stop]
    holdout_features = features[train_stop:]
    llm_reducer = plf.fit_llm_reducer(
        train_features, args.llm_reduce_dim, random_state=seed
    )
    summary_reducer = plf.fit_llm_summary_reducer(
        train_features, args.llm_reduce_dim, random_state=seed
    )
    plf.apply_llm_reducer(holdout_features, llm_reducer, args.llm_reduce_dim)
    plf.apply_llm_summary_reducer(
        holdout_features, summary_reducer, args.llm_reduce_dim
    )
    train_slices = slices[:-1]
    pair_data = build_pair_data(train_features, train_slices, args, seed)
    X = plf.build_rich_pair_feature_matrix(
        train_features,
        pair_data.pairs,
        feature_mode="llm_dual_struct_det_summary",
    )
    print(
        f"[fold] holdout={holdout.name} seed={seed} input_dim={X.shape[1]} "
        f"pairs={len(X)} triplets={len(pair_data.triplets)} "
        f"triangles={len(pair_data.triangles)}",
        flush=True,
    )

    gold = slices[-1].labels
    k = len(set(gold))
    result_rows: list[dict] = []
    stat_rows: list[dict] = []
    for model_arch in args.model_arches:
      args.model_arch = model_arch
      for config in args.configs:
        t0 = time.perf_counter()
        model_pkg = train_unified_model(X, pair_data, args, config, seed)
        prob_raw = predict_probability(model_pkg, holdout_features, args.predict_batch_size)
        model_dir = args.output_dir / "models" / f"seed_{seed}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"holdout_{holdout.name}_{model_arch}_{config}.pt"
        preproc_path = model_path.with_suffix(".preproc.pkl")
        import torch
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model_pkg["model"].state_dict().items()
                },
                "input_dim": model_pkg["input_dim"],
                "width": model_pkg["width"],
                "representation_dim": model_pkg["representation_dim"],
                "dropout": model_pkg["dropout"],
                "config": config,
                "model_arch": model_arch,
            },
            model_path,
        )
        with preproc_path.open("wb") as f:
            pickle.dump(
                {
                    "scaler": model_pkg["scaler"],
                    "llm_reducer": llm_reducer,
                    "llm_summary_reducer": summary_reducer,
                    "llm_reduce_dim": args.llm_reduce_dim,
                },
                f,
            )

        for gamma in args.graph_gammas:
            prob = graph_refine_probability(prob_raw, gamma)
            labels = plf.cluster_from_probability(prob, k)
            method = f"{model_arch}_{config}_graph{gamma:g}"
            pred_path = (
                args.output_dir / "preds" / f"{holdout.name}_{method}_seed{seed}.csv"
            )
            prob_path = (
                args.output_dir / "probs" / f"{holdout.name}_{method}_seed{seed}.npy"
            )
            pred = write_pred(pred_path, slices[-1].cases, labels)
            prob_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(prob_path, prob)
            ba, tpr, tnr = pairwise_scores(gold, pred)
            result_rows.append({
                "seed": seed,
                "holdout_dataset": holdout.name,
                "train_datasets": "+".join(ds.name for ds in train_datasets),
                "config": config,
                "model_arch": model_arch,
                "graph_gamma": gamma,
                "BA": ba,
                "TPR": tpr,
                "TNR": tnr,
                "k": k,
                "cases": len(gold),
                "num_pred_clusters": len(set(pred)),
                "input_dim": X.shape[1],
                "best_val_loss": model_pkg["best_val_loss"],
                "runtime_sec": time.perf_counter() - t0,
                "model_path": str(model_path),
                "pred_path": str(pred_path),
                "prob_path": str(prob_path),
            })
            print(
                f"[eval] holdout={holdout.name} seed={seed} arch={model_arch} config={config} "
                f"graph={gamma:g} BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f}",
                flush=True,
            )
        for stat in pair_data.stats:
            stat_rows.append({
                "seed": seed,
                "holdout_dataset": holdout.name,
                "config": config,
                "model_arch": model_arch,
                **stat,
            })
        history_path = model_dir / f"holdout_{holdout.name}_{model_arch}_{config}_history.json"
        history_path.write_text(
            json.dumps(model_pkg["history"], indent=2) + "\n", encoding="utf-8"
        )
    return result_rows, stat_rows


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("model_arch", "unified_residual")), str(row["config"]), float(row["graph_gamma"]))].append(row)
    output: list[dict] = []
    for (model_arch, config, gamma), values in groups.items():
        dataset_means: dict[str, float] = {}
        for dataset in sorted({str(row["holdout_dataset"]) for row in values}):
            dataset_means[dataset] = float(np.mean([
                float(row["BA"]) for row in values if row["holdout_dataset"] == dataset
            ]))
        bas = list(dataset_means.values())
        output.append({
            "model_arch": model_arch,
            "config": config,
            "graph_gamma": gamma,
            "mean_BA": float(np.mean(bas)),
            "min_dataset_BA": float(np.min(bas)),
            "std_dataset_BA": float(np.std(bas)),
            "robust_score": float(np.mean(bas) - 0.5 * np.std(bas)),
            "mean_TPR": float(np.mean([float(row["TPR"]) for row in values])),
            "mean_TNR": float(np.mean([float(row["TNR"]) for row in values])),
            "datasets": len(dataset_means),
            "runs": len(values),
        })
    return sorted(
        output,
        key=lambda row: (
            float(row["robust_score"]),
            float(row["min_dataset_BA"]),
            float(row["mean_BA"]),
        ),
        reverse=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified multi-dataset pairwise experiments")
    p.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    p.add_argument("--holdouts", nargs="+", default=["all"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--configs", nargs="+", choices=(
        "balanced", "rank", "rank_trans", "rank_trans_domain",
        "hard_pos", "hard_pos_connect", "hard_pos_prototype"
    ), default=["balanced", "rank", "rank_trans", "rank_trans_domain"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    p.add_argument("--model-arches", nargs="+", choices=(
        "unified_residual", "res_mlp", "gated_mlp", "ft_transformer"
    ), default=["unified_residual"])
    p.add_argument("--gate-reg", type=float, default=1e-4)
    p.add_argument("--ft-d-token", type=int, default=64)
    p.add_argument("--ft-layers", type=int, default=2)
    p.add_argument("--ft-heads", type=int, default=4)
    p.add_argument("--ft-dropout", type=float, default=0.1)
    p.add_argument("--ft-attention-dropout", type=float, default=0.1)
    p.add_argument("--ft-ffn-mult", type=int, default=2)
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--llm-reduce-dim", type=int, default=64)
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--hard-positive-fraction", type=float, default=0.40)
    p.add_argument("--positive-mass", type=float, default=1.0 / 3.0)
    p.add_argument("--connectivity-weight", type=float, default=0.20)
    p.add_argument("--connectivity-margin", type=float, default=0.65)
    p.add_argument("--connectivity-top-m", type=int, default=2)
    p.add_argument("--connectivity-edges", type=int, default=8)
    p.add_argument("--connectivity-batch-size", type=int, default=64)
    p.add_argument("--prototype-weight", type=float, default=0.20)
    p.add_argument("--prototype-margin", type=float, default=0.5)
    p.add_argument("--prototype-edges", type=int, default=16)
    p.add_argument("--prototype-batch-size", type=int, default=32)
    p.add_argument("--negative-ratio", type=float, default=2.0)
    p.add_argument("--hard-negative-ratio", type=float, default=0.6)
    p.add_argument("--hard-positive-ratio", type=float, default=0.25)
    p.add_argument("--max-pairs-per-dataset", type=int, default=30000)
    p.add_argument("--max-aux-per-dataset", type=int, default=10000)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--representation-dim", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--aux-batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=2e-4)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--ranking-weight", type=float, default=0.25)
    p.add_argument("--ranking-margin", type=float, default=0.5)
    p.add_argument("--transitivity-weight", type=float, default=0.10)
    p.add_argument("--domain-weight", type=float, default=0.05)
    p.add_argument("--domain-grl-scale", type=float, default=0.20)
    p.add_argument("--validation-fraction", type=float, default=0.10)
    p.add_argument("--early-stop-patience", type=int, default=6)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--graph-gammas", nargs="+", type=float, default=[0.0, 0.10, 0.20])
    p.add_argument("--predict-batch-size", type=int, default=100000)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [resolve(path) for path in args.datasets]
    missing = [str(ds) for ds in datasets if not (ds / "input.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"missing datasets: {missing}")
    if args.holdouts == ["all"]:
        holdout_indices = list(range(len(datasets)))
    else:
        names = {name: idx for idx, name in enumerate(ds.name for ds in datasets)}
        holdout_indices = [names[name] for name in args.holdouts]

    rows: list[dict] = []
    stats: list[dict] = []
    for seed in args.seeds:
        for holdout_index in holdout_indices:
            fold_rows, fold_stats = run_fold(args, datasets, holdout_index, seed)
            rows.extend(fold_rows)
            stats.extend(fold_stats)
            write_csv(args.output_dir / "results.partial.csv", rows, list(rows[0]))

    result_fields = [
        "seed", "holdout_dataset", "train_datasets", "model_arch", "config", "graph_gamma",
        "BA", "TPR", "TNR", "k", "cases", "num_pred_clusters", "input_dim",
        "best_val_loss", "runtime_sec", "model_path", "pred_path", "prob_path",
    ]
    write_csv(args.output_dir / "results.csv", rows, result_fields)
    summary = summarize(rows)
    summary_fields = [
        "model_arch", "config", "graph_gamma", "mean_BA", "min_dataset_BA",
        "std_dataset_BA", "robust_score", "mean_TPR", "mean_TNR",
        "datasets", "runs",
    ]
    write_csv(args.output_dir / "summary.csv", summary, summary_fields)
    if stats:
        write_csv(args.output_dir / "pair_stats.csv", stats, list(stats[0]))
    manifest = {
        "datasets": [str(ds) for ds in datasets],
        "holdouts": [datasets[idx].name for idx in holdout_indices],
        "configs": args.configs,
        "model_arches": args.model_arches,
        "seeds": args.seeds,
        "feature_mode": "llm_dual_struct_det_summary",
        "llm_reduce_dim": args.llm_reduce_dim,
        "formal_predictor_modified": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("\n| rank | architecture | config | graph | mean BA | min BA | std | robust score | TPR | TNR |")
    print("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(summary[:20], 1):
        print(
            f"| {rank} | {row['model_arch']} | {row['config']} | {row['graph_gamma']:.2f} | "
            f"{row['mean_BA']:.4f} | {row['min_dataset_BA']:.4f} | "
            f"{row['std_dataset_BA']:.4f} | {row['robust_score']:.4f} | "
            f"{row['mean_TPR']:.4f} | {row['mean_TNR']:.4f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
