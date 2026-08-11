#!/usr/bin/env python3
"""Train experimental Theta compact pair-student artifacts.

Training may read gold/golden labels. The resulting inference artifact does not.
Pairs are always sampled inside one benchmark episode.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

import evaluation_leakage_guard as elg
import official_style_features as osf
import theta_features as tf
import train_pairwise_llm as tpl
from run_experiments import read_gold


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    Path("old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("official_format_fake_dataset/stable_official_like_multitest_v1"),
    Path("official_format_fake_dataset/directed_cross_v2"),
    Path("test_case/problem/benchmark_set_1"),
    Path("test_case/problem/benchmark_set_2"),
]


@dataclass
class Episode:
    path: Path
    name: str
    family: str
    cases: list[tf.ThetaCaseFeature]
    labels: list[str]
    dropped_duplicates: int = 0


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def infer_family(dataset: Path) -> str:
    name = dataset.name.lower()
    if name in {
        "first_batch_dataset", "stage2_dataset_working",
        "stage3_dataset_32bugs_640cases",
    }:
        return "old_fake_family"
    if name in {"benchmark_set_1", "benchmark_set_2"}:
        return "official_public_family"
    if "directed_cross" in name:
        return "directed_family"
    if "stable_official" in name:
        return "stable_family"
    if "official_vcs" in name:
        return "vcs_family"
    if "benchmark5" in name:
        return "benchmark5_family"
    return dataset.name


def parse_family_map(raw: Sequence[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"invalid --family-map entry {item!r}; expected DATASET=FAMILY")
        dataset, family = item.split("=", 1)
        output[dataset.strip()] = family.strip()
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family-map", nargs="*", default=[])
    parser.add_argument("--embedding-mode", choices=["embedding", "none"], default="embedding")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/theta_llm_cache"))
    parser.add_argument("--llm-batch-size", type=int, default=128)
    parser.add_argument("--llm-timeout-sec", type=float, default=120.0)
    parser.add_argument(
        "--model-type", choices=["gbdt", "logistic", "gated_mlp"], default="gbdt"
    )
    parser.add_argument("--max-pairs-per-family", type=int, default=24000)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.5)
    parser.add_argument("--hard-positive-ratio", type=float, default=0.5)
    parser.add_argument(
        "--family-balance-power", type=float, default=1.0,
        help="0 disables family balancing; 1 gives every family/class equal loss mass.",
    )
    parser.add_argument("--parser", default="drain")
    parser.add_argument("--svd-dim", type=int, default=64)
    parser.add_argument("--context-radius", type=int, default=2)
    parser.add_argument("--context-events", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--gate-reg", type=float, default=1e-4)
    return parser.parse_args(argv)


def load_episodes(args: argparse.Namespace) -> list[Episode]:
    family_overrides = parse_family_map(args.family_map)
    episodes: list[Episode] = []
    for raw in args.train_datasets:
        dataset = resolve(raw)
        cases, _ = tf.build_theta_case_features(
            dataset / "input.csv",
            parser=args.parser,
            svd_dim=args.svd_dim,
            context_radius=args.context_radius,
            context_events=args.context_events,
        )
        labels = read_gold(osf.gold_path(dataset))
        if len(cases) != len(labels):
            raise RuntimeError(
                f"Theta label alignment failed for {dataset}: cases={len(cases)} labels={len(labels)}"
            )
        episodes.append(Episode(
            path=dataset,
            name=dataset.name,
            family=family_overrides.get(dataset.name, infer_family(dataset)),
            cases=cases,
            labels=list(map(str, labels)),
        ))
    return episodes


def deduplicate_within_families(episodes: Sequence[Episode]) -> list[Episode]:
    """Keep the largest episode first, then only unseen cases in that family."""
    output: list[Episode] = []
    families = sorted({episode.family for episode in episodes})
    for family in families:
        members = sorted(
            (episode for episode in episodes if episode.family == family),
            key=lambda episode: (-len(episode.cases), episode.name),
        )
        seen: set[str] = set()
        for episode in members:
            keep = [idx for idx, case in enumerate(episode.cases) if case.fingerprint not in seen]
            for idx in keep:
                seen.add(episode.cases[idx].fingerprint)
            output.append(Episode(
                path=episode.path,
                name=episode.name,
                family=episode.family,
                cases=[episode.cases[idx] for idx in keep],
                labels=[episode.labels[idx] for idx in keep],
                dropped_duplicates=len(episode.cases) - len(keep),
            ))
    return sorted(output, key=lambda episode: episode.name)


def sample_family_balanced_pairs(
    episodes: Sequence[Episode], args: argparse.Namespace,
) -> tuple[
    list[tf.ThetaCaseFeature], list[tuple[int, int]], np.ndarray,
    np.ndarray, list[str], list[dict],
]:
    cases: list[tf.ThetaCaseFeature] = []
    offsets: dict[str, int] = {}
    for episode in episodes:
        offsets[episode.name] = len(cases)
        cases.extend(episode.cases)

    families: dict[str, list[Episode]] = {}
    for episode in episodes:
        if len(episode.cases) >= 2 and len(set(episode.labels)) >= 1:
            families.setdefault(episode.family, []).append(episode)
    all_pairs: list[tuple[int, int]] = []
    all_labels: list[np.ndarray] = []
    all_family_ids: list[np.ndarray] = []
    stats: list[dict] = []
    family_names = sorted(families)
    for family_index, family in enumerate(family_names):
        family_episodes = families[family]
        episode_budget = max(64, int(math.ceil(args.max_pairs_per_family / len(family_episodes))))
        for episode_index, episode in enumerate(family_episodes):
            pairs, y, pair_stats = tpl.sample_pairs(
                [case.base for case in episode.cases],
                episode.labels,
                negative_ratio=args.negative_ratio,
                hard_negative_ratio=args.hard_negative_ratio,
                hard_positive_ratio=args.hard_positive_ratio,
                max_train_pairs=episode_budget,
                random_state=(
                    int(args.random_state) * 1009 + family_index * 131 + episode_index * 17
                ),
                positive_sampling="diverse",
                negative_sampling="confusable",
            )
            offset = offsets[episode.name]
            all_pairs.extend((i + offset, j + offset) for i, j in pairs)
            all_labels.append(y.astype(np.float32))
            all_family_ids.append(np.full(len(y), family_index, dtype=np.int32))
            stats.append({
                "dataset": episode.name,
                "family": family,
                "cases": len(episode.cases),
                "dropped_duplicates": episode.dropped_duplicates,
                "pair_budget": episode_budget,
                "pairs": len(y),
                **pair_stats,
            })
    if not all_labels:
        raise RuntimeError("Theta training produced no labeled pairs")
    return (
        cases,
        all_pairs,
        np.concatenate(all_labels).astype(np.float32),
        np.concatenate(all_family_ids),
        family_names,
        stats,
    )


def family_class_weights(
    labels: np.ndarray,
    family_ids: np.ndarray,
    family_names: Sequence[str],
    balance_power: float,
) -> tuple[np.ndarray, list[dict]]:
    """Balance source families and positive/negative classes independently."""
    labels = np.asarray(labels, dtype=np.float32)
    family_ids = np.asarray(family_ids, dtype=np.int32)
    weights = np.ones(len(labels), dtype=np.float64)
    power = min(1.0, max(0.0, float(balance_power)))
    debug: list[dict] = []
    for family_id, family in enumerate(family_names):
        family_mask = family_ids == family_id
        family_count = int(np.sum(family_mask))
        if family_count == 0:
            continue
        family_scale = family_count ** (-power)
        for target in (0.0, 1.0):
            mask = family_mask & (labels == target)
            count = int(np.sum(mask))
            if count == 0:
                continue
            class_scale = family_count / (2.0 * count)
            weights[mask] = family_scale * class_scale
        debug.append({
            "family": family,
            "pairs": family_count,
            "positive_pairs": int(np.sum(family_mask & (labels > 0.5))),
            "negative_pairs": int(np.sum(family_mask & (labels < 0.5))),
            "raw_weight_mean": float(np.mean(weights[family_mask])),
        })
    weights /= max(float(np.mean(weights)), 1e-12)
    weights = np.clip(weights, 0.02, 50.0).astype(np.float32)
    for row, family_id in zip(debug, range(len(debug))):
        mask = family_ids == family_id
        row["normalized_weight_mean"] = float(np.mean(weights[mask]))
        row["normalized_weight_min"] = float(np.min(weights[mask]))
        row["normalized_weight_max"] = float(np.max(weights[mask]))
    return weights, debug


def train_weighted_model(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    model_type: str,
    random_state: int,
    args: argparse.Namespace,
) -> dict:
    if model_type == "gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=5,
            learning_rate=0.05,
            early_stopping=False,
            random_state=random_state,
        )
        model.fit(X, y, sample_weight=sample_weight)
        return {"model": model, "scaler": None, "model_type": "theta_weighted_gbdt"}

    if model_type == "gated_mlp":
        return train_weighted_gated_mlp(X, y, sample_weight, args)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=2000, random_state=random_state, solver="lbfgs")
    model.fit(X_scaled, y, sample_weight=sample_weight)
    return {"model": model, "scaler": scaler, "model_type": "theta_weighted_logistic"}


def train_weighted_gated_mlp(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    """Train the Theta neural student with family-balanced focal loss."""
    import torch
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    from pairwise_neural_models import build_pairwise_neural_model, default_hidden_dims
    from pairwise_features import resolve_torch_device

    seed = int(args.random_state)
    torch.manual_seed(seed)
    np.random.seed(seed)
    indices = np.arange(len(y))
    if len(y) >= 20 and len(np.unique(y)) == 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        train_idx, val_idx = next(splitter.split(X, y))
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        split = max(1, int(round(0.85 * len(indices))))
        train_idx, val_idx = indices[:split], indices[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_val = scaler.transform(X[val_idx]).astype(np.float32)
    y_train = y[train_idx].astype(np.float32)
    y_val = y[val_idx].astype(np.float32)
    w_train = sample_weight[train_idx].astype(np.float32)
    w_val = sample_weight[val_idx].astype(np.float32)
    device = resolve_torch_device(str(args.device))
    hidden_dims = default_hidden_dims("gated_mlp")
    model_config = {"gate_reg": float(args.gate_reg)}
    model = build_pairwise_neural_model(
        input_dim=X.shape[1],
        model_arch="gated_mlp",
        hidden_dims=hidden_dims,
        dropout=float(args.dropout),
        layernorm=True,
        batchnorm=False,
        **model_config,
    ).to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
            torch.from_numpy(w_train),
        ),
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )

    def weighted_focal(logits, target, weights):  # type: ignore[no-untyped-def]
        raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, prob, 1.0 - prob)
        raw = raw * torch.pow(1.0 - pt, float(args.focal_gamma))
        return torch.sum(raw * weights) / torch.clamp(weights.sum(), min=1e-6)

    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    w_val_t = torch.from_numpy(w_val).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict] = []
    for epoch in range(max(1, int(args.epochs))):
        model.train()
        train_sum = 0.0
        train_batches = 0
        for batch_x, batch_y, batch_w in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = weighted_focal(logits, batch_y, batch_w) + model.regularization_loss()
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach().cpu())
            train_batches += 1
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = float(weighted_focal(val_logits, y_val_t, w_val_t).cpu())
        history.append({
            "epoch": epoch,
            "train_loss": train_sum / max(1, train_batches),
            "val_loss": val_loss,
        })
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= max(1, int(args.early_stop_patience)):
                break
    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    return {
        "model": model,
        "state_dict": best_state,
        "scaler": scaler,
        "model_type": "theta_gated_mlp",
        "input_dim": int(X.shape[1]),
        "hidden_dims": hidden_dims,
        "dropout": float(args.dropout),
        "model_config": model_config,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "history": history,
    }


def save_theta_model(model_pkg: dict, output_dir: Path) -> None:
    """Keep torch tensors and sklearn preprocessing in separate files."""
    if model_pkg.get("model_type") != "theta_gated_mlp":
        joblib.dump(model_pkg, output_dir / "pair_student.pkl", compress=3)
        return
    import torch

    torch.save(
        {
            "state_dict": model_pkg["state_dict"],
            "input_dim": model_pkg["input_dim"],
            "hidden_dims": model_pkg["hidden_dims"],
            "dropout": model_pkg["dropout"],
            "model_config": model_pkg["model_config"],
            "model_type": model_pkg["model_type"],
        },
        output_dir / "pair_student.pt",
    )
    preprocessing = {
        key: value for key, value in model_pkg.items()
        if key not in {"model", "state_dict"}
    }
    joblib.dump(preprocessing, output_dir / "pair_student.pkl", compress=3)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    args.output_dir = resolve(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = deduplicate_within_families(load_episodes(args))
    cases, pairs, y, family_ids, family_names, pair_stats = sample_family_balanced_pairs(
        episodes, args
    )

    embedding_model = "none"
    raw_embedding_dim = 0
    embedding_stats = {"documents": 0, "unique_documents": 0}
    if args.embedding_mode == "embedding":
        embedding_model, raw_embedding_dim, embedding_stats = tf.fetch_theta_embeddings(
            cases,
            cache_dir=args.llm_cache_dir,
            batch_size=args.llm_batch_size,
            timeout_sec=args.llm_timeout_sec,
        )
    reducers = tf.fit_theta_reducers(cases, args.embedding_dim, args.random_state)
    X = tf.build_theta_pair_feature_matrix(cases, pairs)
    sample_weight, weight_stats = family_class_weights(
        y, family_ids, family_names, args.family_balance_power
    )
    model_pkg = train_weighted_model(
        X, y, sample_weight, args.model_type, args.random_state, args
    )

    joblib.dump(reducers, args.output_dir / "reducers.pkl", compress=3)
    save_theta_model(model_pkg, args.output_dir)
    training_paths = [episode.path for episode in episodes]
    manifest = {
        "schema_version": 1,
        "model_name": "theta_v1_compact_student",
        "experimental": True,
        "created_unix": time.time(),
        "parser": args.parser,
        "svd_dim": args.svd_dim,
        "context_radius": args.context_radius,
        "context_events": args.context_events,
        "embedding_mode": args.embedding_mode,
        "embedding_model": embedding_model,
        "raw_embedding_dim": raw_embedding_dim,
        "reduced_embedding_dim": args.embedding_dim if raw_embedding_dim else 0,
        "embedding_stats": embedding_stats,
        "model_type": args.model_type,
        "trained_model_type": model_pkg["model_type"],
        "best_epoch": int(model_pkg.get("best_epoch", -1)),
        "best_val_loss": float(model_pkg.get("best_val_loss", -1.0)),
        "family_balance_power": args.family_balance_power,
        "feature_dim": int(X.shape[1]),
        "random_state": args.random_state,
        "training_cases_after_dedup": len(cases),
        "training_pairs": len(y),
        "positive_pairs": int(np.sum(y > 0.5)),
        "negative_pairs": int(np.sum(y < 0.5)),
        "families": sorted({episode.family for episode in episodes}),
        "episodes": [
            {
                "dataset": episode.name,
                "family": episode.family,
                "cases_after_dedup": len(episode.cases),
                "dropped_duplicates": episode.dropped_duplicates,
            }
            for episode in episodes
        ],
        "pair_stats": pair_stats,
        "weight_stats": weight_stats,
        "training_dataset_names": [path.name for path in training_paths],
        "training_datasets": [
            elg.dataset_identity(path, include_case_logs=True) for path in training_paths
        ],
        "runtime_sec": time.perf_counter() - started,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[theta-train] cases={len(cases)} pairs={len(y)} feature_dim={X.shape[1]} "
        f"families={len(manifest['families'])} embedding_dim={raw_embedding_dim} "
        f"runtime={manifest['runtime_sec']:.2f}s",
        flush=True,
    )
    for row in pair_stats:
        print(
            f"[theta-train] family={row['family']} dataset={row['dataset']} "
            f"cases={row['cases']} dropped={row['dropped_duplicates']} pairs={row['pairs']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
