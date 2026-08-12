#!/usr/bin/env python3
"""Final-submission training: LODO fake pretrain + official fine-tune + save.

For each fake dataset (leave-one-out fold) we fit reducers on the remaining
datasets (8 fake + 2 official), pretrain the TriLog two-tower on fake-only
pairs, then fine-tune the frozen-backbone head on the official pairs. The
fine-tuned package and every reducer needed to reproduce the pair-feature
matrix at inference time are persisted under ``--output-dir/models``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import joblib

import graph_clustering as gc
import official_style_features as osf
import pairwise_llm_features as plf
import run_graph_multiview_experiments as gm
import theta_trace_features as ttf
import theta_trilog_model as ttm
from run_experiments import pairwise_scores, read_gold
from run_official_full_retrain_experiments import write_csv, write_pred
from run_theta_trilog_lodo import (
    _config_matrix,
    _pair_labels,
    _probability_matrix,
    resolve,
)

OFFICIAL_NAMES = {"benchmark_set_1", "benchmark_set_2"}

FINAL_DATASETS = [
    Path("dataset/fake_dataset/old_fake_dataset/first_batch_dataset"),
    Path("dataset/fake_dataset/old_fake_dataset/stage2_dataset_working"),
    Path("dataset/fake_dataset/old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("dataset/fake_dataset/official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("dataset/fake_dataset/official_format_fake_dataset/directed_cross_v2"),
    Path("dataset/fake_dataset/official_format_fake_dataset/directed_cross_v4"),
    Path("dataset/fake_dataset/official_format_fake_dataset/stable_official_like_multitest_v1"),
    Path("dataset/fake_dataset/official_format_fake_dataset/benchmark5_final"),
    Path("dataset/fake_dataset/official_format_fake_dataset/benchmark6_final"),
    Path("dataset/real_dataset/benchmark_set_1"),
    Path("dataset/real_dataset/benchmark_set_2"),
]


def _case_episode_names(slices: Sequence[dict], case_count: int) -> list[str]:
    output = [""] * case_count
    for episode in slices:
        for index in range(int(episode["start"]), int(episode["stop"])):
            output[index] = str(episode["name"])
    if any(not name for name in output):
        raise ValueError("episode case ranges do not cover all cases")
    return output


def _pair_episode_names(
    pairs: Sequence[tuple[int, int]], case_episode: Sequence[str],
) -> list[str]:
    output: list[str] = []
    for left, right in pairs:
        if case_episode[left] != case_episode[right]:
            raise RuntimeError(f"cross-episode pair detected: {case_episode[left]} != {case_episode[right]}")
        output.append(case_episode[left])
    return output


def _finetune_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        random_state=seed,
        device=args.device,
        finetune_scope=args.finetune_scope,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        finetune_weight_decay=args.finetune_weight_decay,
        label_smoothing=args.label_smoothing,
        affine_reg=args.affine_reg,
        ranking_weight=args.ranking_weight,
        ranking_margin=args.ranking_margin,
        connectivity_weight=0.0,
        connectivity_top_m=args.connectivity_top_m,
        transitivity_weight=0.0,
        replay_weight=args.replay_weight,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", type=Path, default=FINAL_DATASETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--view-dim", type=int, default=64)
    parser.add_argument("--svd-dim", type=int, default=64)
    parser.add_argument("--parser", default="drain")
    parser.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    parser.add_argument("--llm-batch-size", type=int, default=64)
    parser.add_argument("--llm-timeout-sec", type=float, default=20.0)
    parser.add_argument("--llm-doc-max-features", type=int, default=80)
    parser.add_argument("--embedding-cache-only", action="store_true")
    parser.add_argument("--embedding-expected-dim", type=int, default=768)
    parser.add_argument("--trace-cache-dir", type=Path, default=Path("/tmp/theta_trilog_trace_cache"))
    parser.add_argument("--trace-segment-count", type=int, default=16)
    parser.add_argument("--trace-chunk-size", type=int, default=512)
    parser.add_argument("--trace-anchor-sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--trace-global-struct-dim", type=int, default=48)
    parser.add_argument("--trace-global-text-dim", type=int, default=48)
    parser.add_argument("--trace-anchor-struct-dim", type=int, default=32)
    parser.add_argument("--trace-anchor-text-dim", type=int, default=32)
    parser.add_argument("--trace-residual-struct-dim", type=int, default=48)
    parser.add_argument("--trace-residual-text-dim", type=int, default=48)
    parser.add_argument("--view-max-pairs-per-dataset", type=int, default=30000)
    parser.add_argument("--view-negative-ratio", type=float, default=2.0)
    parser.add_argument("--view-hard-negative-ratio", type=float, default=0.5)
    parser.add_argument("--view-hard-positive-ratio", type=float, default=1.0)
    parser.add_argument("--view-official-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--pretrain-batch-size", type=int, default=4096)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-weight-decay", type=float, default=1e-4)
    parser.add_argument("--pretrain-dropout", type=float, default=0.2)
    parser.add_argument("--pretrain-patience", type=int, default=6)
    parser.add_argument("--pretrain-focal-gamma", type=float, default=2.0)
    parser.add_argument("--fusion", choices=["concat", "sum"], default="concat")
    parser.add_argument("--supcon-weight", type=float, default=0.0)
    parser.add_argument("--supcon-temperature", type=float, default=0.1)
    parser.add_argument("--finetune-scope", choices=["last", "head"], default="last")
    parser.add_argument("--finetune-epochs", type=int, default=80)
    parser.add_argument("--finetune-lr", type=float, default=2e-4)
    parser.add_argument("--finetune-weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--affine-reg", type=float, default=0.20)
    parser.add_argument("--ranking-weight", type=float, default=0.20)
    parser.add_argument("--ranking-margin", type=float, default=0.50)
    parser.add_argument("--connectivity-top-m", type=int, default=2)
    parser.add_argument("--replay-weight", type=float, default=0.30)
    parser.add_argument("--replay-pairs", type=int, default=4096)
    parser.add_argument("--final-clusterer", default="correlation_cluster")
    parser.add_argument("--source-clusterer", default="agglomerative_avg")
    parser.add_argument("--cannot-link-weight", type=float, default=100.0)
    parser.add_argument("--sanity-max-cases", type=int, default=400,
                        help="skip the O(n^2) held-out sanity score when the held-out dataset exceeds this")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = resolve(args.output_dir)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    datasets = [resolve(path) for path in args.datasets]
    if args.embedding_cache_only:
        import regr_fail_bucketing as rfb

        rfb.fetch_llm_embeddings = gm.fetch_cached_llm_embeddings
    llm_args = gm.make_embedding_args(args)

    base_features: list[plf.LLMCaseFeature] = []
    trace_features: list[ttf.HierarchicalTraceFeature] = []
    custom_documents = {name: [] for name in ("event", "object", "context")}
    slices: list[dict] = []
    offset = 0
    for dataset in datasets:
        episode_features, _ = plf.build_llm_case_features_for_inputs(
            [dataset / "input.csv"], parser=args.parser, svd_dim=args.svd_dim, llm_args=llm_args
        )
        labels = read_gold(osf.gold_path(dataset))
        cases = osf.read_cases(dataset / "input.csv")
        if not (len(episode_features) == len(labels) == len(cases)):
            raise RuntimeError(f"feature/label mismatch for {dataset}")
        base_features.extend(episode_features)
        slices.append({
            "name": dataset.name, "path": dataset, "start": offset,
            "stop": offset + len(labels), "labels": labels, "cases": cases,
        })
        offset += len(labels)
        documents = gm.build_all_view_documents(dataset)
        for name in custom_documents:
            custom_documents[name].extend(documents[name])
        episode_trace, _ = ttf.build_hierarchical_trace_features(
            dataset / "input.csv", cache_dir=args.trace_cache_dir,
            segment_count=args.trace_segment_count, chunk_size=args.trace_chunk_size,
            anchor_sizes=args.trace_anchor_sizes,
        )
        trace_features.extend(episode_trace)
        print(
            f"[final-train-load] dataset={dataset.name} cases={len(labels)} "
            f"trace_ok={sum(feature.has_trace for feature in episode_trace)}",
            flush=True,
        )
    raw_custom = {
        name: gm.fetch_view_embeddings(documents, args, name)
        for name, documents in custom_documents.items()
    }
    case_episode = _case_episode_names(slices, len(base_features))
    fold_names = [episode["name"] for episode in slices if episode["name"] not in OFFICIAL_NAMES]
    rows: list[dict] = []

    for seed in args.seeds:
        for fold in fold_names:
            started = time.perf_counter()
            train_indices = [
                index for episode in slices if episode["name"] != fold
                for index in range(episode["start"], episode["stop"])
            ]
            hold_indices = [
                index for episode in slices if episode["name"] == fold
                for index in range(episode["start"], episode["stop"])
            ]
            train_indices_np = np.asarray(train_indices, dtype=np.int64)

            train_base_features = [base_features[index] for index in train_indices]
            feature_reducer = plf.fit_llm_reducer(train_base_features, args.view_dim, random_state=seed)
            summary_reducer = plf.fit_llm_summary_reducer(train_base_features, args.view_dim, random_state=seed + 17)

            custom_reducers: dict[str, object] = {}
            reduced_custom: dict[str, np.ndarray] = {}
            for pos, (name, raw) in enumerate(raw_custom.items()):
                reducer, _ = plf._fit_reducer_for_matrix(
                    np.asarray(raw, dtype=np.float32)[train_indices_np], args.view_dim, seed + 101 + pos * 13
                )
                custom_reducers[name] = reducer
                reduced_custom[name] = plf._apply_reducer_to_matrix(
                    np.asarray(raw, dtype=np.float32), reducer, args.view_dim
                ).astype(np.float32, copy=False)

            trace_bundle, trace_matrices = ttf.fit_transform_trace_views(
                trace_features, train_indices, seed=seed,
                global_struct_dim=args.trace_global_struct_dim,
                global_text_dim=args.trace_global_text_dim,
                anchor_struct_dim=args.trace_anchor_struct_dim,
                anchor_text_dim=args.trace_anchor_text_dim,
                residual_struct_dim=args.trace_residual_struct_dim,
                residual_text_dim=args.trace_residual_text_dim,
            )

            train_pairs, train_y, train_weight, _ = gm.sample_lodo_train_pairs(
                base_features, slices, fold, args, seed
            )
            pair_episode = _pair_episode_names(train_pairs, case_episode)
            fake_mask = np.asarray([name not in OFFICIAL_NAMES for name in pair_episode], dtype=bool)
            official_mask = np.asarray([name in OFFICIAL_NAMES for name in pair_episode], dtype=bool)
            if not np.any(fake_mask):
                raise RuntimeError(f"no fake pretraining pairs for {fold}")

            view_names = gm.views_for_config("quad_event_object_context")
            train_base = gm.build_multiview_pair_feature_matrix(
                base_features, reduced_custom, view_names, train_pairs
            )
            train_trace = ttf.build_trace_pair_feature_components(
                trace_features, trace_matrices, train_pairs
            )["residual"]
            train_matrix = _config_matrix(
                "trilog_residual", train_base,
                np.zeros((len(train_pairs), 0), dtype=np.float32),
                np.zeros((len(train_pairs), 0), dtype=np.float32),
                train_trace, np.zeros((len(train_pairs), 0), dtype=np.float32),
                np.zeros((len(train_pairs), 0), dtype=np.float32),
            )

            pretrain_args = argparse.Namespace(
                random_state=seed, device=args.device, epochs=args.pretrain_epochs,
                batch_size=args.pretrain_batch_size, lr=args.pretrain_lr,
                weight_decay=args.pretrain_weight_decay, dropout=args.pretrain_dropout,
                early_stop_patience=args.pretrain_patience,
                focal_gamma=args.pretrain_focal_gamma,
                fusion=args.fusion, supcon_weight=args.supcon_weight,
                supcon_temperature=args.supcon_temperature,
            )
            fake_weight = train_weight[fake_mask].copy()
            fake_weight /= max(float(np.mean(fake_weight)), 1e-12)
            pretrain_model = ttm.train_trilog_pair_model(
                train_matrix[fake_mask], train_y[fake_mask], fake_weight,
                int(train_base.shape[1]), pretrain_args,
            )

            package = pretrain_model
            official_rows = np.flatnonzero(official_mask)
            if len(official_rows):
                rng = np.random.default_rng(seed * 1009 + int(hold_indices[0]))
                fake_rows = np.flatnonzero(fake_mask)
                replay_count = min(int(args.replay_pairs), len(fake_rows))
                replay_rows = rng.choice(fake_rows, size=replay_count, replace=False)
                official_pairs = [train_pairs[int(row)] for row in official_rows]
                package = ttm.fine_tune_official_pair_model(
                    pretrain_model,
                    official_matrix=train_matrix[official_rows],
                    official_labels=train_y[official_rows],
                    official_pairs=official_pairs,
                    official_weight=train_weight[official_rows],
                    replay_matrix=train_matrix[replay_rows],
                    args=_finetune_args(args, seed),
                )

            torch.save(package, output_dir / "models" / f"model_{fold}_seed{seed}.pt")
            preprocess = {
                "feature_reducer": feature_reducer,
                "summary_reducer": summary_reducer,
                "custom_reducers": custom_reducers,
                "trace_bundle": trace_bundle,
                "view_names": view_names,
            }
            joblib.dump(preprocess, output_dir / "models" / f"preprocess_{fold}_seed{seed}.pkl")

            # Fold sanity check: score this fold-model on its held-out fake dataset.
            # O(n^2) in the held-out case count, so skip it for large datasets.
            fold_episode = next(ep for ep in slices if ep["name"] == fold)
            if len(hold_indices) <= args.sanity_max_cases:
                hold_pairs = osf.all_pairs(len(hold_indices))
                hold_y = _pair_labels(fold_episode["labels"], hold_pairs)
                hold_base_features = [base_features[index] for index in hold_indices]
                plf.apply_llm_reducer(hold_base_features, feature_reducer, args.view_dim)
                plf.apply_llm_summary_reducer(hold_base_features, summary_reducer, args.view_dim)
                hold_custom = {name: matrix[hold_indices] for name, matrix in reduced_custom.items()}
                hold_base = gm.build_multiview_pair_feature_matrix(
                    hold_base_features, hold_custom, view_names, hold_pairs
                )
                hold_trace_features = [trace_features[index] for index in hold_indices]
                hold_trace_matrices = {name: matrix[hold_indices] for name, matrix in trace_matrices.items()}
                hold_trace = ttf.build_trace_pair_feature_components(
                    hold_trace_features, hold_trace_matrices, hold_pairs
                )["residual"]
                hold_matrix = _config_matrix(
                    "trilog_residual", hold_base,
                    np.zeros((len(hold_pairs), 0), dtype=np.float32),
                    np.zeros((len(hold_pairs), 0), dtype=np.float32),
                    hold_trace, np.zeros((len(hold_pairs), 0), dtype=np.float32),
                    np.zeros((len(hold_pairs), 0), dtype=np.float32),
                )
                flat = ttm.predict_trilog_pair_model(package, hold_matrix)
                probability = _probability_matrix(hold_pairs, flat, len(hold_indices))
                clustered = gc.cluster_with_fallback(
                    probability, len(set(fold_episode["labels"])), cannot_link_weight=args.cannot_link_weight
                )
                pred = write_pred(output_dir / "preds" / f"{fold}_seed{seed}.csv", fold_episode["cases"], clustered.labels)
                ba, tpr, tnr = pairwise_scores(fold_episode["labels"], pred)
                rows.append({
                    "fold": fold, "seed": seed, "BA": ba, "TPR": tpr, "TNR": tnr,
                    "clusters": clustered.num_clusters, "k": len(set(fold_episode["labels"])),
                    "runtime_sec": time.perf_counter() - started,
                })
                print(
                    f"[final-train] fold={fold} seed={seed} BA={ba:.4f} "
                    f"clusters={clustered.num_clusters}/{len(set(fold_episode['labels']))} "
                    f"t={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            else:
                rows.append({
                    "fold": fold, "seed": seed, "BA": None, "TPR": None, "TNR": None,
                    "clusters": None, "k": len(set(fold_episode["labels"])),
                    "runtime_sec": time.perf_counter() - started,
                })
                print(
                    f"[final-train] fold={fold} seed={seed} skip-sanity n={len(hold_indices)} "
                    f"> {args.sanity_max_cases} t={time.perf_counter() - started:.1f}s",
                    flush=True,
                )

    manifest = {
        "folds": fold_names,
        "seeds": args.seeds,
        "view_dim": args.view_dim,
        "svd_dim": args.svd_dim,
        "parser": args.parser,
        "llm_doc_max_features": args.llm_doc_max_features,
        "llm_batch_size": args.llm_batch_size,
        "embedding_expected_dim": args.embedding_expected_dim,
        "fusion": args.fusion,
        "supcon_weight": args.supcon_weight,
        "finetune_scope": args.finetune_scope,
        "trace_component": "residual",
        "trace_segment_count": args.trace_segment_count,
        "trace_chunk_size": args.trace_chunk_size,
        "trace_anchor_sizes": list(args.trace_anchor_sizes),
        "final_clusterer": args.final_clusterer,
        "source_clusterer": args.source_clusterer,
        "consensus_weight": 0.0,
        "cannot_link_weight": args.cannot_link_weight,
        "view_names": gm.views_for_config("quad_event_object_context"),
        "datasets": [str(dataset) for dataset in datasets],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_csv(output_dir / "results.csv", rows, ["fold", "seed", "BA", "TPR", "TNR", "clusters", "k", "runtime_sec"])
    print(f"[final-train] wrote {len(rows)} fold results to {output_dir / 'results.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
