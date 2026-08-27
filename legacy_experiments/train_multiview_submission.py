#!/usr/bin/env python3
"""Train final experimental multi-view GBDT artifacts for submission inference.

This is a training-only utility. It reads labels from the explicitly supplied
datasets, samples pairs within each dataset, and exports reducers/models without
embedding any gold, meta, or trace data in the runtime manifest.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

import official_style_features as osf
import pairwise_llm_features as plf
import regr_fail_bucketing as rfb
import train_pairwise_llm as tpl
from evaluation_leakage_guard import dataset_identity
from beta_multiview_inference import (
    build_feature_documents,
    canonicalize_document,
    install_log_sample_cache,
)
from run_experiments import read_gold
from run_graph_multiview_experiments import (
    build_all_view_documents,
    build_multiview_pair_feature_matrix,
    make_embedding_args,
    train_view_model,
    views_for_config,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_DATASETS = [
    Path("old_fake_dataset/first_batch_dataset"),
    Path("old_fake_dataset/stage2_dataset_working"),
    Path("old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("official_format_fake_dataset/directed_cross_v2"),
    Path("test_case/problem/benchmark_set_1"),
    Path("test_case/problem/benchmark_set_2"),
]


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final multi-view submission artifacts")
    parser.add_argument("--train-datasets", nargs="+", type=Path, default=DEFAULT_TRAIN_DATASETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--view-configs", nargs="+", default=["dual", "quad_event_object_context"])
    parser.add_argument("--view-dim", type=int, default=64)
    parser.add_argument("--max-pairs-per-dataset", type=int, default=30000)
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.5)
    parser.add_argument("--hard-positive-ratio", type=float, default=1.0)
    parser.add_argument("--parser", default="drain")
    parser.add_argument("--svd-dim", type=int, default=64)
    parser.add_argument("--llm-doc-max-features", type=int, default=80)
    parser.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    parser.add_argument("--llm-batch-size", type=int, default=128)
    parser.add_argument("--llm-timeout-sec", type=float, default=600.0)
    parser.add_argument("--canonicalize-case-indices", action="store_true")
    return parser.parse_args(argv)


def sample_training_pairs(features, slices, args, seed: int):
    pairs: list[tuple[int, int]] = []
    labels: list[np.ndarray] = []
    stats: list[dict] = []
    for ds_idx, item in enumerate(slices):
        start, stop = item["start"], item["stop"]
        local_pairs, y, pair_stats = tpl.sample_pairs(
            features[start:stop],
            item["labels"],
            negative_ratio=args.negative_ratio,
            hard_negative_ratio=args.hard_negative_ratio,
            hard_positive_ratio=args.hard_positive_ratio,
            max_train_pairs=args.max_pairs_per_dataset,
            random_state=seed * 1009 + ds_idx * 97 + 31,
            positive_sampling="diverse",
            negative_sampling="confusable",
        )
        pairs.extend((i + start, j + start) for i, j in local_pairs)
        labels.append(y.astype(np.float32))
        stats.append({"dataset": item["name"], "pairs": len(y), **pair_stats})
    return pairs, np.concatenate(labels).astype(np.float32), stats


def main(argv: Sequence[str] | None = None) -> int:
    install_log_sample_cache()
    args = parse_args(argv)
    datasets = [resolve(path) for path in args.train_datasets]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    llm_args = make_embedding_args(args)
    # Each benchmark is an independent episode at inference time. Build its
    # Drain/deterministic features independently during training as well.
    features = []
    docs = {view: [] for view in ("features", "summary", "event", "object", "context")}
    for dataset in datasets:
        dataset_features, _ = plf.build_llm_case_features_for_inputs(
            [dataset / "input.csv"],
            parser=args.parser,
            svd_dim=args.svd_dim,
            llm_args=None,
            log_llm_disabled=False,
        )
        features.extend(dataset_features)
        feature_docs, summary_docs = build_feature_documents(
            dataset / "input.csv", args.parser, args.llm_doc_max_features
        )
        custom_docs = build_all_view_documents(dataset)
        dataset_docs = {"features": feature_docs, "summary": summary_docs, **custom_docs}
        for view, values in dataset_docs.items():
            docs[view].extend(
                canonicalize_document(view, doc) if args.canonicalize_case_indices else doc
                for doc in values
            )

    counts = {view: len(values) for view, values in docs.items()}
    combined_docs = [doc for values in docs.values() for doc in values]
    unique_docs = list(dict.fromkeys(combined_docs))
    unique_index = {doc: idx for idx, doc in enumerate(unique_docs)}
    inverse = np.fromiter(
        (unique_index[doc] for doc in combined_docs), dtype=np.int64, count=len(combined_docs)
    )
    embeddings, model_name = rfb.fetch_llm_embeddings(unique_docs, llm_args)
    unique_matrix = np.asarray(embeddings, dtype=np.float32)
    if unique_matrix.ndim != 2 or unique_matrix.shape != (len(unique_docs), 768):
        raise RuntimeError(f"combined embedding shape {unique_matrix.shape}; refusing fallback artifacts")
    matrix = unique_matrix[inverse]
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), np.float32(1e-12))
    raw_views = {}
    offset = 0
    for view, count in counts.items():
        raw_views[view] = matrix[offset : offset + count].astype(np.float32, copy=False)
        offset += count
        print(
            f"[final-train] view={view} model={model_name} docs={count} "
            f"unique_total={len(unique_docs)} embedding_dim={raw_views[view].shape[1]}",
            flush=True,
        )
    for idx, feature in enumerate(features):
        feature.llm_vec = raw_views["features"][idx]
        feature.llm_summary_vec = raw_views["summary"][idx]
    raw_custom = {view: raw_views[view] for view in ("event", "object", "context")}

    slices = []
    offset = 0
    for dataset in datasets:
        labels = read_gold(osf.gold_path(dataset))
        slices.append({
            "name": dataset.name,
            "start": offset,
            "stop": offset + len(labels),
            "labels": labels,
        })
        offset += len(labels)

    manifest = {
        "schema_version": 1,
        "created_unix": time.time(),
        "seeds": args.seeds,
        "view_configs": args.view_configs,
        "view_dim": args.view_dim,
        "svd_dim": args.svd_dim,
        "parser": args.parser,
        "primary_beta": 0.50,
        "consensus_weight": 0.00,
        "source_clusterer": "agglomerative_complete",
        "final_clusterer": "agglomerative_avg",
        "embedding_dim": 768,
        "canonicalized_docs": bool(args.canonicalize_case_indices),
        "training_dataset_names": [dataset.name for dataset in datasets],
        "training_datasets": [dataset_identity(dataset, include_case_logs=True) for dataset in datasets],
        "training_case_count": len(features),
        "models": [],
    }

    for seed in args.seeds:
        print(f"[final-train] seed={seed} fitting reducers", flush=True)
        feature_reducer = plf.fit_llm_reducer(features, args.view_dim, random_state=seed)
        summary_reducer = plf.fit_llm_summary_reducer(features, args.view_dim, random_state=seed + 17)
        custom_reducers = {}
        reduced_custom = {}
        for view, raw in raw_custom.items():
            reducer, transformed = plf._fit_reducer_for_matrix(raw, args.view_dim, seed + 101)
            custom_reducers[view] = reducer
            reduced_custom[view] = transformed.astype(np.float32, copy=False)
        preproc_path = args.output_dir / f"preprocess_seed{seed}.pkl"
        joblib.dump({
            "feature_reducer": feature_reducer,
            "summary_reducer": summary_reducer,
            "custom_reducers": custom_reducers,
        }, preproc_path, compress=3)

        pairs, y, pair_stats = sample_training_pairs(features, slices, args, seed)
        print(
            f"[final-train] seed={seed} pairs={len(y)} pos={int(np.sum(y > 0.5))} "
            f"neg={int(np.sum(y <= 0.5))}",
            flush=True,
        )
        for view_config in args.view_configs:
            views = views_for_config(view_config)
            started = time.perf_counter()
            X = build_multiview_pair_feature_matrix(features, reduced_custom, views, pairs)
            model_pkg = train_view_model(X, y, "gbdt", seed)
            model_path = args.output_dir / f"model_{view_config}_seed{seed}.pkl"
            joblib.dump(model_pkg, model_path, compress=3)
            item = {
                "seed": seed,
                "view_config": view_config,
                "feature_dim": int(X.shape[1]),
                "pairs": len(y),
                "model_file": model_path.name,
                "preprocess_file": preproc_path.name,
                "runtime_sec": time.perf_counter() - started,
                "pair_stats": pair_stats,
            }
            manifest["models"].append(item)
            print(
                f"[final-train] seed={seed} view={view_config} dim={X.shape[1]} "
                f"runtime={item['runtime_sec']:.2f}s",
                flush=True,
            )

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[final-train] wrote {args.output_dir / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
