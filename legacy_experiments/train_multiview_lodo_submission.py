#!/usr/bin/env python3
"""Train 7-fold LODO multi-view models for Beta v3 deployment.

For each fold (held-out dataset), trains models on the OTHER 6 datasets.
Saves to models_multiview_lodo/ with naming: model_{fold_name}_{view_config}_seed{N}.pkl
"""
import argparse, json, sys, time
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import official_style_features as osf
import pairwise_llm_features as plf
import regr_fail_bucketing as rfb
import train_pairwise_llm as tpl
from beta_multiview_inference import (
    build_feature_documents, canonicalize_document, install_log_sample_cache,
)
from run_experiments import read_gold
from run_graph_multiview_experiments import (
    build_all_view_documents, build_multiview_pair_feature_matrix,
    make_embedding_args, train_view_model, views_for_config,
)
from run_half_split_experiments import DEFAULT_DATASETS as _IGNORE  # use our own list

PROJECT_ROOT = Path(__file__).resolve().parent

DATASETS = [
    Path("old_fake_dataset/first_batch_dataset"),
    Path("old_fake_dataset/stage2_dataset_working"),
    Path("old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("official_format_fake_dataset/directed_cross_v2"),
    Path("test_case/problem/benchmark_set_1"),
    Path("test_case/problem/benchmark_set_2"),
]
SEEDS = [0, 1, 2, 3, 4]
VIEW_CONFIGS = ["dual", "quad_event_object_context"]


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--view-dim", type=int, default=64)
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--max-pairs-per-dataset", type=int, default=30000)
    p.add_argument("--negative-ratio", type=float, default=2.0)
    p.add_argument("--hard-negative-ratio", type=float, default=0.5)
    p.add_argument("--hard-positive-ratio", type=float, default=1.0)
    p.add_argument("--llm-doc-max-features", type=int, default=80)
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--llm-batch-size", type=int, default=128)
    p.add_argument("--llm-timeout-sec", type=float, default=600.0)
    p.add_argument("--canonicalize-case-indices", action="store_true")
    args = p.parse_args()

    install_log_sample_cache()
    datasets = [resolve(d) for d in DATASETS]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    llm_args = make_embedding_args(args)

    # Build global features and embeddings ONCE
    features = []
    docs = {v: [] for v in ("features", "summary", "event", "object", "context")}
    for dataset in datasets:
        feats, _ = plf.build_llm_case_features_for_inputs(
            [dataset / "input.csv"], parser="drain", svd_dim=args.svd_dim,
            llm_args=None, log_llm_disabled=False)
        features.extend(feats)
        fd, sd = build_feature_documents(dataset / "input.csv", "drain", args.llm_doc_max_features)
        custom = build_all_view_documents(dataset)
        for view, values in {"features": fd, "summary": sd, **custom}.items():
            docs[view].extend(
                canonicalize_document(view, doc) if args.canonicalize_case_indices else doc
                for doc in values)

    counts = {v: len(vals) for v, vals in docs.items()}
    combined = [d for vals in docs.values() for d in vals]
    unique_docs = list(dict.fromkeys(combined))
    unique_idx = {d: i for i, d in enumerate(unique_docs)}
    inverse = np.fromiter((unique_idx[d] for d in combined), dtype=np.int64, count=len(combined))
    embeddings, model_name = rfb.fetch_llm_embeddings(unique_docs, llm_args)
    umat = np.asarray(embeddings, dtype=np.float32)
    umat /= np.maximum(np.linalg.norm(umat, axis=1, keepdims=True), np.float32(1e-12))
    matrix = umat[inverse]
    raw_views = {}
    off = 0
    for view, count in counts.items():
        raw_views[view] = matrix[off:off + count].astype(np.float32, copy=False)
        off += count
        print(f"[lodo-train] view={view} docs={count} unique={len(unique_docs)}", flush=True)

    for idx, feat in enumerate(features):
        feat.llm_vec = raw_views["features"][idx]
        feat.llm_summary_vec = raw_views["summary"][idx]
    raw_custom = {v: raw_views[v] for v in ("event", "object", "context")}

    # Build per-dataset slices
    slices = []
    off = 0
    for dataset in datasets:
        labels = read_gold(osf.gold_path(dataset))
        slices.append({"name": dataset.name, "start": off, "stop": off + len(labels), "labels": labels})
        off += len(labels)

    manifest = {
        "schema_version": 1, "created_unix": time.time(),
        "seeds": SEEDS, "view_configs": VIEW_CONFIGS,
        "view_dim": args.view_dim, "svd_dim": args.svd_dim,
        "parser": "drain", "primary_beta": 0.50, "consensus_weight": 0.00,
        "source_clusterer": "agglomerative_complete",
        "final_clusterer": "agglomerative_avg",
        "embedding_dim": 768,
        "canonicalized_docs": bool(args.canonicalize_case_indices),
        "training_datasets": [d.name for d in datasets],
        "training_case_count": len(features),
        "lodo": True,
        "folds": [d.name for d in datasets],
        "models": [],
    }

    for fold_idx, held_out in enumerate(slices):
        fold_name = held_out["name"]
        train_indices = [i for ds in slices if ds["name"] != fold_name
                         for i in range(ds["start"], ds["stop"])]
        train_feats = [features[i] for i in train_indices]

        print(f"\n[lodo-train] Fold {fold_idx+1}/{len(slices)}: held_out={fold_name} "
              f"train_cases={len(train_indices)}", flush=True)

        for seed in SEEDS:
            print(f"[lodo-train]   seed={seed} fitting reducers", flush=True)
            fr = plf.fit_llm_reducer(train_feats, args.view_dim, random_state=seed)
            sr = plf.fit_llm_summary_reducer(train_feats, args.view_dim, random_state=seed + 17)
            custom_reducers = {}
            reduced_custom = {}
            for view, raw in raw_custom.items():
                reducer, _ = plf._fit_reducer_for_matrix(raw[train_indices], args.view_dim, seed + 101)
                custom_reducers[view] = reducer
                # Apply to ALL data (not just train) so pair indices match global positions
                reduced_custom[view] = plf._apply_reducer_to_matrix(
                    raw, reducer, args.view_dim).astype(np.float32, copy=False)

            preproc_path = args.output_dir / f"preprocess_{fold_name}_seed{seed}.pkl"
            joblib.dump({
                "feature_reducer": fr, "summary_reducer": sr,
                "custom_reducers": custom_reducers,
            }, preproc_path, compress=3)

            # Sample training pairs from training datasets only
            train_slices = [ds for ds in slices if ds["name"] != fold_name]
            pairs, y, pair_stats = [], [], []
            for ds_idx, item in enumerate(train_slices):
                s, e = item["start"], item["stop"]
                local_feats = [features[i] for i in range(s, e)]
                lp, ly, ps = tpl.sample_pairs(
                    local_feats, item["labels"],
                    negative_ratio=args.negative_ratio,
                    hard_negative_ratio=args.hard_negative_ratio,
                    hard_positive_ratio=args.hard_positive_ratio,
                    max_train_pairs=args.max_pairs_per_dataset,
                    random_state=seed * 1009 + ds_idx * 97 + 31,
                    positive_sampling="diverse", negative_sampling="confusable",
                )
                pairs.extend((i + s, j + s) for i, j in lp)
                y.append(ly.astype(np.float32))
                pair_stats.append({"dataset": item["name"], "pairs": len(ly), **ps})

            y_vec = np.concatenate(y).astype(np.float32)
            print(f"[lodo-train]   seed={seed} pairs={len(y_vec)} pos={int(np.sum(y_vec > 0.5))}", flush=True)

            # Clone features and apply reducers (so pair matrix uses reduced dims)
            feats_copy = [plf.LLMCaseFeature(
                case_id=f.case_id, det_vec=f.det_vec.copy(),
                llm_vec=f.llm_vec.copy(), llm_vec_reduced=f.llm_vec_reduced,
                llm_summary_vec=f.llm_summary_vec.copy(),
                llm_summary_vec_reduced=f.llm_summary_vec_reduced,
                trace_vec=f.trace_vec.copy(), trace_vec_reduced=f.trace_vec_reduced,
                tokens=list(f.tokens), token_set=f.token_set,
                primary_tokens=f.primary_tokens, sim_tokens=f.sim_tokens,
                regr_tokens=f.regr_tokens, info=dict(f.info),
            ) for f in features]
            plf.apply_llm_reducer(feats_copy, fr, args.view_dim)
            plf.apply_llm_summary_reducer(feats_copy, sr, args.view_dim)

            for vc in VIEW_CONFIGS:
                views = views_for_config(vc)
                t0 = time.perf_counter()
                X = build_multiview_pair_feature_matrix(feats_copy, reduced_custom, views, pairs)
                model_pkg = train_view_model(X, y_vec, "gbdt", seed)
                model_path = args.output_dir / f"model_{fold_name}_{vc}_seed{seed}.pkl"
                joblib.dump(model_pkg, model_path, compress=3)
                item = {
                    "fold": fold_name, "seed": seed, "view_config": vc,
                    "feature_dim": int(X.shape[1]), "pairs": len(y_vec),
                    "model_file": model_path.name,
                    "preprocess_file": preproc_path.name,
                    "runtime_sec": time.perf_counter() - t0,
                    "pair_stats": pair_stats,
                }
                manifest["models"].append(item)
                print(f"[lodo-train]   saved {model_path.name} dim={X.shape[1]}", flush=True)

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n[lodo-train] Done! {len(manifest['models'])} models saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
