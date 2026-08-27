#!/usr/bin/env python3
"""Strict-LODO training for the production bounded-evidence dual student."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

import beta_multiview_inference as bmi
import graph_clustering as gc
import official_style_features as osf
import pairwise_llm_features as plf
import run_graph_multiview_experiments as gm
from bounded_evidence import build_bounded_evidence
from run_experiments import pairwise_scores, read_gold
from selective_multiview_inference import _embed_documents
from sparse_multiview_inference import build_evidence_case_features


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    Path("old_fake_dataset/first_batch_dataset"),
    Path("old_fake_dataset/stage2_dataset_working"),
    Path("old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("official_format_fake_dataset/directed_cross_v2"),
    Path("official_format_fake_dataset/stable_official_like_multitest_v1"),
    Path("test_case/problem/benchmark_set_1"),
    Path("test_case/problem/benchmark_set_2"),
]


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_prediction(path: Path, cases: Sequence[str], labels: Sequence[int]) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    buckets = [f"bucket_{int(label):03d}" for label in labels]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Case", "bucket"])
        writer.writerows(zip(cases, buckets))
    return buckets


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["dataset"])].append(row)
    output = []
    for dataset, values in groups.items():
        output.append({
            "dataset": dataset,
            "mean_BA": float(np.mean([float(row["BA"]) for row in values])),
            "std_BA": float(np.std([float(row["BA"]) for row in values])),
            "mean_TPR": float(np.mean([float(row["TPR"]) for row in values])),
            "mean_TNR": float(np.mean([float(row["TNR"]) for row in values])),
            "min_BA": float(np.min([float(row["BA"]) for row in values])),
            "num_runs": len(values),
        })
    output.append({
        "dataset": "macro",
        "mean_BA": float(np.mean([float(row["BA"]) for row in rows])),
        "std_BA": float(np.std([float(row["BA"]) for row in rows])),
        "mean_TPR": float(np.mean([float(row["TPR"]) for row in rows])),
        "mean_TNR": float(np.mean([float(row["TNR"]) for row in rows])),
        "min_BA": float(np.min([float(row["BA"]) for row in rows])),
        "num_runs": len(rows),
    })
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded-evidence dual student LODO")
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--parser", default="drain")
    parser.add_argument("--svd-dim", type=int, default=64)
    parser.add_argument("--view-dim", type=int, default=64)
    parser.add_argument("--evidence-max-bytes", type=int, default=64 * 1024)
    parser.add_argument("--evidence-dim", type=int, default=256)
    parser.add_argument("--llm-doc-max-features", type=int, default=80)
    parser.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/iccad_beta_multiview_cache"))
    parser.add_argument("--llm-batch-size", type=int, default=128)
    parser.add_argument("--llm-timeout-sec", type=float, default=120.0)
    parser.add_argument("--view-negative-ratio", type=float, default=2.0)
    parser.add_argument("--view-hard-negative-ratio", type=float, default=0.5)
    parser.add_argument("--view-hard-positive-ratio", type=float, default=1.0)
    parser.add_argument("--view-max-pairs-per-dataset", type=int, default=30000)
    parser.add_argument("--clusterer", default="agglomerative_complete")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = resolve(args.output_dir)
    datasets = [resolve(path) for path in args.datasets]
    llm_args = gm.make_embedding_args(args)

    features: list[plf.LLMCaseFeature] = []
    slices: list[dict] = []
    offset = 0
    for dataset in datasets:
        evidence = build_bounded_evidence(
            dataset / "input.csv", args.evidence_max_bytes, args.evidence_dim
        )
        evidence_features = build_evidence_case_features(
            evidence, np.arange(len(evidence), dtype=np.int64)
        )
        feature_docs, summary_docs = bmi.build_feature_documents(
            dataset / "input.csv", args.parser, args.llm_doc_max_features
        )
        raw_views, model_name, docs_total, docs_unique = _embed_documents(
            {"features": feature_docs, "summary": summary_docs},
            llm_args, canonicalize=True,
        )
        if any(
            matrix.shape[0] != len(evidence_features) or matrix.shape[1] != 768
            for matrix in raw_views.values()
        ):
            raise RuntimeError(
                f"{dataset.name}: embedding fallback/dimension mismatch "
                f"{[matrix.shape for matrix in raw_views.values()]}"
            )
        for index, target in enumerate(evidence_features):
            target.llm_vec = raw_views["features"][index]
            target.llm_summary_vec = raw_views["summary"][index]
        labels = read_gold(osf.gold_path(dataset))
        cases = [str(case) for case in osf.read_cases(dataset / "input.csv")]
        features.extend(evidence_features)
        slices.append({
            "name": dataset.name, "start": offset,
            "stop": offset + len(labels), "labels": labels, "cases": cases,
        })
        offset += len(labels)
        features_dim = raw_views["features"].shape[1]
        summary_dim = raw_views["summary"].shape[1]
        print(
            f"[evidence-student] prepared={dataset.name} cases={len(labels)} "
            f"model={model_name} features_dim={features_dim} "
            f"summary_dim={summary_dim} "
            f"docs={docs_total} unique_docs={docs_unique}",
            flush=True,
        )

    rows: list[dict] = []
    for seed in args.seeds:
        for holdout in slices:
            train_indices = [
                idx for item in slices if item["name"] != holdout["name"]
                for idx in range(item["start"], item["stop"])
            ]
            hold_indices = list(range(holdout["start"], holdout["stop"]))
            train_features = [features[idx] for idx in train_indices]
            hold_features = [features[idx] for idx in hold_indices]
            feature_reducer = plf.fit_llm_reducer(
                train_features, args.view_dim, random_state=seed
            )
            summary_reducer = plf.fit_llm_summary_reducer(
                train_features, args.view_dim, random_state=seed + 17
            )
            # Reducers are fit on training domains only, then applied to every
            # case so train and serving pair vectors share the same 293d schema.
            plf.apply_llm_reducer(features, feature_reducer, args.view_dim)
            plf.apply_llm_summary_reducer(features, summary_reducer, args.view_dim)
            train_pairs, y, _sample_weight, pair_stats = gm.sample_lodo_train_pairs(
                features, slices, holdout["name"], args, seed
            )
            hold_pairs = osf.all_pairs(len(hold_features))
            train_x = gm.build_multiview_pair_feature_matrix(
                features, {}, ["features", "summary"], train_pairs
            )
            hold_x = gm.build_multiview_pair_feature_matrix(
                hold_features, {}, ["features", "summary"], hold_pairs
            )
            if train_x.shape[1] != 293 or hold_x.shape[1] != 293:
                raise RuntimeError(
                    f"production schema mismatch: train={train_x.shape} hold={hold_x.shape}"
                )
            model_pkg = gm.train_view_model(train_x, y, "gbdt", seed)
            probability = gm.predict_view_probabilities(
                model_pkg, hold_x, hold_pairs, len(hold_features)
            )
            k = len(set(holdout["labels"]))
            result = gc.cluster_probability_graph(probability, k, args.clusterer)
            pred_path = args.output_dir / "preds" / (
                f"{holdout['name']}_dual_evidence_tail_gbdt_{args.clusterer}_seed{seed}.csv"
            )
            pred = write_prediction(pred_path, holdout["cases"], result.labels)
            prob_path = args.output_dir / "probs" / (
                f"{holdout['name']}_dual_evidence_tail_gbdt_{args.clusterer}_seed{seed}.npy"
            )
            prob_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(prob_path, probability)
            model_root = args.output_dir / "models" / holdout["name"]
            model_root.mkdir(parents=True, exist_ok=True)
            joblib.dump(model_pkg, model_root / f"model_dual_seed{seed}.pkl")
            joblib.dump({
                "feature_reducer": feature_reducer,
                "summary_reducer": summary_reducer,
                "view_dim": args.view_dim,
            }, model_root / f"preprocess_seed{seed}.pkl")
            ba, tpr, tnr = pairwise_scores(holdout["labels"], pred)
            rows.append({
                "dataset": holdout["name"], "seed": seed,
                "method": "dual_evidence_tail_gbdt_lodo",
                "BA": ba, "TPR": tpr, "TNR": tnr,
                "cases": len(pred), "k": k,
                "num_clusters": len(set(pred)),
                "train_pairs": len(y),
                "pair_stats": json.dumps(pair_stats, sort_keys=True),
                "pred_path": str(pred_path), "prob_path": str(prob_path),
            })
            print(
                f"[evidence-student] seed={seed} holdout={holdout['name']} "
                f"BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f}", flush=True,
            )

    write_csv(args.output_dir / "results.csv", rows)
    summary = summarize(rows)
    write_csv(args.output_dir / "summary.csv", summary)
    print("| dataset | mean BA | std | TPR | TNR | min BA | runs |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in summary:
        print(
            f"| {row['dataset']} | {row['mean_BA']:.4f} | {row['std_BA']:.4f} | "
            f"{row['mean_TPR']:.4f} | {row['mean_TNR']:.4f} | "
            f"{row['min_BA']:.4f} | {row['num_runs']} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
