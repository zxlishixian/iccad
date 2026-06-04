#!/usr/bin/env python3
"""Official-tuned full pairwise retraining experiments.

Experimental only. Trains pairwise LLM rich models from selected gold/golden
labeled datasets and evaluates on selected datasets. It does not modify the
formal regr_fail_bucketing.py predictor.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import official_style_features as osf
import pairwise_llm_features as plf
from run_experiments import pairwise_scores, read_gold
import train_pairwise_llm as tpl

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_EVAL = [
    Path("fake_dataset/first_batch_dataset"),
    Path("fake_dataset/stage2_dataset_working"),
    Path("fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("test_case/problem/benchmark_set_1"),
    Path("test_case/problem/benchmark_set_2"),
    Path("fake_dataset/official_directed_stage1_sanitized_3bugs_85cases"),
]


def resolve(p: Path) -> Path:
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def write_pred(path: Path, cases: Sequence[str], labels: Sequence[int]) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = [f"bucket_{int(x):03d}" for x in labels]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Case", "bucket"]); w.writerows(zip(cases, pred))
    return pred


def labels_for_dataset(dataset: Path) -> list[str]:
    return read_gold(osf.gold_path(dataset))


def dataset_source(dataset: Path) -> str:
    if "test_case/problem" in str(dataset) or osf.gold_path(dataset).name == "golden.csv":
        return "official"
    if "sanitized" in dataset.name:
        return "sanitized"
    return "fake"


def sample_within_dataset_pairs(
    features: list[plf.LLMCaseFeature],
    labels: list[str],
    offset: int,
    args: argparse.Namespace,
    seed: int,
) -> tuple[list[tuple[int, int]], np.ndarray, dict]:
    local_pairs, y, stats = tpl.sample_pairs(
        features,
        labels,
        negative_ratio=args.negative_ratio,
        hard_negative_ratio=args.hard_negative_ratio,
        hard_positive_ratio=args.hard_positive_ratio,
        max_train_pairs=args.max_pairs_per_dataset,
        random_state=seed,
        positive_sampling=args.positive_sampling,
        negative_sampling=args.negative_sampling,
    )
    return [(i + offset, j + offset) for i, j in local_pairs], y, stats


def train_one(args: argparse.Namespace, train_datasets: list[Path], tag: str, seed: int) -> dict:
    t0 = time.perf_counter()
    input_csvs = [ds / "input.csv" for ds in train_datasets]
    llm_args = plf._make_llm_args(
        llm_mode="embedding" if not args.no_llm else "none",
        llm_doc_style="features",
        llm_cache_dir=args.llm_cache_dir,
        svd_dim=args.svd_dim,
        llm_dual=args.feature_mode in plf.DUAL_FEATURE_MODES,
    )
    features, _bundle = plf.build_llm_case_features_for_inputs(input_csvs, svd_dim=args.svd_dim, llm_args=llm_args)
    offsets = []
    labels_by_ds = []
    offset = 0
    for ds in train_datasets:
        labels = labels_for_dataset(ds)
        labels_by_ds.append(labels)
        offsets.append(offset)
        offset += len(labels)
    if offset != len(features):
        raise RuntimeError(f"feature/label mismatch: features={len(features)} labels={offset}")

    llm_reducer = None; llm_summary_reducer = None
    if args.feature_mode in plf.DUAL_FEATURE_MODES and args.llm_reduce_dim > 0:
        llm_reducer = plf.fit_llm_reducer(features, args.llm_reduce_dim, random_state=seed)
        llm_summary_reducer = plf.fit_llm_summary_reducer(features, args.llm_reduce_dim, random_state=seed)
        print(f"[features] llm_reduce_dim={args.llm_reduce_dim}", flush=True)

    all_pairs: list[tuple[int, int]] = []
    ys = []
    sample_weights = []
    pair_stats = []
    rng_seed = seed * 1009 + 17
    for ds, labels, off in zip(train_datasets, labels_by_ds, offsets):
        local_features = features[off:off + len(labels)]
        pairs, y, stats = sample_within_dataset_pairs(local_features, labels, off, args, rng_seed + off)
        src = dataset_source(ds)
        weight = args.official_pair_weight if src == "official" else args.fake_pair_weight
        all_pairs.extend(pairs)
        ys.append(y)
        sample_weights.append(np.full(len(y), float(weight), dtype=np.float32))
        stats.update({"dataset": ds.name, "source": src, "pairs": len(y), "weight": weight})
        pair_stats.append(stats)
    y_all = np.concatenate(ys).astype(np.float32)
    w_all = np.concatenate(sample_weights).astype(np.float32)
    order = np.arange(len(all_pairs))
    np.random.default_rng(seed).shuffle(order)
    all_pairs = [all_pairs[int(i)] for i in order]
    y_all = y_all[order]
    w_all = w_all[order]
    if args.max_train_pairs > 0 and len(all_pairs) > args.max_train_pairs:
        keep = np.random.default_rng(seed + 99).choice(np.arange(len(all_pairs)), size=args.max_train_pairs, replace=False)
        all_pairs = [all_pairs[int(i)] for i in keep]
        y_all = y_all[keep]
        w_all = w_all[keep]

    X = plf.build_rich_pair_feature_matrix(features, all_pairs, feature_mode=args.feature_mode)
    print(f"[train] tag={tag} model={args.model_type} X={X.shape} pos={int(y_all.sum())} neg={int((y_all==0).sum())}", flush=True)

    if args.model_type == "logistic":
        model_pkg = plf.train_logistic_model(X, y_all, random_state=seed)
    elif args.model_type == "gbdt":
        model_pkg = plf.train_gbdt_model(X, y_all, random_state=seed)
    elif args.model_type == "mlp":
        # Current plf MLP has no sample_weight; official weighting is via sampling/duplication for MLP.
        if not np.allclose(w_all, 1.0):
            reps = np.clip(np.rint(w_all), 1, 20).astype(int)
            idx = np.repeat(np.arange(len(y_all)), reps)
            X_fit = X[idx]; y_fit = y_all[idx]
        else:
            X_fit = X; y_fit = y_all
        model_pkg = plf.train_mlp_model(
            X_fit, y_fit, input_dim=X.shape[1], hidden_dims=args.hidden_dims,
            dropout=args.dropout, batch_size=args.batch_size, epochs=args.epochs,
            lr=args.lr, weight_decay=args.weight_decay, device=args.device,
            random_state=seed, mlp_arch=args.mlp_arch, loss=args.loss,
            focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha,
            early_stop_patience=args.early_stop_patience, layernorm=True,
        )
    else:
        raise ValueError(args.model_type)
    model_pkg.update({
        "feature_mode": args.feature_mode,
        "llm_reduce_dim": args.llm_reduce_dim,
        "llm_reducer": llm_reducer,
        "llm_summary_reducer": llm_summary_reducer,
        "svd_dim": args.svd_dim,
        "train_datasets": [str(ds) for ds in train_datasets],
        "model_type": args.model_type,
    })
    model_dir = args.output_dir / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    ext = "pt" if args.model_type == "mlp" else "pkl"
    model_path = model_dir / f"{tag}_seed{seed}_{args.model_type}.{ext}"
    plf.save_model_pkg(model_pkg, model_path)
    return {"model_pkg": model_pkg, "model_path": model_path, "pair_stats": pair_stats, "train_time_sec": time.perf_counter() - t0}


def evaluate_model(args: argparse.Namespace, model_pkg: dict, model_path: Path, eval_datasets: list[Path], tag: str, seed: int) -> list[dict]:
    rows = []
    llm_args = plf._make_llm_args(
        llm_mode="embedding" if not args.no_llm else "none",
        llm_doc_style="features",
        llm_cache_dir=args.llm_cache_dir,
        svd_dim=args.svd_dim,
        llm_dual=args.feature_mode in plf.DUAL_FEATURE_MODES,
    )
    for ds in eval_datasets:
        t0 = time.perf_counter()
        input_csv = ds / "input.csv"
        features, _bundle = plf.build_llm_case_features(input_csv, svd_dim=args.svd_dim, llm_args=llm_args)
        prob_model = plf.predict_probability_matrix_sklearn(model_pkg, features, batch_size=args.predict_batch_size)
        for alpha in args.blend_alphas:
            if alpha < 1.0:
                try:
                    prob_base, *_ = __import__('run_official_directed_trace_eval').build_current_best_probability(args, input_csv)
                except Exception:
                    prob_base = None
                if prob_base is None:
                    prob = prob_model
                    note = "base unavailable; model only"
                else:
                    prob = float(alpha) * prob_model + (1.0 - float(alpha)) * prob_base
                    note = f"blend alpha={alpha} with current no-trace base"
            else:
                prob = prob_model; note = "model only"
            labels = plf.cluster_from_probability(prob.astype(np.float32), len(set(labels_for_dataset(ds))))
            pred_path = args.output_dir / "preds" / f"{tag}_{args.model_type}_seed{seed}_{ds.name}_a{alpha:.2f}.csv"
            prob_path = args.output_dir / "probs" / f"{tag}_{args.model_type}_seed{seed}_{ds.name}_a{alpha:.2f}.npy"
            pred = write_pred(pred_path, osf.read_cases(input_csv), labels)
            prob_path.parent.mkdir(parents=True, exist_ok=True); np.save(prob_path, prob.astype(np.float32))
            gold = labels_for_dataset(ds)
            ba, tpr, tnr = pairwise_scores(gold, pred)
            rows.append({
                "tag": tag, "seed": seed, "train_datasets": "+".join(Path(x).name for x in model_pkg.get("train_datasets", [])),
                "eval_dataset": ds.name, "source": dataset_source(ds), "model_type": args.model_type,
                "feature_mode": args.feature_mode, "alpha": alpha, "BA": ba, "TPR": tpr, "TNR": tnr,
                "k": len(set(gold)), "cases": len(gold), "num_pred_clusters": len(set(pred)),
                "runtime_sec": time.perf_counter() - t0, "model_path": str(model_path),
                "pred_path": str(pred_path), "prob_path": str(prob_path), "notes": note,
            })
            print(f"[eval] tag={tag} ds={ds.name} alpha={alpha:.2f} BA={ba:.4f} TPR={tpr:.4f} TNR={tnr:.4f}", flush=True)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r['tag'], r['model_type'], r['alpha'])].append(r)
    out = []
    for (tag, mt, alpha), rs in sorted(groups.items()):
        def mean(vals): return float(np.mean(list(vals))) if vals else float('nan')
        out.append({
            "tag": tag, "model_type": mt, "alpha": alpha,
            "mean_BA": mean(r['BA'] for r in rs), "min_BA": min(r['BA'] for r in rs),
            "official_mean_BA": mean(r['BA'] for r in rs if r['source']=='official'),
            "fake_mean_BA": mean(r['BA'] for r in rs if r['source']=='fake'),
            "sanitized_BA": mean(r['BA'] for r in rs if r['source']=='sanitized'),
            "num_eval": len(rs),
        })
    return sorted(out, key=lambda r: (r['min_BA'], r['mean_BA']), reverse=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Official-tuned full rich pairwise retraining experiments")
    p.add_argument("--train-datasets", nargs="+", type=Path, required=True)
    p.add_argument("--eval-datasets", nargs="+", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tag", default="official_full_retrain")
    p.add_argument("--model-type", choices=("logistic","gbdt","mlp"), default="mlp")
    p.add_argument("--feature-mode", default="llm_dual_struct_det_summary", choices=tuple(plf.FEATURE_MODES))
    p.add_argument("--llm-reduce-dim", type=int, default=64)
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("auto","cpu","cuda"), default="cuda")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--hidden-dims", nargs="+", type=int, default=None)
    p.add_argument("--mlp-arch", choices=("shallow","deep","residual"), default="residual")
    p.add_argument("--loss", choices=("bce","focal"), default="focal")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", default="auto")
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--negative-ratio", type=float, default=3.0)
    p.add_argument("--hard-negative-ratio", type=float, default=0.7)
    p.add_argument("--hard-positive-ratio", type=float, default=0.5)
    p.add_argument("--positive-sampling", choices=("det_low","diverse"), default="diverse")
    p.add_argument("--negative-sampling", choices=("det_high","confusable"), default="confusable")
    p.add_argument("--max-pairs-per-dataset", type=int, default=120000)
    p.add_argument("--max-train-pairs", type=int, default=300000)
    p.add_argument("--official-pair-weight", type=float, default=1.0)
    p.add_argument("--fake-pair-weight", type=float, default=1.0)
    p.add_argument("--blend-alphas", nargs="+", type=float, default=[1.0, 0.75, 0.50])
    p.add_argument("--predict-batch-size", type=int, default=100000)
    # knobs for build_current_best_probability compatibility
    p.add_argument("--rich-model-root", type=Path, default=Path("/tmp/input_signal_5seed_top/models/llm_dual_struct_det_summary_dim64"))
    p.add_argument("--model-tag", default="llm_dual_struct_det_summary_dim64")
    p.add_argument("--ensemble-model-dir", type=Path, default=Path("/tmp/pairwise_llm_exp_full/models"))
    p.add_argument("--alpha", type=float, default=0.88)
    p.add_argument("--rich-temp", type=float, default=1.15)
    p.add_argument("--ensemble-temp", type=float, default=1.00)
    p.add_argument("--reuse-base-probs-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_datasets = [resolve(p) for p in args.train_datasets]
    eval_datasets = [resolve(p) for p in args.eval_datasets]
    trained = train_one(args, train_datasets, args.tag, args.seed)
    rows = evaluate_model(args, trained['model_pkg'], trained['model_path'], eval_datasets, args.tag, args.seed)
    fields = ["tag","seed","train_datasets","eval_dataset","source","model_type","feature_mode","alpha","BA","TPR","TNR","k","cases","num_pred_clusters","runtime_sec","model_path","pred_path","prob_path","notes"]
    write_csv(args.output_dir / "results.csv", rows, fields)
    summary = summarize(rows)
    write_csv(args.output_dir / "summary.csv", summary, ["tag","model_type","alpha","mean_BA","min_BA","official_mean_BA","fake_mean_BA","sanitized_BA","num_eval"])
    (args.output_dir / "pair_stats.json").write_text(json.dumps(trained['pair_stats'], indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print("\n| rank | tag | model | alpha | mean_BA | min_BA | official_mean | fake_mean | sanitized |")
    print("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(summary[:20], 1):
        print(f"| {i} | {r['tag']} | {r['model_type']} | {float(r['alpha']):.2f} | {r['mean_BA']:.6f} | {r['min_BA']:.6f} | {r['official_mean_BA']:.6f} | {r['fake_mean_BA']:.6f} | {r['sanitized_BA']:.6f} |")
    print(f"Results: {args.output_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
