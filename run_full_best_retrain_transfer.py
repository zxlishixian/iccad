#!/usr/bin/env python3
"""Retrain the complete calibrated dual-input blend on selected datasets.

Experimental only. Training may read gold/golden labels. Evaluation labels are
used only for scoring. The formal regr_fail_bucketing.py path is unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

import official_style_features as osf
import pairwise_llm_features as plf
from run_experiments import pairwise_scores, read_gold
from run_input_signal_calibration import _temperature
from run_official_full_retrain_experiments import resolve, train_one, write_csv, write_pred

ENSEMBLE_TYPES = ("logistic", "gbdt", "mlp")
ENSEMBLE_WEIGHTS = (0.20, 0.40, 0.40)


def clone_args(args: argparse.Namespace, **updates) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def train_complete(args: argparse.Namespace, train_datasets: list[Path]) -> dict:
    rich_args = clone_args(
        args,
        model_type="mlp",
        feature_mode="llm_dual_struct_det_summary",
        llm_reduce_dim=64,
        mlp_arch="residual",
        loss="focal",
        epochs=args.rich_epochs,
        batch_size=args.rich_batch_size,
        negative_ratio=args.rich_negative_ratio,
        hard_negative_ratio=args.rich_hard_negative_ratio,
        hard_positive_ratio=args.rich_hard_positive_ratio,
        positive_sampling="diverse",
        negative_sampling="confusable",
    )
    rich = train_one(rich_args, train_datasets, f"{args.tag}_rich", args.seed)

    ensemble = []
    for model_type in ENSEMBLE_TYPES:
        ens_args = clone_args(
            args,
            model_type=model_type,
            feature_mode="summary21",
            llm_reduce_dim=0,
            mlp_arch="shallow",
            loss="bce",
            epochs=args.ensemble_epochs,
            batch_size=args.ensemble_batch_size,
            negative_ratio=2.0,
            hard_negative_ratio=0.5,
            hard_positive_ratio=0.5,
            positive_sampling="det_low",
            negative_sampling="det_high",
        )
        trained = train_one(ens_args, train_datasets, f"{args.tag}_ensemble_{model_type}", args.seed)
        ensemble.append(trained)
    return {"rich": rich, "ensemble": ensemble}


def evaluate(args: argparse.Namespace, trained: dict, eval_datasets: list[Path]) -> list[dict]:
    rich_pkg = trained["rich"]["model_pkg"]
    ensemble_pkgs = [item["model_pkg"] for item in trained["ensemble"]]
    rich_llm_args = plf._make_llm_args(
        llm_mode="embedding", llm_doc_style="features", llm_cache_dir=args.llm_cache_dir,
        svd_dim=args.svd_dim, llm_dual=True,
    )
    ensemble_llm_args = plf._make_llm_args(
        llm_mode="embedding", llm_doc_style="features", llm_cache_dir=args.llm_cache_dir,
        svd_dim=args.svd_dim, llm_dual=False,
    )
    rows = []
    for ds in eval_datasets:
        input_csv = ds / "input.csv"
        gold_csv = osf.gold_path(ds)
        gold = read_gold(gold_csv)
        k = len(set(gold))
        cases = osf.read_cases(input_csv)

        rich_features, _ = plf.build_llm_case_features(input_csv, svd_dim=args.svd_dim, llm_args=rich_llm_args)
        p_rich_raw = plf.predict_probability_matrix_sklearn(rich_pkg, rich_features, batch_size=args.predict_batch_size)
        ensemble_features, _ = plf.build_llm_case_features(input_csv, svd_dim=args.svd_dim, llm_args=ensemble_llm_args)
        p_ensemble_raw = plf.predict_probability_matrix_ensemble(
            ensemble_pkgs, list(ENSEMBLE_WEIGHTS), ensemble_features,
            ensemble_mode="prob_average", batch_size=args.predict_batch_size,
        )
        p_rich = _temperature(p_rich_raw, args.rich_temp)
        p_ensemble = _temperature(p_ensemble_raw, args.ensemble_temp)
        p_final = args.alpha * p_rich + (1.0 - args.alpha) * p_ensemble
        np.fill_diagonal(p_final, 1.0)

        for method, prob in (
            ("rich", p_rich_raw),
            ("ensemble", p_ensemble_raw),
            ("calibrated_blend", p_final),
        ):
            labels = plf.cluster_from_probability(prob.astype(np.float32), k)
            pred_path = args.output_dir / "preds" / f"{ds.name}_{method}_seed{args.seed}.csv"
            prob_path = args.output_dir / "probs" / f"{ds.name}_{method}_seed{args.seed}.npy"
            pred = write_pred(pred_path, cases, labels)
            prob_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(prob_path, prob.astype(np.float32))
            ba, tpr, tnr = pairwise_scores(gold, pred)
            row = {
                "seed": args.seed, "dataset": ds.name, "method": method,
                "BA": ba, "TPR": tpr, "TNR": tnr, "k": k, "cases": len(gold),
                "num_pred_clusters": len(set(pred)), "alpha": args.alpha,
                "rich_temp": args.rich_temp, "ensemble_temp": args.ensemble_temp,
                "pred_path": str(pred_path), "prob_path": str(prob_path),
            }
            rows.append(row)
            print(f"[eval] seed={args.seed} dataset={ds.name} method={method} BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f}", flush=True)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain complete calibrated dual-input blend")
    p.add_argument("--train-datasets", nargs="+", type=Path, required=True)
    p.add_argument("--eval-datasets", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tag", default="full_best_retrain")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--predict-batch-size", type=int, default=100000)
    p.add_argument("--alpha", type=float, default=0.88)
    p.add_argument("--rich-temp", type=float, default=1.15)
    p.add_argument("--ensemble-temp", type=float, default=1.00)
    p.add_argument("--rich-epochs", type=int, default=40)
    p.add_argument("--ensemble-epochs", type=int, default=40)
    p.add_argument("--rich-batch-size", type=int, default=4096)
    p.add_argument("--ensemble-batch-size", type=int, default=4096)
    p.add_argument("--rich-negative-ratio", type=float, default=4.0)
    p.add_argument("--rich-hard-negative-ratio", type=float, default=0.8)
    p.add_argument("--rich-hard-positive-ratio", type=float, default=0.5)
    p.add_argument("--max-pairs-per-dataset", type=int, default=120000)
    p.add_argument("--max-train-pairs", type=int, default=300000)
    p.add_argument("--official-pair-weight", type=float, default=1.0)
    p.add_argument("--fake-pair-weight", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--hidden-dims", nargs="+", type=int, default=None)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", default="auto")
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--no-llm", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_datasets = [resolve(x) for x in args.train_datasets]
    eval_datasets = [resolve(x) for x in args.eval_datasets]
    trained = train_complete(args, train_datasets)
    rows = evaluate(args, trained, eval_datasets)
    fields = ["seed", "dataset", "method", "BA", "TPR", "TNR", "k", "cases", "num_pred_clusters", "alpha", "rich_temp", "ensemble_temp", "pred_path", "prob_path"]
    write_csv(args.output_dir / "results.csv", rows, fields)
    manifest = {
        "seed": args.seed,
        "train_datasets": [str(x) for x in train_datasets],
        "alpha": args.alpha, "rich_temp": args.rich_temp, "ensemble_temp": args.ensemble_temp,
        "ensemble_weights": list(ENSEMBLE_WEIGHTS),
        "rich_model": str(trained["rich"]["model_path"]),
        "ensemble_models": [str(x["model_path"]) for x in trained["ensemble"]],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
