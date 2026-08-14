#!/usr/bin/env python3
"""Ensemble eval for the siamese model: average per-seed embeddings, cluster once.

Loads N seed encoders (+ their train-fit reducers), encodes each eval dataset
with every encoder, averages the embeddings, then clusters once.  This reduces
the small-dataset variance of independent per-seed k-means runs.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import joblib

import official_style_features as osf
import theta_siamese_model as tsm
import run_siamese_train as rst
from run_experiments import pairwise_scores


def load_seeds(seed_dirs: list[Path]):
    encoders = []
    preps = []
    for d in seed_dirs:
        ckpt = torch.load(d / "encoder.pt", map_location="cpu")
        enc = tsm.SiameseEncoder(int(ckpt["input_dim"]))
        enc.load_state_dict(ckpt["state_dict"])
        encoders.append(enc.eval())
        preps.append(joblib.load(d / "preprocess.pkl"))
    return encoders, preps


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Siamese ensemble eval")
    p.add_argument("--seed-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--eval-datasets", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    # feature-building args (must match training)
    p.add_argument("--parser", default="drain")
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--view-dim", type=int, default=64)
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--llm-doc-max-features", type=int, default=80)
    p.add_argument("--llm-batch-size", type=int, default=64)
    p.add_argument("--llm-timeout-sec", type=float, default=60.0)
    p.add_argument("--embedding-expected-dim", type=int, default=768)
    p.add_argument("--trace-cache-dir", type=Path, default=Path("/tmp/theta_trilog_trace_cache"))
    p.add_argument("--trace-segment-count", type=int, default=16)
    p.add_argument("--trace-chunk-size", type=int, default=512)
    p.add_argument("--trace-anchor-sizes", nargs="+", type=int, default=[32, 64, 128])
    p.add_argument("--use-trace", action="store_true", default=False)
    p.add_argument("--use-mismatch-llm", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.seed = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_dirs = [Path(d) for d in args.seed_dirs]
    encoders, preps = load_seeds(seed_dirs)
    eval_datasets = [Path(d) for d in args.eval_datasets]

    rows = []
    for ds in eval_datasets:
        emb_list = []
        gold = None
        k = 0
        for enc, pre in zip(encoders, preps):
            cm, case_labels, _, _, _, _ = rst.build_case_matrix(
                args, [ds], reducer=pre["reducer"], trace_bundle=pre["trace_bundle"],
                snippet_reducer=pre["snippet_reducer"], test_name_vocab=pre.get("test_name_vocab"),
            )
            emb = tsm.encode_cases(enc, cm)
            emb_list.append(emb)
            gold = [str(b) for b in case_labels]
            k = len(set(gold))
        avg_emb = np.mean(np.stack(emb_list), axis=0)
        labels = tsm.cluster_embeddings(avg_emb, k, random_state=0)
        pred = [f"bucket_{int(x):03d}" for x in labels]
        ba, tpr, tnr = pairwise_scores(gold, pred)
        rows.append({"dataset": ds.name, "BA": ba, "TPR": tpr, "TNR": tnr, "k": k, "cases": len(gold), "clusters": len(set(pred))})
        print(f"[ensemble] {ds.name} BA={ba:.4f} TPR={tpr:.4f} TNR={tnr:.4f} clusters={len(set(pred))}/{k}", flush=True)

    with open(args.output_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "BA", "TPR", "TNR", "k", "cases", "clusters"])
        w.writeheader(); w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
