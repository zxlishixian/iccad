#!/usr/bin/env python3
"""Procrustes-aligned 5-seed ensemble eval for the siamese model.

Loads N seed encoders (+ their train-fit reducers), encodes each eval dataset
with every encoder, aligns seeds 1..N to seed 0 via orthogonal Procrustes (SVD),
averages the aligned embeddings, then clusters once.  This is the ensemble used
by `siamese_predict.py` (handoff pitfall #27): it wins big on the official dev
set set2 vs the naive embedding mean.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import joblib

import theta_siamese_model as tsm
import run_siamese_train as rst
from run_experiments import pairwise_scores


def load_seeds(seed_dirs: list[Path]):
    encoders, preps = [], []
    for d in seed_dirs:
        ckpt = torch.load(d / "encoder.pt", map_location="cpu")
        # The encoder may have been trained with the trace-sequence branch (use_seq=True);
        # detect it by checking whether the state dict has seq layers.
        use_seq = any(k.startswith("seq_enc.") for k in ckpt["state_dict"])
        enc = tsm.SiameseEncoder(int(ckpt["input_dim"]), use_seq=use_seq)
        enc.load_state_dict(ckpt["state_dict"])
        encoders.append(enc.eval())
        preps.append(joblib.load(d / "preprocess.pkl"))
    return encoders, preps


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Siamese Procrustes ensemble eval")
    p.add_argument("--seed-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--eval-datasets", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
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
    p.add_argument("--use-fatal-llm", action="store_true", default=False)
    return p.parse_args(argv)


def _procrustes_cluster(emb_list: list[np.ndarray], k: int) -> np.ndarray:
    """Align emb_list[1:] to emb_list[0] via orthogonal Procrustes, average, k-means."""
    n = emb_list[0].shape[0]
    k = max(1, min(int(k), n))
    if k == 1:
        return np.zeros(n, dtype=np.int64)
    ref = emb_list[0]
    aligned = [ref]
    for emb in emb_list[1:]:
        u, _, vt = np.linalg.svd(emb.T @ ref)
        aligned.append(emb @ (u @ vt))
    avg = np.mean(np.stack(aligned), axis=0)
    return tsm.cluster_embeddings(avg, k, random_state=0)


def main(argv=None):
    args = parse_args(argv)
    args.seed = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoders, preps = load_seeds([Path(d) for d in args.seed_dirs])

    rows = []
    for ds in [Path(d) for d in args.eval_datasets]:
        emb_list = []
        gold = None
        k = 0
        for enc, pre in zip(encoders, preps):
            cm, case_labels, _, _, _, _, _, _, case_seq = rst.build_case_matrix(
                args, [ds], reducer=pre["reducer"], trace_bundle=pre["trace_bundle"],
                snippet_reducer=pre["snippet_reducer"], test_name_vocab=pre.get("test_name_vocab"),
                fatal_reducer=pre.get("fatal_reducer"),
            )
            emb_list.append(tsm.encode_cases(enc, cm, case_seq=case_seq))
            gold = [str(b) for b in case_labels]
            k = len(set(gold))
        labels = _procrustes_cluster(emb_list, k)
        pred = [f"bucket_{int(x):03d}" for x in labels]
        ba, tpr, tnr = pairwise_scores(gold, pred)
        rows.append({"dataset": ds.name, "BA": ba, "TPR": tpr, "TNR": tnr, "k": k, "cases": len(gold), "clusters": len(set(pred))})
        print(f"[procrustes] {ds.name} BA={ba:.4f} TPR={tpr:.4f} TNR={tnr:.4f} clusters={len(set(pred))}/{k}", flush=True)

    with open(args.output_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "BA", "TPR", "TNR", "k", "cases", "clusters"])
        w.writeheader(); w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
