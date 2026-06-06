#!/usr/bin/env python3
"""ICCAD 2026 Problem B alpha submission entrypoint.

Primary route:
  dual LLM embeddings -> calibrated rich/ensemble pair probabilities
  -> official-style root-cause adapter -> agglomerative clustering

The prediction path reads only input.csv and its referenced log files. It does
not read gold.csv, golden.csv, meta.csv, or trace.log.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

import official_style_features as osf
import pairwise_llm_features as plf
import regr_fail_bucketing as rfb


RICH_TAG = "llm_dual_struct_det_summary_dim64"
SEEDS = tuple(range(10))
ENSEMBLE_WEIGHTS = (0.20, 0.40, 0.40)
RICH_TEMP = 1.15
ENSEMBLE_TEMP = 1.00
RICH_ALPHA = 0.88
ADAPTER_ALPHA = 0.50


def package_root() -> Path:
    bundled = getattr(sys, "_MEIPASS", None)
    return Path(bundled) if bundled else Path(__file__).resolve().parent


def temperature(prob: np.ndarray, value: float) -> np.ndarray:
    if abs(float(value) - 1.0) < 1e-9:
        return prob.astype(np.float32, copy=False)
    clipped = np.clip(prob.astype(np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)) / float(value)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def ensemble_paths(root: Path, seed: int) -> list[Path]:
    suffixes = (
        f"model_seed{seed}_combo000_logistic.pkl",
        f"model_seed{seed}_combo000_gbdt.pkl",
        (
            f"model_seed{seed}_combo000_mlp.pt"
            if seed < 10
            else f"model_seed{seed}_combo000_mlp_summary21_shallow_bce.pt"
        ),
    )
    return [root / "models" / "ensemble" / name for name in suffixes]


def rich_path(root: Path, seed: int) -> Path:
    return (
        root
        / "models"
        / "rich"
        / f"model_seed{seed}_combo000_{RICH_TAG}.pt"
    )


def build_base_probability(input_csv: Path, root: Path) -> tuple[np.ndarray, int]:
    cache_dir = Path(os.getenv("REGR_FAIL_EMBED_CACHE", "/tmp/regr_fail_llm_cache"))
    rich_args = plf._make_llm_args(
        llm_mode="embedding",
        llm_doc_style="features",
        llm_cache_dir=cache_dir,
        svd_dim=64,
        llm_dual=True,
    )
    ensemble_args = plf._make_llm_args(
        llm_mode="embedding",
        llm_doc_style="features",
        llm_cache_dir=cache_dir,
        svd_dim=64,
        llm_dual=False,
    )
    rich_features, _ = plf.build_llm_case_features(
        input_csv, svd_dim=64, llm_args=rich_args
    )
    ensemble_features, _ = plf.build_llm_case_features(
        input_csv, svd_dim=64, llm_args=ensemble_args
    )

    total = np.zeros((len(rich_features), len(rich_features)), dtype=np.float64)
    used = 0
    for seed in SEEDS:
        rpath = rich_path(root, seed)
        epaths = ensemble_paths(root, seed)
        if not rpath.is_file() or not all(path.is_file() for path in epaths):
            print(f"[submission] seed={seed} artifacts missing; skipped", file=sys.stderr)
            continue
        rich_pkg = plf.load_model_pkg(rpath)
        ensemble_pkgs = [plf.load_model_pkg(path) for path in epaths]
        p_rich = plf.predict_probability_matrix_sklearn(
            rich_pkg, rich_features, batch_size=100000
        )
        p_ensemble = plf.predict_probability_matrix_ensemble(
            ensemble_pkgs,
            list(ENSEMBLE_WEIGHTS),
            ensemble_features,
            ensemble_mode="prob_average",
            batch_size=100000,
        )
        combined = (
            RICH_ALPHA * temperature(p_rich, RICH_TEMP)
            + (1.0 - RICH_ALPHA) * temperature(p_ensemble, ENSEMBLE_TEMP)
        )
        total += combined.astype(np.float64)
        used += 1
        print(f"[submission] seed={seed} loaded", file=sys.stderr)
    if used == 0:
        raise RuntimeError("no usable calibrated-blend model artifacts")
    prob = (total / float(used)).astype(np.float32)
    np.fill_diagonal(prob, 1.0)
    return prob, used


def apply_adapter(input_csv: Path, base_prob: np.ndarray, root: Path) -> np.ndarray:
    model_path = root / "models" / "adapter" / "official_style_tags_logistic.pkl"
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    records = osf.build_case_records("prediction", input_csv, gold_csv=None)
    pairs = osf.all_pairs(len(records))
    X = osf.build_pair_feature_matrix(
        records,
        pairs,
        base_prob,
        include_graph=False,
        include_anchor=False,
    )
    scores = model.predict_proba(X)[:, 1].astype(np.float32)
    adapter_prob = np.eye(len(records), dtype=np.float32)
    for (i, j), score in zip(pairs, scores):
        adapter_prob[i, j] = adapter_prob[j, i] = float(score)
    final = ADAPTER_ALPHA * adapter_prob + (1.0 - ADAPTER_ALPHA) * base_prob
    np.fill_diagonal(final, 1.0)
    return final.astype(np.float32)


def write_prediction(input_csv: Path, output_csv: Path, labels: list[int]) -> None:
    rows, fields = rfb.read_csv_rows(input_csv)
    cases = rfb.output_case_values(rows, fields)
    rfb.write_output(output_csv, rfb.remap_labels(labels), cases)


def run_fallback(input_csv: Path, output_csv: Path, k: int) -> int:
    print("[submission] using deterministic no-trace fallback", file=sys.stderr)
    return rfb.main(
        [
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
            "--k",
            str(k),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ICCAD regression failure bucketing")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(0)
    np.random.seed(0)
    os.environ["PYTHONHASHSEED"] = "0"
    started = time.perf_counter()
    input_csv = args.input.resolve()
    output_csv = args.output.resolve()
    try:
        root = package_root()
        base_prob, used = build_base_probability(input_csv, root)
        final_prob = apply_adapter(input_csv, base_prob, root)
        labels = plf.cluster_from_probability(final_prob, args.k)
        write_prediction(input_csv, output_csv, labels)
        print(
            f"[submission] method=calibrated_dual_blend+official_tags "
            f"seeds={used} cases={len(labels)} buckets={len(set(labels))} "
            f"runtime_sec={time.perf_counter() - started:.3f}",
            file=sys.stderr,
        )
        return 0
    except Exception as exc:
        print(f"WARNING: primary model failed: {exc}", file=sys.stderr)
        return run_fallback(input_csv, output_csv, args.k)


if __name__ == "__main__":
    raise SystemExit(main())
