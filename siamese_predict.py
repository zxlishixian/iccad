#!/usr/bin/env python3
"""Self-contained siamese inference: NumPy encoder + k-means, no PyTorch.

Reads input.csv, builds per-case features (LLM + failure signature + test-name
category + trace residual), reduces them with the frozen train-fit reducers,
encodes with a NumPy-converted siamese encoder (5-seed average), clusters with
k-means, and writes Case,bucket.  Falls back to the deterministic baseline on any
failure so a valid CSV is always produced.
"""
from __future__ import annotations

import os as _os

# Single-thread BLAS/LAPACK (OpenBLAS/MKL) before numpy/scipy import, mirroring
# run_siamese_train.py.  The trace builder forks a multiprocessing.Pool; a
# multi-threaded OpenBLAS can deadlock on any BLAS call after that fork.  Inference
# only does one SVD (before the Pool) so it is not currently hit, but this keeps the
# packaged binary fork-safe for the N=3000 benchmarks.
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ[_var] = "1"

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import joblib

import official_style_features as osf
import pairwise_llm_features as plf
import run_graph_multiview_experiments as gm
import theta_trace_features as ttf
import failure_signature as fs

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"


# ---- NumPy siamese encoder (weights exported from PyTorch) ----
def _gelu(x: np.ndarray) -> np.ndarray:
    # Exact GELU (erf) — must match torch.nn.GELU() (approximate='none', the default).
    # The old tanh approximation (4.7e-4 max error) was masked by the case_index leak
    # signal in v7, but after removing the leak it degrades the Procrustes ensemble
    # (0.979 -> 0.733 on set2), so use the exact erf form.
    from scipy.special import erf
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _numpy_forward(x: np.ndarray, w: dict) -> np.ndarray:
    x = x @ w["W0"].T + w["b0"]
    x = _layernorm(x, w["ln0_w"], w["ln0_b"])
    x = _gelu(x)
    x = x @ w["W1"].T + w["b1"]
    x = _layernorm(x, w["ln1_w"], w["ln1_b"])
    x = _gelu(x)
    x = x @ w["W2"].T + w["b2"]
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return x


def _load_models(model_dir: Path) -> list[tuple[dict, Any]]:
    """Load (encoder_weights, preprocess) pairs, one per seed, ordered by seed.

    The LLM reducer and trace bundle are fit with ``random_state=seed`` during
    training, so each seed's encoder must be paired with its OWN reducer — using
    seed 0's reducer for seeds 1..N silently rotates their features and collapses
    the ensemble (set2 0.979 -> 0.733).  See handoff pitfall #39.
    """
    models = []
    for p in sorted(model_dir.glob("encoder_seed*.npz")):
        z = np.load(p)
        enc = {k: z[k] for k in z.files}
        suffix = p.stem[len("encoder_seed"):]  # "0", "1", ...
        prep_path = model_dir / f"preprocess_seed{suffix}.pkl"
        if not prep_path.is_file():
            prep_path = model_dir / "preprocess.pkl"  # legacy single-seed fallback
        models.append((enc, joblib.load(prep_path)))
    return models


# ---- feature building (mirrors run_siamese_train.build_case_matrix) ----
def _signature_features(signatures):
    n_type = 3
    type_idx = {"mismatch": 0, "test_fail": 1, "other": 2}
    sig = np.zeros((len(signatures), len(fs.FAMILY_LIST) + n_type), dtype=np.float32)
    for i, (dtype, opcode, _reg, _pc) in enumerate(signatures):
        sig[i, fs.FAMILY_IDX[fs._family_of(opcode)]] = 1.0
        sig[i, len(fs.FAMILY_LIST) + type_idx.get(dtype, 2)] = 1.0
    return sig


def _test_categories(test_names):
    test_categories = ["csr", "interrupt", "debug", "mmu", "branch", "jump", "machine",
                       "mul", "div", "shift", "load", "store", "fence", "illegal"]
    tmat = np.zeros((len(test_names), len(test_categories)), dtype=np.float32)
    for i, tn in enumerate(test_names):
        tl = tn.lower()
        for j, cat in enumerate(test_categories):
            if cat in tl:
                tmat[i, j] = 1.0
    return tmat


def _fatal_char_ngram(messages, dim: int = 128):
    """Deterministic char n-gram of the sim.log UVM failure message (mirrors run_siamese_train)."""
    import re
    feats = np.zeros((len(messages), dim), dtype=np.float32)
    for i, msg in enumerate(messages):
        norm = re.sub(r"\d+", "N", str(msg))
        for ch in norm:
            code = ord(ch)
            if code < dim:
                feats[i, code] += 1.0
        s = feats[i].sum()
        if s > 0:
            feats[i] = feats[i] / s
    return feats


def _build_base(args, input_csv: Path):
    """Build seed-independent raw features (the expensive part: reads logs, LLM, trace)."""
    llm_args = gm.make_embedding_args(args)
    ep, _ = plf.build_llm_case_features_for_inputs(
        [input_csv], parser="drain", svd_dim=64, llm_args=llm_args
    )
    sig = fs.extract_rich_signatures(input_csv)
    names = fs.extract_test_names(input_csv)
    cases = osf.read_cases(input_csv)
    n = len(ep)
    llm_raw = np.stack([f.llm_vec for f in ep]).astype(np.float32)
    sig_mat = _signature_features(sig)
    test_mat = _test_categories(names)
    fatal_char_mat = _fatal_char_ngram(fs.extract_sim_failure_messages(input_csv))
    tr, _ = ttf.build_hierarchical_trace_features(
        input_csv, cache_dir=Path("/tmp/theta_trilog_trace_cache"),
        segment_count=16, chunk_size=512, anchor_sizes=[32, 64, 128],
    )
    anchor_mat = np.stack([f.anchor_struct for f in tr]).astype(np.float32)
    return llm_raw, sig_mat, test_mat, fatal_char_mat, anchor_mat, tr, cases, n


def _reduce_matrix(llm_raw, sig_mat, test_mat, fatal_char_mat, anchor_mat, tr, pre, view_dim):
    """Apply ONE seed's reducers to the raw base -> 364-dim matrix (cheap)."""
    llm_mat = plf._apply_reducer_to_matrix(llm_raw, pre["reducer"], view_dim).astype(np.float32)
    tm = ttf.apply_trace_reducers(pre["trace_bundle"], tr)
    residual = np.hstack([tm["residual_struct"], tm["residual_text"]]).astype(np.float32)
    return np.nan_to_num(np.hstack([llm_mat, sig_mat, test_mat, fatal_char_mat, anchor_mat, residual]), nan=0.0).astype(np.float32)


def _cluster_kmeans(emb: np.ndarray, k: int, seed: int = 0) -> list[int]:
    from sklearn.cluster import KMeans
    n = emb.shape[0]
    k = max(1, min(int(k), n))
    if k == 1:
        return [0] * n
    if k == n:
        return list(range(n))
    return [int(x) for x in KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(emb)]


def _procrustes_cluster(emb_list: list[np.ndarray], k: int) -> list[int]:
    """Procrustes-aligned embedding average -> k-means (handoff pitfall #15/#26).

    The per-seed encoders live in mutually-rotated embedding spaces, so neither the
    naive mean nor pairwise co-association is reliable.  Align seeds 1..N to seed 0
    via orthogonal Procrustes, average the aligned embeddings, then k-means.  Wins
    big on the official dev set set2 (0.727 -> 0.962) and batch4, at the cost of
    small regressions on some lui-cascade fake sets.
    """
    n = emb_list[0].shape[0]
    k = max(1, min(int(k), n))
    if k == 1:
        return [0] * n
    if k == n:
        return list(range(n))
    ref = emb_list[0]
    aligned = [ref]
    for emb in emb_list[1:]:
        u, _, vt = np.linalg.svd(emb.T @ ref)
        aligned.append(emb @ (u @ vt))
    avg = np.mean(np.stack(aligned), axis=0)
    return _cluster_kmeans(avg, k)


def run_siamese(args) -> int:
    models = _load_models(MODEL_DIR)
    if not models:
        raise RuntimeError("no encoder npz found")
    llm_raw, sig_mat, test_mat, fatal_char_mat, anchor_mat, tr, cases, n = _build_base(args, args.input)
    emb_list = []
    for enc, pre in models:
        matrix = _reduce_matrix(llm_raw, sig_mat, test_mat, fatal_char_mat, anchor_mat, tr, pre, args.view_dim)
        emb_list.append(_numpy_forward(matrix, enc))
    labels = _procrustes_cluster(emb_list, args.k)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case", "bucket"])
        for case, lab in zip(cases, labels):
            w.writerow([case, f"bucket_{int(lab):03d}"])
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--view-dim", type=int, default=64)
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--llm-doc-max-features", type=int, default=80)
    # Large batches + short per-batch timeout: official eval counts network latency
    # toward the runtime limit, and excessive API calls can zero the benchmark.
    # benchmark6 (3000 cases) needs ~6 calls at 512 instead of ~94 at 64.
    p.add_argument("--llm-batch-size", type=int, default=512)
    p.add_argument("--llm-timeout-sec", type=float, default=20.0)
    p.add_argument("--embedding-expected-dim", type=int, default=768)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return run_siamese(args)
    except Exception as exc:  # noqa: BLE001 - deterministic fallback keeps a valid CSV
        import regr_fail_bucketing as rfb
        print(f"[siamese] failed ({exc}); falling back to deterministic baseline", file=sys.stderr)
        return rfb.main(["--input", str(args.input), "--output", str(args.output), "--k", str(args.k)])


if __name__ == "__main__":
    raise SystemExit(main())
