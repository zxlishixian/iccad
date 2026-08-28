#!/usr/bin/env python3
"""Train a siamese per-case encoder with SupCon + prototype loss.

O(N) inference (encode cases then k-means), unlike the O(N^2) pair model.
Per-case features = LLM embedding (reduced) + failure-signature one-hot.
Trace features are planned as a follow-up addition.
"""
from __future__ import annotations

import os as _os

# Single-thread the BLAS/LAPACK backends (OpenBLAS/MKL) BEFORE numpy/scipy import.
# The trace feature builder forks a multiprocessing.Pool (per dataset); after that
# fork, a multi-threaded OpenBLAS (scipy-openblas MAX_THREADS=64) deadlocks on the
# very next BLAS call (TruncatedSVD -> scipy.linalg.lu -> futex_wait).  The SVD here
# is small (TF-IDF -> 64 dims), so single-threaded BLAS costs nothing but removes the
# fork/thread-pool corruption entirely.
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ[_var] = "1"

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

import official_style_features as osf
import pairwise_llm_features as plf
import run_graph_multiview_experiments as gm
import theta_siamese_model as tsm
import theta_trace_features as ttf
from run_experiments import pairwise_scores, read_gold
from run_official_full_retrain_experiments import resolve, write_csv
from failure_signature import extract_sim_failure_messages
from run_alpha_improved_train import (
    extract_rich_signatures, extract_mismatch_snippets, extract_test_names,
    _family_of, FAMILY_LIST, FAMILY_IDX,
)


def _fatal_char_ngram(messages: Sequence[str], dim: int = 128) -> np.ndarray:
    """Deterministic char-frequency n-gram of the sim.log UVM failure message.

    The sim.log's first UVM_FATAL/ERROR line carries the *why* of a failure
    ('Did not receive core_s...' vs '[ASSERT FAILED]...' vs a bare 'UVM_ERROR'),
    which is a strong, training-free discriminator for the fatal-type bugs
    (handoff §3.7 + the official char_embedding sample).  Digits are collapsed to
    'N' to drop timestamps/line numbers; rows are L1-normalized so message length
    doesn't dominate.
    """
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


def signature_features(signatures: Sequence[tuple[str, str, str, int]]) -> np.ndarray:
    """Per-case failure-signature: functional-unit family (11) + divergence type (3).

    Deliberately excludes the opcode/register one-hots: the first mismatch opcode
    is polluted by initialization instructions (`lui`/`c.li`) on benchmark6 and is
    not discriminative (see handoff.md pitfall #13).  The family groups `lui` with
    other ALU instructions, staying clean, and divergence type (mismatch / test-fail
    / other) captures the bug's manifestation mode.
    """
    n_type = 3
    type_idx = {"mismatch": 0, "test_fail": 1, "other": 2}
    feats = np.zeros((len(signatures), len(FAMILY_LIST) + n_type), dtype=np.float32)
    for i, (dtype, opcode, _register, _pc) in enumerate(signatures):
        feats[i, FAMILY_IDX[_family_of(opcode)]] = 1.0
        feats[i, len(FAMILY_LIST) + type_idx.get(dtype, 2)] = 1.0
    return feats


# Official-Ibex-bug-id datasets share the same bug ids (bug_128 == "bne decoded as beq"
# in both), so their labels must merge across datasets rather than be prefixed apart.
SHARED_BUG_ID_DATASETS = {
    "catalog_global_hardgate_final_20260824",
    "large_expansion_20260824_944_official",
}


def build_case_matrix(args, datasets, reducer=None, trace_bundle=None, snippet_reducer=None, test_name_vocab=None, fatal_reducer=None):
    """Build (n_cases, feature_dim) per-case matrix + integer bug labels.

    ``reducer`` / ``trace_bundle`` / ``snippet_reducer`` / ``test_name_vocab`` /
    ``fatal_reducer`` are fit on TRAIN data; pass them when building eval features
    so train/eval live in the same subspace / vocabulary.
    """
    llm_args = gm.make_embedding_args(args)
    all_features: list[plf.LLMCaseFeature] = []
    all_labels: list[str] = []
    all_signatures: list[tuple[str, str, str, int]] = []
    all_trace: list[ttf.HierarchicalTraceFeature] = []
    all_snippets: list[str] = []
    all_test_names: list[str] = []
    all_fatal: list[str] = []
    all_fatal_msgs: list[str] = []
    all_domains: list[int] = []
    official_names = {"benchmark_set_1", "benchmark_set_2"}
    for dataset in datasets:
        ep, _ = plf.build_llm_case_features_for_inputs(
            [dataset / "input.csv"], parser=args.parser, svd_dim=args.svd_dim, llm_args=llm_args
        )
        gold = read_gold(osf.gold_path(dataset))
        sig = extract_rich_signatures(dataset)
        if not (len(ep) == len(gold) == len(sig)):
            raise RuntimeError(f"mismatch for {dataset.name}")
        # Bug labels collide across datasets ('bug_037' in dataset A != dataset B).
        # Prefix with the dataset name so each bug is globally unique — EXCEPT the two
        # official-Ibex-bug-id datasets (catalog + v4 expansion) which share the SAME
        # bug ids (bug_128 = "bne decoded as beq" in both): those must share ONE label
        # so the same RTL bug's cases cluster together across the two datasets.
        is_official = dataset.name in official_names
        if dataset.name in SHARED_BUG_ID_DATASETS:
            prefix = "ibex::"
        else:
            prefix = f"{dataset.name}::"
        all_features.extend(ep)
        all_labels.extend([f"{prefix}{b}" for b in gold])
        all_signatures.extend(sig)
        all_test_names.extend(extract_test_names(dataset))
        all_domains.extend([1 if is_official else 0] * len(gold))
        if args.use_mismatch_llm:
            all_snippets.extend(extract_mismatch_snippets(dataset))
        if args.use_fatal_llm:
            all_fatal.extend(extract_sim_failure_messages(dataset))
        # deterministic sim.log UVM failure-message feature (always on, cheap)
        all_fatal_msgs.extend(extract_sim_failure_messages(dataset))
        if args.use_trace:
            tr, _ = ttf.build_hierarchical_trace_features(
                dataset / "input.csv", cache_dir=args.trace_cache_dir,
                segment_count=args.trace_segment_count, chunk_size=args.trace_chunk_size,
                anchor_sizes=args.trace_anchor_sizes,
            )
            all_trace.extend(tr)
        print(f"[load] {dataset.name} cases={len(gold)} trace_ok={sum(f.has_trace for f in all_trace[-len(gold):]) if args.use_trace and all_trace else '-'}", flush=True)

    # fit + apply LLM reducer (768 -> view_dim); reuse the train-fit reducer on eval
    if reducer is None:
        reducer = plf.fit_llm_reducer(all_features, args.view_dim, random_state=args.seed)
    plf.apply_llm_reducer(all_features, reducer, args.view_dim)
    llm_mat = np.stack([f.effective_llm_vec for f in all_features]).astype(np.float32)
    sig_mat = signature_features(all_signatures)

    # failing-test-name SEMANTIC category flags (csr/interrupt/debug/... group the
    # functional area; exact one-hot would split 'interrupt_csr' vs 'csr' wrongly).
    test_categories = ["csr", "interrupt", "debug", "mmu", "branch", "jump", "machine",
                       "mul", "div", "shift", "load", "store", "fence", "illegal"]
    test_mat = np.zeros((len(all_test_names), len(test_categories)), dtype=np.float32)
    if not getattr(args, "no_test_name", False):
        for i, tn in enumerate(all_test_names):
            tl = tn.lower()
            for j, cat in enumerate(test_categories):
                if cat in tl:
                    test_mat[i, j] = 1.0

    # LLM embedding of the semantic mismatch snippet (NOT the whole-log boilerplate)
    if args.use_mismatch_llm and all_snippets:
        import regr_fail_bucketing as rfb
        emb, _ = rfb.fetch_llm_embeddings(all_snippets, llm_args)
        snippet_mat = np.asarray(emb, dtype=np.float32)
        if snippet_reducer is None:
            snippet_reducer, snippet_reduced = plf._fit_reducer_for_matrix(
                snippet_mat, args.view_dim, args.seed + 999
            )
        else:
            snippet_reduced = plf._apply_reducer_to_matrix(snippet_mat, snippet_reducer, args.view_dim)
        snippet_reduced = np.nan_to_num(np.asarray(snippet_reduced, dtype=np.float32), nan=0.0)
    else:
        snippet_reduced = np.zeros((len(all_features), 0), dtype=np.float32)

    # LLM embedding of the sim.log UVM failure message (interrupt/timeout bugs)
    if args.use_fatal_llm and all_fatal:
        import regr_fail_bucketing as rfb
        emb, _ = rfb.fetch_llm_embeddings(all_fatal, llm_args)
        fatal_mat = np.asarray(emb, dtype=np.float32)
        if fatal_reducer is None:
            fatal_reducer, fatal_reduced = plf._fit_reducer_for_matrix(
                fatal_mat, args.view_dim, args.seed + 777
            )
        else:
            fatal_reduced = plf._apply_reducer_to_matrix(fatal_mat, fatal_reducer, args.view_dim)
        fatal_reduced = np.nan_to_num(np.asarray(fatal_reduced, dtype=np.float32), nan=0.0)
    else:
        fatal_reduced = np.zeros((len(all_features), 0), dtype=np.float32)

    # fit + apply trace reducers; per-case residual = residual_struct + residual_text (96-dim)
    if args.use_trace and all_trace:
        if trace_bundle is None:
            trace_bundle, trace_matrices = ttf.fit_transform_trace_views(
                all_trace, np.arange(len(all_trace), dtype=np.int64), seed=args.seed
            )
        else:
            trace_matrices = ttf.apply_trace_reducers(trace_bundle, all_trace)
        residual = np.hstack([
            trace_matrices["residual_struct"], trace_matrices["residual_text"]
        ]).astype(np.float32)
    else:
        residual = np.zeros((len(all_features), 0), dtype=np.float32)

    # deterministic char n-gram of the sim.log UVM failure message (fatal-type discriminator)
    fatal_char_mat = _fatal_char_ngram(all_fatal_msgs)

    # divergence-window (anchor) distribution, kept as explicit columns (NOT inside the
    # SVD-reduced trace residual, which drowns it): the failure's local context is a
    # stronger, lower-noise signal than the whole-trace global sketch.
    if args.use_trace and all_trace:
        anchor_mat = np.stack([f.anchor_struct for f in all_trace]).astype(np.float32)
    else:
        anchor_mat = np.zeros((len(all_features), 0), dtype=np.float32)

    case_matrix = np.nan_to_num(np.hstack([llm_mat, sig_mat, test_mat, fatal_char_mat, anchor_mat, snippet_reduced, fatal_reduced, residual]), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ordered instruction-family sequence around the divergence (for the 1D-CNN)
    case_seq = ttf.build_trace_sequence(all_trace) if (args.use_trace and all_trace and not getattr(args, "no_seq", False)) else None

    # integer bug labels (global across datasets)
    bug_to_id: dict[str, int] = {}
    case_labels = []
    for b in all_labels:
        if b not in bug_to_id:
            bug_to_id[b] = len(bug_to_id)
        case_labels.append(bug_to_id[b])
    case_labels = np.asarray(case_labels, dtype=np.int64)
    case_domains = np.asarray(all_domains, dtype=np.float32)
    return case_matrix, case_labels, reducer, trace_bundle, snippet_reducer, test_name_vocab, fatal_reducer, case_domains, case_seq


def evaluate(args, model, datasets, reducer, trace_bundle, snippet_reducer, test_name_vocab, fatal_reducer):
    rows = []
    for ds in datasets:
        case_matrix, case_labels, _, _, _, _, _, _, case_seq = build_case_matrix(
            args, [ds], reducer=reducer, trace_bundle=trace_bundle,
            snippet_reducer=snippet_reducer, test_name_vocab=test_name_vocab, fatal_reducer=fatal_reducer,
        )
        gold = [str(b) for b in case_labels]
        k = len(set(gold))
        emb = tsm.encode_cases(model, case_matrix, case_seq=case_seq)
        labels = tsm.cluster_embeddings(emb, k, random_state=args.seed)
        pred = [f"bucket_{int(x):03d}" for x in labels]
        ba, tpr, tnr = pairwise_scores(gold, pred)
        rows.append({"dataset": ds.name, "BA": ba, "TPR": tpr, "TNR": tnr, "k": k, "cases": len(gold), "clusters": len(set(pred))})
        print(f"[eval] {ds.name} BA={ba:.4f} TPR={tpr:.4f} TNR={tnr:.4f} clusters={len(set(pred))}/{k}", flush=True)
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Siamese + prototype bucketing")
    p.add_argument("--train-datasets", nargs="+", type=Path, required=True)
    p.add_argument("--eval-datasets", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
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
    p.add_argument("--no-seq", action="store_true", default=False,
                   help="Disable the trace-sequence branch (GRU over opcode sequence); keep the trace residual features.")
    p.add_argument("--no-test-name", action="store_true", default=False,
                   help="Ablation: zero out the failing-test-name semantic category features.")
    p.add_argument("--use-mismatch-llm", action="store_true", default=False)
    p.add_argument("--use-fatal-llm", action="store_true", default=False)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--supcon-weight", type=float, default=1.0)
    p.add_argument("--prototype-weight", type=float, default=0.3)
    p.add_argument("--prototype-subcenters", type=int, default=1)
    p.add_argument("--pos-agg-weight", type=float, default=0.0)
    p.add_argument("--domain-adv-weight", type=float, default=0.0)
    p.add_argument("--supcon-temperature", type=float, default=0.1)
    p.add_argument("--early-stop-patience", type=int, default=8)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_datasets = [resolve(x) for x in args.train_datasets]
    eval_datasets = [resolve(x) for x in args.eval_datasets]
    t0 = time.perf_counter()
    args.random_state = args.seed  # train_siamese_model reads args.random_state
    case_matrix, case_labels, reducer, trace_bundle, snippet_reducer, test_name_vocab, fatal_reducer, case_domains, case_seq = build_case_matrix(args, train_datasets)
    print(f"[train] cases={case_matrix.shape[0]} feat_dim={case_matrix.shape[1]} bugs={len(set(case_labels.tolist()))} seq={None if case_seq is None else case_seq.shape}", flush=True)
    pkg = tsm.train_siamese_model(case_matrix, case_labels, args, case_domains=case_domains, case_seq=case_seq)
    # Save the encoder + reducers so seeds can be ensembled later.
    import torch
    import joblib
    torch.save({"state_dict": pkg["model"].state_dict(), "input_dim": int(case_matrix.shape[1])},
               args.output_dir / "encoder.pt")
    joblib.dump({"reducer": reducer, "trace_bundle": trace_bundle, "snippet_reducer": snippet_reducer,
                 "test_name_vocab": test_name_vocab, "fatal_reducer": fatal_reducer},
                args.output_dir / "preprocess.pkl")
    rows = evaluate(args, pkg["model"], eval_datasets, reducer, trace_bundle, snippet_reducer, test_name_vocab, fatal_reducer)
    write_csv(args.output_dir / "results.csv", rows, ["dataset", "BA", "TPR", "TNR", "k", "cases", "clusters"])
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "seed": args.seed, "supcon_weight": args.supcon_weight,
        "prototype_weight": args.prototype_weight, "feat_dim": int(case_matrix.shape[1]),
    }, indent=2) + "\n")
    print(f"[done] total={time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
