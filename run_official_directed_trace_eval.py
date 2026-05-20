#!/usr/bin/env python3
"""Direct trace-aware evaluation for official-directed fake datasets.

Experimental only. This script may read gold.csv and trace.log(.gz) for analysis,
but it does not change the official regr_fail_bucketing.py prediction path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import pairwise_llm_features as plf
import regr_fail_bucketing as rfb
import trace_anchor as ta
import trace_features as tf
import trace_policy as tpol
from run_experiments import pairwise_scores, read_gold, read_pred
from run_input_signal_experiments import ENSEMBLE_WEIGHTS, find_ensemble_models

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "fake_dataset/official_directed_stage1_valid_failure_only"


def _write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_cases(input_csv: Path) -> list[str]:
    rows, fields = rfb.read_csv_rows(input_csv)
    by_key = {"".join(ch for ch in f.lower() if ch.isalnum()): f for f in fields}
    case_col = None
    for key in ("case", "caseid", "id"):
        if key in by_key:
            case_col = by_key[key]
            break
    out = []
    for idx, row in enumerate(rows):
        value = str(row.get(case_col, "") if case_col else "").strip()
        out.append(value if value else str(idx + 1))
    return out


def _write_pred(path: Path, cases: Sequence[str], labels: Sequence[int]) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = [f"bucket_{int(x):03d}" for x in labels]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Case", "bucket"])
        for case, bucket in zip(cases, pred):
            writer.writerow([case, bucket])
    return pred


def _temperature(prob: np.ndarray, temp: float) -> np.ndarray:
    if abs(float(temp) - 1.0) < 1e-12:
        out = prob.astype(np.float32, copy=True)
        np.fill_diagonal(out, 1.0)
        return out
    clipped = np.clip(prob.astype(np.float64), 1e-5, 1.0 - 1e-5)
    logits = np.log(clipped / (1.0 - clipped)) / float(temp)
    out = 1.0 / (1.0 + np.exp(-logits))
    np.fill_diagonal(out, 1.0)
    return out.astype(np.float32)


def _logit_vec(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float32), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32)


def _all_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _labels_for_pairs(gold: Sequence[str], pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    return np.asarray([1.0 if gold[i] == gold[j] else 0.0 for i, j in pairs], dtype=np.float32)


def _prob_from_pair_scores(n: int, pairs: Sequence[tuple[int, int]], scores: np.ndarray) -> np.ndarray:
    prob = np.eye(n, dtype=np.float32)
    for (i, j), p in zip(pairs, scores):
        prob[i, j] = prob[j, i] = float(p)
    return prob


def _train_gbdt(X: np.ndarray, y: np.ndarray, seed: int) -> object:
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=220,
        max_depth=5,
        learning_rate=0.04,
        l2_regularization=0.01,
        early_stopping=False,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(X, y)
    return model


def _predict_model(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(np.float32)
    return np.clip(model.predict(X).astype(np.float32), 1e-5, 1.0 - 1e-5)


def _score_prob(
    method: str,
    prob: np.ndarray,
    input_csv: Path,
    gold_csv: Path,
    k: int,
    output_dir: Path,
    runtime: float,
    notes: str = "",
    extra: dict | None = None,
) -> dict:
    cases = _read_cases(input_csv)
    gold = read_gold(gold_csv)
    labels = plf.cluster_from_probability(prob.astype(np.float32), k)
    pred_path = output_dir / f"pred_{method}.csv"
    prob_path = output_dir / f"prob_{method}.npy"
    pred = _write_pred(pred_path, cases, labels)
    np.save(prob_path, prob.astype(np.float32))
    ba, tpr, tnr = pairwise_scores(gold, pred)
    row = {
        "method": method,
        "k": k,
        "num_cases": len(gold),
        "num_pred_clusters": len(set(pred)),
        "BA": ba,
        "TPR": tpr,
        "TNR": tnr,
        "runtime_sec": runtime,
        "pred_path": str(pred_path),
        "prob_path": str(prob_path),
        "notes": notes,
    }
    if extra:
        row.update(extra)
    return row


def run_deterministic(args: argparse.Namespace, input_csv: Path, gold_csv: Path, k: int, output_dir: Path) -> dict:
    t0 = time.perf_counter()
    pred_path = output_dir / "pred_deterministic.csv"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "regr_fail_bucketing.py"),
        "--input", str(input_csv),
        "--output", str(pred_path),
        "--k", str(k),
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True, check=False)
    runtime = time.perf_counter() - t0
    if proc.returncode != 0:
        return {
            "method": "deterministic",
            "k": k,
            "num_cases": len(read_gold(gold_csv)),
            "num_pred_clusters": "",
            "BA": 0.0,
            "TPR": 0.0,
            "TNR": 0.0,
            "runtime_sec": runtime,
            "pred_path": str(pred_path),
            "prob_path": "",
            "notes": "FAILED: " + (proc.stderr.strip() or proc.stdout.strip())[-600:],
        }
    gold = read_gold(gold_csv)
    pred = read_pred(pred_path)
    ba, tpr, tnr = pairwise_scores(gold, pred)
    return {
        "method": "deterministic",
        "k": k,
        "num_cases": len(gold),
        "num_pred_clusters": len(set(pred)),
        "BA": ba,
        "TPR": tpr,
        "TNR": tnr,
        "runtime_sec": runtime,
        "pred_path": str(pred_path),
        "prob_path": "",
        "notes": "default no-trace pipeline",
    }


def _model_path(root: Path, seed: int, tag: str) -> Path:
    return root / f"model_seed{seed}_combo000_{tag}.pt"


def build_current_best_probability(args: argparse.Namespace, input_csv: Path) -> tuple[np.ndarray | None, list[plf.LLMCaseFeature] | None, str, float]:
    t0 = time.perf_counter()
    if not args.rich_model_root.exists() or not args.ensemble_model_dir.exists():
        return None, None, "missing current-best artifact directory", time.perf_counter() - t0
    rich_args = plf._make_llm_args(
        llm_mode="embedding",
        llm_doc_style="features",
        llm_cache_dir=args.llm_cache_dir,
        svd_dim=args.svd_dim,
        llm_dual=True,
    )
    ens_args = plf._make_llm_args(
        llm_mode="embedding",
        llm_doc_style="features",
        llm_cache_dir=args.llm_cache_dir,
        svd_dim=args.svd_dim,
        llm_dual=False,
    )
    rich_features, _ = plf.build_llm_case_features(input_csv, svd_dim=args.svd_dim, llm_args=rich_args)
    ens_features, _ = plf.build_llm_case_features(input_csv, svd_dim=args.svd_dim, llm_args=ens_args)
    prob_sum = np.zeros((len(rich_features), len(rich_features)), dtype=np.float64)
    used = 0
    missing: list[str] = []
    for seed in args.seeds:
        rich_path = _model_path(args.rich_model_root, seed, args.model_tag)
        if not rich_path.exists():
            missing.append(str(rich_path))
            continue
        try:
            rich_pkg = plf.load_model_pkg(rich_path)
            ensemble_pkgs = [plf.load_model_pkg(p) for p in find_ensemble_models(args.ensemble_model_dir, seed, 0)]
        except Exception as exc:
            missing.append(f"seed {seed}: {exc}")
            continue
        p_rich = plf.predict_probability_matrix_sklearn(rich_pkg, rich_features, batch_size=args.predict_batch_size)
        p_ens = plf.predict_probability_matrix_ensemble(
            ensemble_pkgs,
            list(ENSEMBLE_WEIGHTS),
            ens_features,
            ensemble_mode="prob_average",
            batch_size=args.predict_batch_size,
        )
        p = args.alpha * _temperature(p_rich, args.rich_temp) + (1.0 - args.alpha) * _temperature(p_ens, args.ensemble_temp)
        prob_sum += p.astype(np.float64)
        used += 1
        print(f"[current_best] seed={seed} ok", file=sys.stderr)
    if used == 0:
        return None, rich_features, "no usable current-best seed; " + "; ".join(missing[:3]), time.perf_counter() - t0
    prob = (prob_sum / float(used)).astype(np.float32)
    np.fill_diagonal(prob, 1.0)
    note = f"seed_average={used}; alpha={args.alpha}; rich_temp={args.rich_temp}; ensemble_temp={args.ensemble_temp}"
    if missing:
        note += f"; missing_or_failed={len(missing)}"
    return prob, rich_features, note, time.perf_counter() - t0


def _trace_feature_matrix(
    mode: str,
    input_csv: Path,
    pairs: Sequence[tuple[int, int]],
    window_size: int,
    tail_lines: int,
) -> tuple[np.ndarray, Sequence[tf.TraceCaseFeature], list[dict]]:
    if mode == "tail":
        feats = tf.build_trace_case_features(input_csv, tail_lines=tail_lines)
        return tf.build_trace_pair_feature_matrix(feats, pairs), feats, []
    if mode == "anchor":
        feats, debug = ta.build_anchor_trace_case_features([input_csv], window_size=window_size)
        return ta.build_anchor_trace_pair_feature_matrix(feats, pairs), feats, debug
    raise ValueError(mode)


def _build_supervised_trace_X(
    trace_X: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    base_prob: np.ndarray | None,
    rich_features: list[plf.LLMCaseFeature] | None,
    context: str,
) -> np.ndarray:
    blocks = [trace_X.astype(np.float32, copy=False)]
    if base_prob is not None and context in {"base", "summary", "rich"}:
        p = np.asarray([base_prob[i, j] for i, j in pairs], dtype=np.float32)
        blocks.append(np.vstack([p, _logit_vec(p), np.abs(p - 0.5)]).T.astype(np.float32))
    if rich_features is not None and context == "summary":
        blocks.append(plf.build_rich_pair_feature_matrix(list(rich_features), list(pairs), feature_mode="summary21"))
    if rich_features is not None and context == "rich":
        blocks.append(plf.build_rich_pair_feature_matrix(list(rich_features), list(pairs), feature_mode="llm_dual_struct_det_summary"))
    return np.hstack(blocks).astype(np.float32, copy=False)


def run_trace_supervised(
    args: argparse.Namespace,
    input_csv: Path,
    gold_csv: Path,
    k: int,
    output_dir: Path,
    mode: str,
    window_size: int,
    base_prob: np.ndarray | None,
    rich_features: list[plf.LLMCaseFeature] | None,
) -> tuple[dict, list[dict]]:
    t0 = time.perf_counter()
    gold = read_gold(gold_csv)
    pairs = _all_pairs(len(gold))
    y = _labels_for_pairs(gold, pairs)
    trace_X, trace_feats, debug_rows = _trace_feature_matrix(mode, input_csv, pairs, window_size, args.tail_lines)
    context = args.trace_context
    if rich_features is None and context in {"summary", "rich"}:
        context = "base" if base_prob is not None else "trace_only"
    X = _build_supervised_trace_X(trace_X, pairs, base_prob, rich_features, context)
    model = _train_gbdt(X, y, args.random_state)
    prob = _prob_from_pair_scores(len(gold), pairs, _predict_model(model, X))
    method = f"trace_tail_{context}" if mode == "tail" else f"anchor_trace_w{window_size}_{context}"
    located_counts = Counter(row.get("located_by", "") for row in debug_rows)
    fallback_rate = float(located_counts.get("tail", 0)) / max(1, len(debug_rows)) if debug_rows else 0.0
    missing_pairs = sum(1 for i, j in pairs if trace_feats[i].missing or trace_feats[j].missing)
    row = _score_prob(
        method,
        prob,
        input_csv,
        gold_csv,
        k,
        output_dir,
        time.perf_counter() - t0,
        notes="direct supervised trace GBDT; train=evaluate same gold dataset",
        extra={
            "refined_pairs": "",
            "trace_missing_pairs": missing_pairs,
            "located_by_counts": json.dumps(dict(located_counts), sort_keys=True),
            "fallback_rate": fallback_rate,
            "feature_dim": X.shape[1],
        },
    )
    return row, debug_rows


def run_trace_policy(
    args: argparse.Namespace,
    input_csv: Path,
    gold_csv: Path,
    k: int,
    output_dir: Path,
    base_prob: np.ndarray | None,
    rich_features: list[plf.LLMCaseFeature] | None,
) -> dict:
    t0 = time.perf_counter()
    if base_prob is None or rich_features is None:
        return {
            "method": "trace_policy",
            "k": k,
            "num_cases": len(read_gold(gold_csv)),
            "num_pred_clusters": "",
            "BA": 0.0,
            "TPR": 0.0,
            "TNR": 0.0,
            "runtime_sec": time.perf_counter() - t0,
            "pred_path": "",
            "prob_path": "",
            "notes": "skipped: no base probability/rich features",
        }
    trace_feats = tf.build_trace_case_features(input_csv, tail_lines=args.tail_lines)
    params = tpol.TracePolicyParams(trace_policy=args.trace_policy)
    prob, stats = tpol.apply_trace_policy(base_prob, trace_feats, rich_features, params)
    row = _score_prob(
        f"trace_policy_{args.trace_policy}",
        prob,
        input_csv,
        gold_csv,
        k,
        output_dir,
        time.perf_counter() - t0,
        notes=f"policy={args.trace_policy}; default conservative params",
        extra={
            "refined_pairs": stats.pairs_vetoed + stats.pairs_boosted,
            "trace_missing_pairs": stats.trace_missing_pairs,
            "pairs_vetoed": stats.pairs_vetoed,
            "pairs_boosted": stats.pairs_boosted,
        },
    )
    return row


def _distribution(prob: np.ndarray, gold: Sequence[str], pred: Sequence[str]) -> dict[str, list[float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for i in range(len(gold)):
        for j in range(i + 1, len(gold)):
            same_gold = gold[i] == gold[j]
            same_pred = pred[i] == pred[j]
            p = float(prob[i, j])
            if same_gold:
                buckets["positive"].append(p)
                if not same_pred:
                    buckets["FN"].append(p)
            else:
                buckets["negative"].append(p)
                if same_pred:
                    buckets["FP"].append(p)
    return buckets


def _quantiles(values: Sequence[float]) -> str:
    if not values:
        return "n=0"
    arr = np.asarray(values, dtype=np.float32)
    qs = np.percentile(arr, [0, 10, 25, 50, 75, 90, 100])
    return "n={} min={:.3f} p10={:.3f} p25={:.3f} med={:.3f} p75={:.3f} p90={:.3f} max={:.3f}".format(len(arr), *qs)


def build_error_report(dataset: Path, rows: Sequence[dict], output_dir: Path) -> str:
    gold = read_gold(dataset / "gold.csv")
    cases = _read_cases(dataset / "input.csv")
    lines: list[str] = []
    lines.append("# Official-directed Trace Evaluation Error Analysis\n")
    lines.append(f"Dataset: `{dataset}`  ")
    lines.append(f"Cases: {len(gold)}  Gold bugs: {dict(Counter(gold))}\n")
    lines.append("## Method Scores\n")
    lines.append("| method | BA | TPR | TNR | clusters | notes |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(f"| {row['method']} | {float(row['BA']):.6f} | {float(row['TPR']):.6f} | {float(row['TNR']):.6f} | {row.get('num_pred_clusters','')} | {str(row.get('notes','')).replace('|','/')} |")
    for row in rows:
        pred_path = Path(str(row.get("pred_path", "")))
        if not pred_path.exists():
            continue
        pred = read_pred(pred_path)
        lines.append(f"\n## {row['method']}\n")
        gold_bucket_counts: dict[str, Counter] = defaultdict(Counter)
        bucket_gold_counts: dict[str, Counter] = defaultdict(Counter)
        for g, p in zip(gold, pred):
            gold_bucket_counts[g][p] += 1
            bucket_gold_counts[p][g] += 1
        lines.append("Top fragmented gold bugs:")
        frag = sorted(gold_bucket_counts.items(), key=lambda kv: (len(kv[1]), sum(kv[1].values())), reverse=True)
        for bug, counts in frag[:8]:
            lines.append(f"- {bug}: {len(counts)} buckets, {dict(counts)}")
        lines.append("\nTop mixed predicted buckets:")
        mixed = []
        for bucket, counts in bucket_gold_counts.items():
            total = sum(counts.values())
            purity = max(counts.values()) / max(1, total)
            mixed.append((purity, total, bucket, counts))
        for purity, total, bucket, counts in sorted(mixed, key=lambda x: (x[0], -x[1]))[:8]:
            lines.append(f"- {bucket}: purity={purity:.3f}, n={total}, {dict(counts)}")
        fp_pairs = Counter()
        fn_bugs = Counter()
        for i in range(len(gold)):
            for j in range(i + 1, len(gold)):
                if gold[i] == gold[j] and pred[i] != pred[j]:
                    fn_bugs[gold[i]] += 1
                if gold[i] != gold[j] and pred[i] == pred[j]:
                    fp_pairs[tuple(sorted((gold[i], gold[j])))] += 1
        lines.append("\nTop FN bugs:")
        for bug, cnt in fn_bugs.most_common(8):
            lines.append(f"- {bug}: {cnt}")
        lines.append("\nTop FP bug pairs:")
        for pair, cnt in fp_pairs.most_common(12):
            marker = "  <= target" if set(pair) <= {"bug_036", "bug_037", "bug_038", "bug_041"} else ""
            lines.append(f"- {pair[0]} / {pair[1]}: {cnt}{marker}")
        prob_raw = str(row.get("prob_path", "") or "")
        prob_path = Path(prob_raw) if prob_raw else None
        if prob_path is not None and prob_path.is_file():
            prob = np.load(prob_path)
            dist = _distribution(prob, gold, pred)
            lines.append("\nProbability distributions:")
            for key in ["positive", "negative", "FN", "FP"]:
                lines.append(f"- {key}: {_quantiles(dist.get(key, []))}")
        lines.append("\nBug_036/037/038/041 confusion counts:")
        target = {"bug_036", "bug_037", "bug_038", "bug_041"}
        target_counts = Counter()
        for i in range(len(gold)):
            for j in range(i + 1, len(gold)):
                if gold[i] in target and gold[j] in target and gold[i] != gold[j] and pred[i] == pred[j]:
                    target_counts[tuple(sorted((gold[i], gold[j])))] += 1
        if target_counts:
            for pair, cnt in target_counts.most_common():
                lines.append(f"- {pair[0]} / {pair[1]}: {cnt}")
        else:
            lines.append("- none")
    report = "\n".join(lines) + "\n"
    (output_dir / "error_analysis.md").write_text(report, encoding="utf-8")
    return report


def run(args: argparse.Namespace) -> list[dict]:
    dataset = (PROJECT_ROOT / args.dataset).resolve() if not args.dataset.is_absolute() else args.dataset.resolve()
    input_csv = dataset / "input.csv"
    gold_csv = dataset / "gold.csv"
    if not input_csv.exists() or not gold_csv.exists():
        raise FileNotFoundError(f"missing input/gold under {dataset}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gold = read_gold(gold_csv)
    k = len(set(gold)) if args.k_from_gold else int(args.k)
    rows: list[dict] = []
    base_prob: np.ndarray | None = None
    rich_features: list[plf.LLMCaseFeature] | None = None
    debug_rows: list[dict] = []

    methods = set(args.methods)
    if "deterministic" in methods:
        rows.append(run_deterministic(args, input_csv, gold_csv, k, args.output_dir))
        print(f"[result] deterministic BA={float(rows[-1]['BA']):.6f}", file=sys.stderr)
    if "no_trace_best" in methods or "trace_policy" in methods or args.trace_context in {"base", "summary", "rich"}:
        prob, feats, note, runtime = build_current_best_probability(args, input_csv)
        base_prob = prob
        rich_features = feats
        if prob is not None and "no_trace_best" in methods:
            row = _score_prob("no_trace_best", prob, input_csv, gold_csv, k, args.output_dir, runtime, notes=note)
            rows.append(row)
            print(f"[result] no_trace_best BA={float(row['BA']):.6f}", file=sys.stderr)
        elif prob is None:
            rows.append({
                "method": "no_trace_best",
                "k": k,
                "num_cases": len(gold),
                "num_pred_clusters": "",
                "BA": 0.0,
                "TPR": 0.0,
                "TNR": 0.0,
                "runtime_sec": runtime,
                "pred_path": "",
                "prob_path": "",
                "notes": "skipped: " + note,
            })
    if "trace_policy" in methods:
        row = run_trace_policy(args, input_csv, gold_csv, k, args.output_dir, base_prob, rich_features)
        rows.append(row)
        print(f"[result] {row['method']} BA={float(row['BA']):.6f}", file=sys.stderr)
    if "trace_tail" in methods:
        row, dbg = run_trace_supervised(args, input_csv, gold_csv, k, args.output_dir, "tail", args.tail_lines, base_prob, rich_features)
        rows.append(row)
        debug_rows.extend(dbg)
        print(f"[result] {row['method']} BA={float(row['BA']):.6f}", file=sys.stderr)
    if "anchor_trace" in methods:
        for window in args.window_sizes:
            row, dbg = run_trace_supervised(args, input_csv, gold_csv, k, args.output_dir, "anchor", int(window), base_prob, rich_features)
            rows.append(row)
            debug_rows.extend(dict(x, window_size=window) for x in dbg)
            print(f"[result] {row['method']} BA={float(row['BA']):.6f}", file=sys.stderr)
    if "trace_transformer" in methods:
        rows.append({
            "method": "trace_transformer_embedding",
            "k": k,
            "num_cases": len(gold),
            "num_pred_clusters": "",
            "BA": 0.0,
            "TPR": 0.0,
            "TNR": 0.0,
            "runtime_sec": 0.0,
            "pred_path": "",
            "prob_path": "",
            "notes": "skipped: direct runner supports the method name, but no cached trace encoder was supplied",
        })
    fields = [
        "method", "k", "num_cases", "num_pred_clusters", "BA", "TPR", "TNR",
        "runtime_sec", "pred_path", "prob_path", "refined_pairs", "trace_missing_pairs",
        "pairs_vetoed", "pairs_boosted", "located_by_counts", "fallback_rate", "feature_dim", "notes",
    ]
    _write_csv(args.output_dir / "results.csv", rows, fields)
    _write_csv(args.output_dir / "summary.csv", rows, fields)
    if debug_rows:
        debug_fields = sorted({key for row in debug_rows for key in row})
        _write_csv(args.output_dir / "anchor_debug.csv", debug_rows, debug_fields)
    build_error_report(dataset, rows, args.output_dir)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct official-directed trace-aware evaluation (experimental).")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/official_directed_trace_eval"))
    p.add_argument("--methods", nargs="+", default=["deterministic", "no_trace_best", "trace_tail", "trace_policy", "anchor_trace"])
    p.add_argument("--window-sizes", nargs="+", type=int, default=[32, 64, 128])
    p.add_argument("--tail-lines", type=int, default=500)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--k-from-gold", action="store_true")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--rich-model-root", type=Path, default=Path("/tmp/input_signal_5seed_top/models/llm_dual_struct_det_summary_dim64"))
    p.add_argument("--model-tag", default="llm_dual_struct_det_summary_dim64")
    p.add_argument("--ensemble-model-dir", type=Path, default=Path("/tmp/pairwise_llm_exp_full/models"))
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--predict-batch-size", type=int, default=100000)
    p.add_argument("--alpha", type=float, default=0.88)
    p.add_argument("--rich-temp", type=float, default=1.15)
    p.add_argument("--ensemble-temp", type=float, default=1.00)
    p.add_argument("--trace-context", choices=("trace_only", "base", "summary", "rich"), default="rich")
    p.add_argument("--trace-policy", choices=("none", "veto", "boost", "veto_boost"), default="veto_boost")
    p.add_argument("--use-llm", action="store_true", help="Compatibility flag; current-best already uses configured embedding if available.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = run(args)
    print("\n| method | BA | TPR | TNR | clusters | runtime_sec | notes |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            f"| {row['method']} | {float(row['BA']):.6f} | {float(row['TPR']):.6f} | "
            f"{float(row['TNR']):.6f} | {row.get('num_pred_clusters','')} | "
            f"{float(row.get('runtime_sec') or 0.0):.2f} | {str(row.get('notes','')).replace('|','/')} |"
        )
    print(f"\nResults: {args.output_dir / 'results.csv'}")
    print(f"Summary: {args.output_dir / 'summary.csv'}")
    print(f"Error analysis: {args.output_dir / 'error_analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
