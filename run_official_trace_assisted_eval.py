#!/usr/bin/env python3
"""Official-gold trace-assisted validation.

Experimental only. This runner may read official ``golden.csv`` files and
``trace.log(.gz)`` for evaluation/postprocess research, but it does not change
the default ``regr_fail_bucketing.py`` prediction path.
"""

from __future__ import annotations

import argparse
import csv
import math
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
from run_experiments import pairwise_scores, read_gold
from run_official_directed_trace_eval import (
    _read_cases,
    _score_prob,
    build_current_best_probability,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def _write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _gold_path(dataset: Path) -> Path:
    for name in ("gold.csv", "golden.csv", "answer.csv", "answers.csv", "labels.csv"):
        path = dataset / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no gold/golden csv found under {dataset}")


def _all_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    return float(sum(values) / len(values)) if values else float(default)


def _same_bucket(labels: Sequence[int]) -> np.ndarray:
    n = len(labels)
    out = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = float(labels[i] == labels[j])
    return out


def _diff_to_similarity(value: float, scale: float = 1.0) -> float:
    return 1.0 / (1.0 + max(0.0, float(value)) / max(float(scale), 1e-6))


def _trace_similarity_from_vector(vec: np.ndarray) -> float:
    """Convert structural trace pair features to a probability-like score.

    The underlying trace features are not learned. Similarity dimensions are
    averaged with conservative weights; difference/missing dimensions apply
    only mild penalties so zero-shot trace cannot dominate unless the trace
    windows are strongly coherent.
    """
    v = np.asarray(vec, dtype=np.float32)
    parts: list[tuple[float, float]] = []
    # Base trace features from trace_features.build_trace_pair_feature_vector.
    sim_weights = {
        0: 0.18,   # opcode Jaccard
        1: 0.16,   # opcode count cosine
        2: 0.07,   # register Jaccard
        3: 0.05,   # register count cosine
        4: 0.14,   # PC region Jaccard
        5: 0.05,   # PC prefix Jaccard
        6: 0.12,   # tail/window LCS
        7: 0.10,   # opcode 2-gram Jaccard
        8: 0.06,   # opcode 3-gram Jaccard
        9: 0.03,   # last opcode same
        10: 0.03,  # last-5 opcode overlap
        11: 0.01,  # exception marker same
    }
    for idx, weight in sim_weights.items():
        if idx < len(v):
            parts.append((float(np.clip(v[idx], 0.0, 1.0)), weight))
    # Ratio/length differences.
    for idx, weight, scale in (
        (12, 0.025, 0.20),  # branch ratio diff
        (13, 0.025, 0.20),  # load ratio diff
        (14, 0.025, 0.20),  # store ratio diff
        (15, 0.025, 0.20),  # csr ratio diff
        (16, 0.025, 2.00),  # log length diff
    ):
        if idx < len(v):
            parts.append((_diff_to_similarity(float(v[idx]), scale), weight))
    # Extra anchor features, when present.
    if len(v) > 19:
        anchor_weights = {
            19: 0.08,  # center opcode same
            20: 0.08,  # center PC region same
            21: 0.06,  # pre opcode overlap
            22: 0.06,  # post opcode overlap
            23: 0.10,  # full window LCS
            24: 0.03,  # anchor source same
            25: 0.03,  # located_by same
            28: 0.04,  # target register same
            29: 0.06,  # anchor tag Jaccard
        }
        for idx, weight in anchor_weights.items():
            if idx < len(v):
                parts.append((float(np.clip(v[idx], 0.0, 1.0)), weight))
        if len(v) > 26:
            parts.append((_diff_to_similarity(float(v[26]), 4.0), 0.02))
    score = sum(value * weight for value, weight in parts) / max(sum(weight for _, weight in parts), 1e-12)
    missing_penalty = 0.0
    if len(v) > 17:
        missing_penalty += 0.20 * float(v[17])
    if len(v) > 18:
        missing_penalty += 0.30 * float(v[18])
    return float(np.clip(score - missing_penalty, 0.0, 1.0))


def _prob_from_trace_pairs(n: int, pairs: Sequence[tuple[int, int]], pair_matrix: np.ndarray) -> np.ndarray:
    prob = np.eye(n, dtype=np.float32)
    for row, (i, j) in enumerate(pairs):
        prob[i, j] = prob[j, i] = _trace_similarity_from_vector(pair_matrix[row])
    return prob


def _score_labels(
    dataset_name: str,
    method: str,
    labels: Sequence[int],
    input_csv: Path,
    gold_csv: Path,
    k: int,
    output_dir: Path,
    runtime: float,
    notes: str = "",
    prob: np.ndarray | None = None,
    extra: dict | None = None,
) -> dict:
    cases = _read_cases(input_csv)
    pred = [f"bucket_{int(label):03d}" for label in labels]
    pred_path = output_dir / "preds" / f"{dataset_name}_{method}.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Case", "bucket"])
        writer.writerows(zip(cases, pred))
    prob_path = ""
    if prob is not None:
        prob_file = output_dir / "probs" / f"{dataset_name}_{method}.npy"
        prob_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(prob_file, prob.astype(np.float32))
        prob_path = str(prob_file)
    ba, tpr, tnr = pairwise_scores(read_gold(gold_csv), pred)
    row = {
        "dataset": dataset_name,
        "method": method,
        "k": k,
        "cases": len(pred),
        "num_pred_clusters": len(set(pred)),
        "BA": ba,
        "TPR": tpr,
        "TNR": tnr,
        "runtime_sec": runtime,
        "pred_path": str(pred_path),
        "prob_path": prob_path,
        "notes": notes,
    }
    if extra:
        row.update(extra)
    return row


def _score_probability(
    dataset_name: str,
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
    row = _score_prob(method, prob, input_csv, gold_csv, k, output_dir / dataset_name, runtime, notes=notes, extra=extra)
    pred_path = Path(str(row["pred_path"]))
    prob_path = Path(str(row["prob_path"]))
    new_pred = output_dir / "preds" / f"{dataset_name}_{method}.csv"
    new_prob = output_dir / "probs" / f"{dataset_name}_{method}.npy"
    new_pred.parent.mkdir(parents=True, exist_ok=True)
    new_prob.parent.mkdir(parents=True, exist_ok=True)
    new_pred.write_text(pred_path.read_text(encoding="utf-8"), encoding="utf-8")
    new_prob.write_bytes(prob_path.read_bytes())
    row.update({
        "dataset": dataset_name,
        "cases": row.pop("num_cases"),
        "pred_path": str(new_pred),
        "prob_path": str(new_prob),
    })
    return row


def _tail_debug_rows(dataset_name: str, method: str, feats: Sequence[tf.TraceCaseFeature]) -> list[dict]:
    rows = []
    for feat in feats:
        rows.append({
            "dataset": dataset_name,
            "case_id": feat.case_id,
            "method": method,
            "anchor_located_by": "",
            "anchor_type": "",
            "target_time": "",
            "target_pc": "",
            "reason_tags": "",
            "window_size": "",
            "trace_status": feat.file_status,
            "fallback_used": "",
            "top_opcodes": ";".join(f"{op}:{cnt}" for op, cnt in feat.opcode_counts.most_common(8)),
            "pc_regions": ";".join(sorted(feat.pc_regions)[:8]),
        })
    return rows


def _anchor_debug_rows(dataset_name: str, method: str, window_size: int, feats: Sequence[ta.AnchorTraceFeature], debug: Sequence[dict]) -> list[dict]:
    rows = []
    by_case = {str(row.get("case_id", "")): row for row in debug}
    for feat in feats:
        dbg = by_case.get(feat.case_id, {})
        anchor = feat.anchor
        rows.append({
            "dataset": dataset_name,
            "case_id": feat.case_id,
            "method": method,
            "anchor_located_by": feat.located_by,
            "anchor_type": dbg.get("anchor_source", ""),
            "target_time": dbg.get("sim_time", ""),
            "target_pc": dbg.get("dut_pc", ""),
            "reason_tags": ";".join(anchor.anchor_tags) if anchor else "",
            "window_size": window_size,
            "trace_status": feat.file_status,
            "fallback_used": str(feat.located_by == "tail"),
            "top_opcodes": ";".join(f"{op}:{cnt}" for op, cnt in feat.opcode_counts.most_common(8)),
            "pc_regions": ";".join(sorted(feat.pc_regions)[:8]),
        })
    return rows


def run_trace_unsupervised(
    dataset_name: str,
    input_csv: Path,
    gold_csv: Path,
    k: int,
    output_dir: Path,
    mode: str,
    window_size: int = 64,
    tail_lines: int = 500,
) -> tuple[dict, list[dict], np.ndarray]:
    t0 = time.perf_counter()
    n = len(_read_cases(input_csv))
    pairs = _all_pairs(n)
    if mode == "tail":
        feats = tf.build_trace_case_features(input_csv, tail_lines=tail_lines)
        mat = tf.build_trace_pair_feature_matrix(feats, pairs)
        prob = _prob_from_trace_pairs(n, pairs, mat)
        labels = plf.cluster_from_probability(prob, k)
        row = _score_labels(dataset_name, "trace_tail_unsupervised", labels, input_csv, gold_csv, k, output_dir, time.perf_counter() - t0, notes=f"tail_lines={tail_lines}", prob=prob)
        return row, _tail_debug_rows(dataset_name, "trace_tail_unsupervised", feats), prob
    feats, debug = ta.build_anchor_trace_case_features([input_csv], window_size=window_size)
    mat = ta.build_anchor_trace_pair_feature_matrix(feats, pairs)
    prob = _prob_from_trace_pairs(n, pairs, mat)
    labels = plf.cluster_from_probability(prob, k)
    method = f"anchor_trace_unsupervised_w{window_size}"
    row = _score_labels(dataset_name, method, labels, input_csv, gold_csv, k, output_dir, time.perf_counter() - t0, notes=f"window_size={window_size}", prob=prob)
    return row, _anchor_debug_rows(dataset_name, method, window_size, feats, debug), prob


def run_trace_guided_split(
    dataset_name: str,
    input_csv: Path,
    gold_csv: Path,
    k: int,
    output_dir: Path,
    base_prob: np.ndarray | None,
    anchor_prob: np.ndarray,
    window_size: int,
) -> dict:
    t0 = time.perf_counter()
    method = f"trace_guided_split_w{window_size}"
    if base_prob is None:
        labels = plf.cluster_from_probability(anchor_prob, k)
        return _score_labels(dataset_name, method, labels, input_csv, gold_csv, k, output_dir, time.perf_counter() - t0, notes="no base prob; used anchor trace clustering", prob=anchor_prob)
    base_labels = list(map(int, plf.cluster_from_probability(base_prob, k)))
    n = len(base_labels)
    counts = Counter(base_labels)
    largest_label, largest_size = counts.most_common(1)[0]
    if not (n <= 16 and k <= 3 and largest_size >= max(4, math.ceil(0.65 * n)) and k == 2):
        return _score_labels(dataset_name, method, base_labels, input_csv, gold_csv, k, output_dir, time.perf_counter() - t0, notes="gated_noop: not small-k dominant-bucket case", prob=base_prob)
    major = [idx for idx, label in enumerate(base_labels) if label == largest_label]
    minor = [idx for idx, label in enumerate(base_labels) if label != largest_label]
    if len(major) < 3:
        return _score_labels(dataset_name, method, base_labels, input_csv, gold_csv, k, output_dir, time.perf_counter() - t0, notes="gated_noop: dominant bucket too small", prob=base_prob)
    sub_prob = anchor_prob[np.ix_(major, major)]
    sub_labels = list(map(int, plf.cluster_from_probability(sub_prob, 2)))
    final = [-1] * n
    for local_idx, global_idx in enumerate(major):
        final[global_idx] = int(sub_labels[local_idx])
    # Keep the total cluster count at k by attaching previous singleton/small
    # clusters to the nearest trace-derived side, rather than creating a third
    # cluster. This is still zero-shot: no gold labels are inspected.
    for idx in minor:
        scores = []
        for label in (0, 1):
            members = [m for m in major if final[m] == label]
            scores.append(_safe_mean([float(anchor_prob[idx, m]) for m in members], default=0.0))
        final[idx] = int(np.argmax(scores))
    out_prob = _same_bucket(final)
    return _score_labels(
        dataset_name,
        method,
        final,
        input_csv,
        gold_csv,
        k,
        output_dir,
        time.perf_counter() - t0,
        notes=f"split_largest={largest_size}/{n}; window_size={window_size}",
        prob=out_prob,
        extra={"split_largest_bucket_size": largest_size},
    )


def run_trace_policy_zero_shot(
    dataset_name: str,
    input_csv: Path,
    gold_csv: Path,
    k: int,
    output_dir: Path,
    base_prob: np.ndarray | None,
    case_features: list[plf.LLMCaseFeature] | None,
) -> tuple[dict, list[dict]]:
    t0 = time.perf_counter()
    if base_prob is None or case_features is None:
        row = {
            "dataset": dataset_name,
            "method": "trace_policy_zero_shot",
            "k": k,
            "cases": len(read_gold(gold_csv)),
            "num_pred_clusters": "",
            "BA": 0.0,
            "TPR": 0.0,
            "TNR": 0.0,
            "runtime_sec": time.perf_counter() - t0,
            "pred_path": "",
            "prob_path": "",
            "notes": "skipped: missing no-trace base probability",
        }
        return row, []
    feats = tf.build_trace_case_features(input_csv, tail_lines=500)
    params = tpol.TracePolicyParams(
        trace_policy="veto_boost",
        veto_base_min=0.55,
        veto_trace_max=0.20,
        veto_cap=0.35,
        boost_base_low=0.25,
        boost_base_high=0.60,
        boost_trace_min=0.55,
        boost_floor=0.65,
    )
    prob, stats = tpol.apply_trace_policy(base_prob, feats, case_features, params)
    row = _score_probability(
        dataset_name,
        "trace_policy_zero_shot",
        prob,
        input_csv,
        gold_csv,
        k,
        output_dir,
        time.perf_counter() - t0,
        notes="veto base>=0.55/agreement<=0.20 cap=0.35; boost base=[0.25,0.60]/agreement>=0.55 floor=0.65",
        extra={
            "pairs_vetoed": stats.pairs_vetoed,
            "pairs_boosted": stats.pairs_boosted,
            "trace_missing_pairs": stats.trace_missing_pairs,
        },
    )
    return row, _tail_debug_rows(dataset_name, "trace_policy_zero_shot", feats)


def _collect_trace_paths(input_csv: Path) -> list[tuple[str, Path | None]]:
    rows, fields = rfb.read_csv_rows(input_csv)
    trace_col = tf.pick_trace_column(fields)
    cases = _read_cases(input_csv)
    out = []
    for idx, row in enumerate(rows):
        path = rfb.resolve_log_path(input_csv, row.get(trace_col) if trace_col else None)
        out.append((cases[idx], path if path and path.exists() else None))
    return out


def run_existing_trace_embedding(
    dataset_name: str,
    input_csv: Path,
    gold_csv: Path,
    k: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    t0 = time.perf_counter()
    encoder_dir = args.trace_encoder_dir
    model_root = args.trace_embedding_model_root
    if not encoder_dir.exists() or not model_root.exists():
        return {
            "dataset": dataset_name,
            "method": "existing_trace_embedding",
            "k": k,
            "cases": len(read_gold(gold_csv)),
            "num_pred_clusters": "",
            "BA": 0.0,
            "TPR": 0.0,
            "TNR": 0.0,
            "runtime_sec": time.perf_counter() - t0,
            "pred_path": "",
            "prob_path": "",
            "notes": "skipped: missing trace encoder/model artifact",
        }
    try:
        from trace_transformer_pretrain import load_pretrained

        encoder = load_pretrained(encoder_dir, device=args.device)
        llm_args = plf._make_llm_args(
            llm_mode="embedding",
            llm_doc_style="features",
            llm_cache_dir=args.llm_cache_dir,
            svd_dim=args.svd_dim,
            llm_dual=True,
        )
        features, _ = plf.build_llm_case_features(input_csv, svd_dim=args.svd_dim, llm_args=llm_args)
        collected_paths = _collect_trace_paths(input_csv)
        trace_by_case = dict(collected_paths)
        encoded = 0
        for idx, feat in enumerate(features):
            path = trace_by_case.get(feat.case_id)
            if path is None and feat.case_id.startswith("case_"):
                path = trace_by_case.get(feat.case_id.removeprefix("case_"))
            if path is None and idx < len(collected_paths):
                path = collected_paths[idx][1]
            if path is not None:
                feat.trace_vec = encoder.encode_trace_tail(str(path), tail_lines=args.tail_lines)
                encoded += int(feat.trace_vec.size > 0)
        plf.normalize_trace_vectors(features)
        prob_sum = np.zeros((len(features), len(features)), dtype=np.float64)
        used = 0
        for seed in args.trace_embedding_seeds:
            model_path = model_root / f"model_seed{seed}_combo000_trace_embedding.pt"
            if not model_path.exists():
                continue
            pkg = plf.load_model_pkg(model_path)
            prob_sum += plf.predict_probability_matrix_sklearn(pkg, features, batch_size=args.predict_batch_size)
            used += 1
        if used == 0:
            raise FileNotFoundError("no trace embedding model_seed*_combo000_trace_embedding.pt artifacts found")
        prob = (prob_sum / float(used)).astype(np.float32)
        np.fill_diagonal(prob, 1.0)
        return _score_probability(
            dataset_name,
            "existing_trace_embedding",
            prob,
            input_csv,
            gold_csv,
            k,
            output_dir,
            time.perf_counter() - t0,
            notes=f"seed_average={used}; encoder={encoder_dir}",
        )
    except Exception as exc:
        return {
            "dataset": dataset_name,
            "method": "existing_trace_embedding",
            "k": k,
            "cases": len(read_gold(gold_csv)),
            "num_pred_clusters": "",
            "BA": 0.0,
            "TPR": 0.0,
            "TNR": 0.0,
            "runtime_sec": time.perf_counter() - t0,
            "pred_path": "",
            "prob_path": "",
            "notes": f"skipped/failed: {type(exc).__name__}: {exc}",
        }


def _cluster_composition(gold: Sequence[str], pred: Sequence[str], cases: Sequence[str]) -> list[str]:
    by_bucket: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for case, bug, bucket in zip(cases, gold, pred):
        by_bucket[bucket].append((case, bug))
    lines = []
    for bucket, values in sorted(by_bucket.items()):
        counts = Counter(bug for _, bug in values)
        case_list = ",".join(case for case, _ in values)
        lines.append(f"- {bucket}: n={len(values)} bugs={dict(counts)} cases={case_list}")
    return lines


def _pair_errors(gold: Sequence[str], pred: Sequence[str], cases: Sequence[str]) -> tuple[Counter[str], Counter[tuple[str, str]], list[str], list[str]]:
    fn_bugs: Counter[str] = Counter()
    fp_pairs: Counter[tuple[str, str]] = Counter()
    fn_lines: list[str] = []
    fp_lines: list[str] = []
    for i in range(len(gold)):
        for j in range(i + 1, len(gold)):
            same_gold = gold[i] == gold[j]
            same_pred = pred[i] == pred[j]
            if same_gold and not same_pred:
                fn_bugs[gold[i]] += 1
                fn_lines.append(f"{cases[i]}-{cases[j]} gold={gold[i]} pred={pred[i]}/{pred[j]}")
            elif not same_gold and same_pred:
                pair = tuple(sorted((gold[i], gold[j])))
                fp_pairs[pair] += 1
                fp_lines.append(f"{cases[i]}-{cases[j]} gold={gold[i]}/{gold[j]} pred={pred[i]}")
    return fn_bugs, fp_pairs, fn_lines, fp_lines


def build_error_report(dataset_name: str, input_csv: Path, gold_csv: Path, rows: Sequence[dict], output_dir: Path) -> None:
    cases = _read_cases(input_csv)
    gold = read_gold(gold_csv)
    lines = [f"# Error Analysis: {dataset_name}", ""]
    for row in rows:
        pred_raw = str(row.get("pred_path", "") or "")
        if not pred_raw:
            continue
        pred_path = Path(pred_raw)
        if not pred_path.is_file():
            continue
        with pred_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            pred = [r.get("bucket", "") for r in reader]
        ba, tpr, tnr = pairwise_scores(gold, pred)
        fn_bugs, fp_pairs, fn_lines, fp_lines = _pair_errors(gold, pred, cases)
        lines.extend([
            f"## {row['method']}",
            "",
            f"BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f} clusters={len(set(pred))}",
            "",
            "Cluster composition:",
            *_cluster_composition(gold, pred, cases),
            "",
            "Top FN bugs:",
        ])
        if fn_bugs:
            lines.extend(f"- {bug}: {cnt}" for bug, cnt in fn_bugs.most_common(8))
        else:
            lines.append("- none")
        lines.append("")
        lines.append("Top FP bug pairs:")
        if fp_pairs:
            lines.extend(f"- {a} / {b}: {cnt}" for (a, b), cnt in fp_pairs.most_common(12))
        else:
            lines.append("- none")
        if dataset_name.endswith("benchmark_set_1") or dataset_name == "benchmark_set_1":
            lines.extend([
                "",
                "Set1 case-level wrong merges/splits:",
                "FP pairs:",
            ])
            lines.extend((f"- {line}" for line in fp_lines[:30]) if fp_lines else ["- none"])
            lines.append("FN pairs:")
            lines.extend((f"- {line}" for line in fn_lines[:30]) if fn_lines else ["- none"])
        lines.append("")
    (output_dir / f"error_analysis_{dataset_name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dataset(dataset: Path, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    dataset = dataset.resolve()
    dataset_name = dataset.name
    input_csv = dataset / "input.csv"
    gold_csv = _gold_path(dataset)
    k = len(set(read_gold(gold_csv)))
    rows: list[dict] = []
    debug_rows: list[dict] = []
    methods = set(args.methods)
    base_prob: np.ndarray | None = None
    case_features: list[plf.LLMCaseFeature] | None = None
    anchor_probs: dict[int, np.ndarray] = {}

    if "no_trace_best" in methods or "trace_guided_split" in methods or "trace_policy_zero_shot" in methods:
        prob, feats, note, runtime = build_current_best_probability(args, input_csv)
        base_prob = prob
        case_features = feats
        if prob is not None and "no_trace_best" in methods:
            rows.append(_score_probability(dataset_name, "no_trace_best", prob, input_csv, gold_csv, k, args.output_dir, runtime, notes=note))
        elif prob is None:
            rows.append({
                "dataset": dataset_name,
                "method": "no_trace_best",
                "k": k,
                "cases": len(read_gold(gold_csv)),
                "num_pred_clusters": "",
                "BA": 0.0,
                "TPR": 0.0,
                "TNR": 0.0,
                "runtime_sec": runtime,
                "pred_path": "",
                "prob_path": "",
                "notes": "skipped: " + note,
            })

    if "trace_tail_unsupervised" in methods:
        row, dbg, _ = run_trace_unsupervised(dataset_name, input_csv, gold_csv, k, args.output_dir, "tail", tail_lines=args.tail_lines)
        rows.append(row)
        debug_rows.extend(dbg)

    if "anchor_trace" in methods or "trace_guided_split" in methods:
        for window in args.window_sizes:
            row, dbg, prob = run_trace_unsupervised(dataset_name, input_csv, gold_csv, k, args.output_dir, "anchor", window_size=window, tail_lines=args.tail_lines)
            anchor_probs[int(window)] = prob
            if "anchor_trace" in methods:
                rows.append(row)
                debug_rows.extend(dbg)

    if "trace_guided_split" in methods:
        for window in args.window_sizes:
            prob = anchor_probs.get(int(window))
            if prob is None:
                continue
            rows.append(run_trace_guided_split(dataset_name, input_csv, gold_csv, k, args.output_dir, base_prob, prob, int(window)))

    if "trace_policy_zero_shot" in methods:
        row, dbg = run_trace_policy_zero_shot(dataset_name, input_csv, gold_csv, k, args.output_dir, base_prob, case_features)
        rows.append(row)
        debug_rows.extend(dbg)

    if "existing_trace_embedding" in methods:
        rows.append(run_existing_trace_embedding(dataset_name, input_csv, gold_csv, k, args.output_dir, args))

    build_error_report(dataset_name, input_csv, gold_csv, rows, args.output_dir)
    return rows, debug_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Official-gold zero-shot trace-assisted evaluation (experimental).")
    p.add_argument("--benchmarks", nargs="+", type=Path, default=[
        Path("test_case/problem/benchmark_set_1"),
        Path("test_case/problem/benchmark_set_2"),
    ])
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/official_trace_assisted_eval"))
    p.add_argument("--methods", nargs="+", default=[
        "no_trace_best",
        "trace_tail_unsupervised",
        "anchor_trace",
        "trace_guided_split",
        "trace_policy_zero_shot",
        "existing_trace_embedding",
    ])
    p.add_argument("--window-sizes", nargs="+", type=int, default=[32, 64, 128])
    p.add_argument("--tail-lines", type=int, default=500)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--rich-model-root", type=Path, default=Path("/tmp/input_signal_5seed_top/models/llm_dual_struct_det_summary_dim64"))
    p.add_argument("--model-tag", default="llm_dual_struct_det_summary_dim64")
    p.add_argument("--ensemble-model-dir", type=Path, default=Path("/tmp/pairwise_llm_exp_full/models"))
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--predict-batch-size", type=int, default=100000)
    p.add_argument("--alpha", type=float, default=0.88)
    p.add_argument("--rich-temp", type=float, default=1.15)
    p.add_argument("--ensemble-temp", type=float, default=1.00)
    p.add_argument("--trace-encoder-dir", type=Path, default=Path("/tmp/trace_transformer_encoders/trace_encoder_official"))
    p.add_argument("--trace-embedding-model-root", type=Path, default=Path("/tmp/trace_transformer_exp/models/trace_embedding"))
    p.add_argument("--trace-embedding-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    all_debug: list[dict] = []
    for benchmark in args.benchmarks:
        rows, debug = run_dataset((PROJECT_ROOT / benchmark).resolve() if not benchmark.is_absolute() else benchmark, args)
        all_rows.extend(rows)
        all_debug.extend(debug)
    fields = [
        "dataset", "method", "k", "cases", "num_pred_clusters", "BA", "TPR", "TNR",
        "runtime_sec", "pred_path", "prob_path", "pairs_vetoed", "pairs_boosted",
        "trace_missing_pairs", "split_largest_bucket_size", "notes",
    ]
    _write_csv(args.output_dir / "results.csv", all_rows, fields)
    _write_csv(args.output_dir / "summary.csv", all_rows, fields)
    if all_debug:
        debug_fields = [
            "dataset", "case_id", "method", "anchor_located_by", "anchor_type",
            "target_time", "target_pc", "reason_tags", "window_size", "trace_status",
            "fallback_used", "top_opcodes", "pc_regions",
        ]
        _write_csv(args.output_dir / "trace_debug.csv", all_debug, debug_fields)
        by_dataset: dict[str, list[dict]] = defaultdict(list)
        for row in all_debug:
            by_dataset[str(row.get("dataset", ""))].append(row)
        for dataset, rows in by_dataset.items():
            _write_csv(args.output_dir / f"trace_debug_{dataset}.csv", rows, debug_fields)

    print("\n| dataset | method | BA | TPR | TNR | clusters | notes |")
    print("|---|---|---:|---:|---:|---:|---|")
    for row in all_rows:
        print(
            f"| {row['dataset']} | {row['method']} | {float(row['BA']):.6f} | "
            f"{float(row['TPR']):.6f} | {float(row['TNR']):.6f} | "
            f"{row.get('num_pred_clusters', '')} | {str(row.get('notes', '')).replace('|', '/')[:100]} |"
        )
    print(f"\nResults: {args.output_dir / 'results.csv'}")
    print(f"Summary: {args.output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
