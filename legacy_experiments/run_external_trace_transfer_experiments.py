#!/usr/bin/env python3
"""External trace transfer experiments for official-directed sanitized targets.

Experimental only. The target dataset gold is used only for final scoring and
error analysis; it is never used for training, model selection, or calibration.
The official regr_fail_bucketing.py default path is unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import pairwise_llm_features as plf
import regr_fail_bucketing as rfb
import trace_anchor as ta
from run_experiments import pairwise_scores, read_gold
from run_input_signal_experiments import ENSEMBLE_WEIGHTS, find_ensemble_models
from run_official_directed_trace_eval import build_current_best_probability

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = [
    PROJECT_ROOT / "fake_dataset/first_batch_dataset",
    PROJECT_ROOT / "fake_dataset/stage2_dataset_working",
    PROJECT_ROOT / "fake_dataset/stage3_dataset_32bugs_640cases",
]
DEFAULT_BENCH = [
    PROJECT_ROOT / "test_case/problem/benchmark_set_1",
    PROJECT_ROOT / "test_case/problem/benchmark_set_2",
]
DEFAULT_TARGET = PROJECT_ROOT / "fake_dataset/official_directed_stage1_sanitized_3bugs_85cases"
MANUAL_LABEL_FILES = {
    "benchmark_set_1": PROJECT_ROOT / "benchmark_manual_label/benchmark1/benchmark_set_1_manual_labels.csv",
    "benchmark_set_2": PROJECT_ROOT / "benchmark_manual_label/benchmark2/benchmark_set_2_manual_labels.csv",
}


@dataclass
class DatasetBlock:
    name: str
    input_csv: Path
    labels: list[str]
    source: str
    weak: bool


def _write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _norm_key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _pick_col(fields: Sequence[str], names: Sequence[str], fallback: int = 0) -> str | None:
    by = {_norm_key(f): f for f in fields}
    for name in names:
        key = _norm_key(name)
        if key in by:
            return by[key]
    return fields[fallback] if fields else None


def _read_input_cases(input_csv: Path) -> list[str]:
    rows, fields = rfb.read_csv_rows(input_csv)
    col = _pick_col(fields, ("Case", "case_id", "case", "id"), 0)
    out = []
    for idx, row in enumerate(rows):
        val = str(row.get(col, "") if col else "").strip()
        out.append(val if val else str(idx + 1))
    return out


def _prefixed_labels(source: str, labels: Sequence[str]) -> list[str]:
    return [f"{source}::{label}" for label in labels]


def _read_manual_labels(dataset_dir: Path) -> list[str]:
    input_csv = dataset_dir / "input.csv"
    cases = _read_input_cases(input_csv)
    label_file = MANUAL_LABEL_FILES.get(dataset_dir.name)
    if label_file is None or not label_file.exists():
        raise FileNotFoundError(f"manual weak label CSV not found for {dataset_dir.name}: {label_file}")
    with label_file.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    case_col = _pick_col(fields, ("Case", "case", "case_id", "id"), 0)
    label_col = _pick_col(fields, ("bug", "bucket", "bug_id", "manual_bucket", "label", "gold"), 2 if len(fields) > 2 else 1)
    mapping: dict[str, str] = {}
    for row in rows:
        raw_case = str(row.get(case_col, "")).strip() if case_col else ""
        label = str(row.get(label_col, "")).strip() if label_col else ""
        if not raw_case or not label:
            continue
        keys = {raw_case, raw_case.removeprefix("case_"), f"case_{raw_case}"}
        try:
            keys.add(str(int(raw_case)))
            keys.add(f"case_{int(raw_case)}")
        except ValueError:
            pass
        for key in keys:
            mapping[key] = label
    labels = []
    missing = []
    for case in cases:
        candidates = [case, case.removeprefix("case_"), f"case_{case}"]
        found = ""
        for key in candidates:
            if key in mapping:
                found = mapping[key]
                break
        if not found:
            missing.append(case)
            found = f"missing_manual_{case}"
        labels.append(found)
    if missing:
        print(f"[manual] WARNING {dataset_dir.name}: missing labels for {missing[:8]}", file=sys.stderr)
    print(f"[manual] {dataset_dir.name}: loaded {len(labels)} weak labels from {label_file}", file=sys.stderr)
    return labels


def load_dataset_blocks(train_dirs: Sequence[Path], benchmark_dirs: Sequence[Path], include_benchmark: bool) -> list[DatasetBlock]:
    blocks: list[DatasetBlock] = []
    for ds in train_dirs:
        ds = (PROJECT_ROOT / ds).resolve() if not ds.is_absolute() else ds.resolve()
        labels = read_gold(ds / "gold.csv")
        blocks.append(DatasetBlock(ds.name, ds / "input.csv", _prefixed_labels(ds.name, labels), "gold", False))
        print(f"[data] train {ds.name}: cases={len(labels)} bugs={len(set(labels))}", file=sys.stderr)
    if include_benchmark:
        for ds in benchmark_dirs:
            ds = (PROJECT_ROOT / ds).resolve() if not ds.is_absolute() else ds.resolve()
            labels = _read_manual_labels(ds)
            blocks.append(DatasetBlock(ds.name, ds / "input.csv", _prefixed_labels(ds.name, labels), "manual_weak", True))
            print(f"[data] weak {ds.name}: cases={len(labels)} classes={len(set(labels))}", file=sys.stderr)
    return blocks


def _all_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def sample_pairs_for_blocks(
    blocks: Sequence[DatasetBlock],
    negative_ratio: float,
    random_state: int,
    max_pairs: int,
) -> tuple[list[tuple[int, int]], np.ndarray, list[dict]]:
    rng = np.random.default_rng(random_state)
    pairs_out: list[tuple[int, int]] = []
    y_out: list[float] = []
    stats: list[dict] = []
    offset = 0
    for block in blocks:
        labels = block.labels
        pos: list[tuple[int, int]] = []
        neg: list[tuple[int, int]] = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] == labels[j]:
                    pos.append((offset + i, offset + j))
                else:
                    neg.append((offset + i, offset + j))
        rng.shuffle(pos)
        rng.shuffle(neg)
        max_neg = int(round(max(1, len(pos)) * float(negative_ratio))) if pos else min(len(neg), 1024)
        neg = neg[: min(len(neg), max_neg)]
        pairs = pos + neg
        labels_y = [1.0] * len(pos) + [0.0] * len(neg)
        if max_pairs > 0 and len(pairs) > max_pairs:
            idx = rng.choice(len(pairs), size=max_pairs, replace=False)
            pairs = [pairs[int(i)] for i in idx]
            labels_y = [labels_y[int(i)] for i in idx]
        pairs_out.extend(pairs)
        y_out.extend(labels_y)
        stats.append({
            "dataset": block.name,
            "source": block.source,
            "cases": len(labels),
            "classes": len(set(labels)),
            "positive_pairs": len(pos),
            "sampled_negative_pairs": len(neg),
            "sampled_pairs": len(pairs),
        })
        offset += len(labels)
    return pairs_out, np.asarray(y_out, dtype=np.float32), stats


def _build_anchor_features(blocks: Sequence[DatasetBlock], window_size: int) -> tuple[list[ta.AnchorTraceFeature], list[dict]]:
    inputs = [b.input_csv for b in blocks]
    feats, debug = ta.build_anchor_trace_case_features(inputs, window_size=window_size)
    enriched: list[dict] = []
    idx = 0
    for block in blocks:
        for local_idx, label in enumerate(block.labels):
            row = dict(debug[idx])
            row["source_dataset"] = block.name
            row["gold_or_manual_label"] = label
            row["label_source"] = block.source
            row["weak_label"] = block.weak
            row["fallback_used"] = row.get("located_by") == "tail"
            row["trace_status"] = feats[idx].file_status
            row["target_time"] = row.get("sim_time", "")
            row["target_pc"] = row.get("dut_pc", "")
            row["target_reg"] = row.get("mismatch_register", "")
            row["anchor_type"] = row.get("anchor_source", "")
            row["reason_tags"] = row.get("anchor_tags", "")
            enriched.append(row)
            idx += 1
    return feats, enriched


def _fit_model(model_type: str, X: np.ndarray, y: np.ndarray, random_state: int) -> dict:
    if len(set(y.tolist())) < 2:
        return {"model_type": "constant", "prob": float(np.mean(y)) if len(y) else 0.5}
    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
        model.fit(Xs, y)
        return {"model_type": "logistic", "model": model, "scaler": scaler}
    if model_type == "gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=220,
            max_depth=5,
            learning_rate=0.04,
            l2_regularization=0.01,
            early_stopping=False,
            class_weight="balanced",
            random_state=random_state,
        )
        model.fit(X, y)
        return {"model_type": "gbdt", "model": model, "scaler": None}
    raise ValueError(f"unknown model_type: {model_type}")


def _predict_model(pkg: dict, X: np.ndarray) -> np.ndarray:
    if pkg.get("model_type") == "constant":
        return np.full(X.shape[0], float(pkg.get("prob", 0.5)), dtype=np.float32)
    scaler = pkg.get("scaler")
    if scaler is not None:
        X = scaler.transform(X)
    model = pkg["model"]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(np.float32)
    return np.clip(model.predict(X).astype(np.float32), 1e-5, 1.0 - 1e-5)


def _prob_from_pairs(n: int, pairs: Sequence[tuple[int, int]], scores: np.ndarray) -> np.ndarray:
    prob = np.eye(n, dtype=np.float32)
    for (i, j), p in zip(pairs, scores):
        prob[i, j] = prob[j, i] = float(p)
    return prob


def _cases_for_pred(input_csv: Path) -> list[str]:
    return _read_input_cases(input_csv)


def _write_prediction(path: Path, input_csv: Path, labels: Sequence[int]) -> list[str]:
    cases = _cases_for_pred(input_csv)
    pred = [f"bucket_{int(label):03d}" for label in labels]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Case", "bucket"])
        for case, bucket in zip(cases, pred):
            writer.writerow([case, bucket])
    return pred


def score_probability(
    method: str,
    prob: np.ndarray,
    target_dir: Path,
    output_dir: Path,
    runtime: float,
    notes: str,
    extra: dict | None = None,
) -> dict:
    gold = read_gold(target_dir / "gold.csv")
    k = len(set(gold))
    labels = plf.cluster_from_probability(prob.astype(np.float32), k)
    pred_path = output_dir / "preds" / f"{method}.csv"
    prob_path = output_dir / "probs" / f"{method}.npy"
    pred = _write_prediction(pred_path, target_dir / "input.csv", labels)
    prob_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(prob_path, prob.astype(np.float32))
    ba, tpr, tnr = pairwise_scores(gold, pred)
    row = {
        "method": method,
        "BA": ba,
        "TPR": tpr,
        "TNR": tnr,
        "num_cases": len(gold),
        "k": k,
        "num_pred_clusters": len(set(pred)),
        "runtime_sec": runtime,
        "pred_path": str(pred_path),
        "prob_path": str(prob_path),
        "notes": notes,
    }
    if extra:
        row.update(extra)
    return row


def current_best_probability(args: argparse.Namespace, target_dir: Path) -> tuple[np.ndarray, str, float]:
    class A:
        pass
    a = A()
    a.rich_model_root = args.rich_model_root
    a.ensemble_model_dir = args.ensemble_model_dir
    a.llm_cache_dir = args.llm_cache_dir
    a.svd_dim = args.svd_dim
    a.seeds = args.no_trace_seeds
    a.model_tag = args.model_tag
    a.predict_batch_size = args.predict_batch_size
    a.alpha = args.alpha
    a.rich_temp = args.rich_temp
    a.ensemble_temp = args.ensemble_temp
    prob, _features, note, runtime = build_current_best_probability(a, target_dir / "input.csv")
    if prob is None:
        raise RuntimeError(f"no-trace current-best probability failed: {note}")
    return prob, note, runtime


def train_anchor_transfer(
    blocks: Sequence[DatasetBlock],
    target_dir: Path,
    output_dir: Path,
    config_name: str,
    window_size: int,
    model_type: str,
    random_state: int,
    negative_ratio: float,
    max_pairs_per_dataset: int,
    p_no_trace: np.ndarray | None = None,
    betas: Sequence[float] = (),
) -> tuple[list[dict], list[dict]]:
    t0 = time.perf_counter()
    train_feats, train_debug = _build_anchor_features(blocks, window_size)
    pairs, y, stats = sample_pairs_for_blocks(blocks, negative_ratio, random_state, max_pairs_per_dataset)
    X_train = ta.build_anchor_trace_pair_feature_matrix(train_feats, pairs)
    model = _fit_model(model_type, X_train, y, random_state)
    target_block = DatasetBlock(target_dir.name, target_dir / "input.csv", ["target"] * len(read_gold(target_dir / "gold.csv")), "target_hidden", False)
    target_feats, target_debug = _build_anchor_features([target_block], window_size)
    target_pairs = _all_pairs(len(target_feats))
    X_test = ta.build_anchor_trace_pair_feature_matrix(target_feats, target_pairs)
    p_trace = _prob_from_pairs(len(target_feats), target_pairs, _predict_model(model, X_test))
    method = f"{config_name}_w{window_size}_{model_type}_rs{random_state}"
    rows = [score_probability(
        method,
        p_trace,
        target_dir,
        output_dir,
        time.perf_counter() - t0,
        notes="anchor trace transfer; target gold used only for scoring",
        extra={
            "config": config_name,
            "window_size": window_size,
            "model_type": model_type,
            "random_state": random_state,
            "feature_dim": X_train.shape[1],
            "train_pairs": len(pairs),
            "train_pos_pairs": int(np.sum(y)),
            "train_neg_pairs": int(len(y) - np.sum(y)),
            "train_stats": json.dumps(stats, sort_keys=True),
            "target_located_by": json.dumps(dict(Counter(row.get("located_by") for row in target_debug)), sort_keys=True),
        },
    )]
    if p_no_trace is not None and betas:
        for beta in betas:
            p_final = float(beta) * p_trace + (1.0 - float(beta)) * p_no_trace
            rows.append(score_probability(
                f"anchor_blend_{config_name}_w{window_size}_{model_type}_b{beta:.2f}_rs{random_state}",
                p_final,
                target_dir,
                output_dir,
                time.perf_counter() - t0,
                notes="analysis blend; beta not tuned on target",
                extra={"config": "anchor_blend", "window_size": window_size, "model_type": model_type, "random_state": random_state, "beta": beta},
            ))
    return rows, train_debug + target_debug


def _attach_trace_embeddings(features: list[plf.LLMCaseFeature], input_csvs: Sequence[Path], encoder) -> int:
    from trace_sequence import collect_trace_paths_from_input
    idx = 0
    ok = 0
    for input_csv in input_csvs:
        collected = collect_trace_paths_from_input(input_csv)
        for _case_id, path, status in collected:
            if idx >= len(features):
                break
            if status == "ok" and path is not None:
                features[idx].trace_vec = encoder.encode_trace_tail(str(path))
                ok += 1
            idx += 1
    plf.normalize_trace_vectors(features)
    return ok


def train_trace_encoder_transfer(
    args: argparse.Namespace,
    blocks: Sequence[DatasetBlock],
    target_dir: Path,
    output_dir: Path,
    window_size: int,
    model_type: str,
    random_state: int,
    p_no_trace: np.ndarray | None,
) -> tuple[list[dict], list[dict]]:
    t0 = time.perf_counter()
    try:
        from trace_transformer_pretrain import load_pretrained
    except Exception as exc:
        return [{"method": "trace_encoder_old_fake_benchmark", "BA": 0.0, "TPR": 0.0, "TNR": 0.0, "notes": f"skipped: {exc}"}], []
    if not args.trace_encoder_dir.exists():
        return [{"method": "trace_encoder_old_fake_benchmark", "BA": 0.0, "TPR": 0.0, "TNR": 0.0, "notes": f"skipped missing encoder {args.trace_encoder_dir}"}], []
    encoder = load_pretrained(args.trace_encoder_dir, device=args.device if args.device != "auto" else "cpu")
    llm_args = plf._make_llm_args(
        llm_mode="embedding", llm_doc_style="features", llm_cache_dir=args.llm_cache_dir,
        svd_dim=args.svd_dim, llm_dual=True,
    )
    train_inputs = [b.input_csv for b in blocks]
    rich_train, _ = plf.build_llm_case_features_for_inputs(train_inputs, svd_dim=args.svd_dim, llm_args=llm_args)
    encoded_train = _attach_trace_embeddings(rich_train, train_inputs, encoder)
    llm_reducer = plf.fit_llm_reducer(rich_train, args.llm_reduce_dim, random_state=random_state)
    llm_summary_reducer = plf.fit_llm_summary_reducer(rich_train, args.llm_reduce_dim, random_state=random_state)
    trace_reducer = plf.fit_trace_reducer(rich_train, args.trace_reduce_dim, random_state=random_state)
    anchor_train, train_debug = _build_anchor_features(blocks, window_size)
    pairs, y, stats = sample_pairs_for_blocks(blocks, args.negative_ratio, random_state, args.max_pairs_per_dataset)
    X_rich = plf.build_rich_pair_feature_matrix(rich_train, pairs, feature_mode="llm_dual_struct_det_summary_trace")
    X_anchor = ta.build_anchor_trace_pair_feature_matrix(anchor_train, pairs)
    X_train = np.hstack([X_rich, X_anchor]).astype(np.float32)
    model = _fit_model(model_type, X_train, y, random_state)

    target_input = target_dir / "input.csv"
    rich_test, _ = plf.build_llm_case_features(target_input, svd_dim=args.svd_dim, llm_args=llm_args)
    encoded_test = _attach_trace_embeddings(rich_test, [target_input], encoder)
    plf.apply_llm_reducer(rich_test, llm_reducer, args.llm_reduce_dim)
    plf.apply_llm_summary_reducer(rich_test, llm_summary_reducer, args.llm_reduce_dim)
    plf.apply_trace_reducer(rich_test, trace_reducer, args.trace_reduce_dim)
    target_block = DatasetBlock(target_dir.name, target_input, ["target"] * len(read_gold(target_dir / "gold.csv")), "target_hidden", False)
    anchor_test, target_debug = _build_anchor_features([target_block], window_size)
    target_pairs = _all_pairs(len(rich_test))
    X_test = np.hstack([
        plf.build_rich_pair_feature_matrix(rich_test, target_pairs, feature_mode="llm_dual_struct_det_summary_trace"),
        ta.build_anchor_trace_pair_feature_matrix(anchor_test, target_pairs),
    ]).astype(np.float32)
    p_trace = _prob_from_pairs(len(rich_test), target_pairs, _predict_model(model, X_test))
    rows = [score_probability(
        f"trace_encoder_old_fake_benchmark_w{window_size}_{model_type}_rs{random_state}",
        p_trace,
        target_dir,
        output_dir,
        time.perf_counter() - t0,
        notes="rich+trace-encoder+anchor transfer; target gold used only for scoring",
        extra={
            "config": "trace_encoder_old_fake_benchmark",
            "window_size": window_size,
            "model_type": model_type,
            "random_state": random_state,
            "feature_dim": X_train.shape[1],
            "train_pairs": len(pairs),
            "train_pos_pairs": int(np.sum(y)),
            "train_neg_pairs": int(len(y) - np.sum(y)),
            "trace_encoded_train": encoded_train,
            "trace_encoded_target": encoded_test,
            "train_stats": json.dumps(stats, sort_keys=True),
        },
    )]
    return rows, train_debug + target_debug


def build_error_report(rows: Sequence[dict], target_dir: Path, output_dir: Path) -> None:
    gold = read_gold(target_dir / "gold.csv")
    lines = ["# External Trace Transfer Error Analysis\n", f"Target: `{target_dir}`  ", f"Gold counts: {dict(Counter(gold))}\n"]
    lines += ["| method | BA | TPR | TNR | clusters | notes |", "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        if "BA" not in row:
            continue
        lines.append(f"| {row.get('method','')} | {float(row.get('BA',0)):.6f} | {float(row.get('TPR',0)):.6f} | {float(row.get('TNR',0)):.6f} | {row.get('num_pred_clusters','')} | {str(row.get('notes','')).replace('|','/')} |")
    target_bugs = {"bug_036", "bug_037", "bug_038"}
    for row in rows:
        pred_path = Path(str(row.get("pred_path", "")))
        if not pred_path.exists():
            continue
        with pred_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            pred = [r.get("bucket", "") for r in reader]
        fp_pairs = Counter()
        fn_bugs = Counter()
        for i in range(len(gold)):
            for j in range(i + 1, len(gold)):
                if gold[i] == gold[j] and pred[i] != pred[j]:
                    fn_bugs[gold[i]] += 1
                if gold[i] != gold[j] and pred[i] == pred[j]:
                    fp_pairs[tuple(sorted((gold[i], gold[j])))] += 1
        lines.append(f"\n## {row['method']}\n")
        lines.append("Top FP bug pairs:")
        for pair, cnt in fp_pairs.most_common(8):
            mark = " <= target" if set(pair) <= target_bugs else ""
            lines.append(f"- {pair[0]} / {pair[1]}: {cnt}{mark}")
        lines.append("Top FN bugs:")
        for bug, cnt in fn_bugs.most_common(8):
            lines.append(f"- {bug}: {cnt}")
        if not fp_pairs:
            lines.append("- no FP pairs")
        if not fn_bugs:
            lines.append("- no FN bugs")
    (output_dir / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if "BA" in row:
            groups[str(row["method"])].append(row)
    out = []
    for method, items in sorted(groups.items()):
        out.append({
            "method": method,
            "mean_BA": statistics.mean(float(r["BA"]) for r in items),
            "std_BA": statistics.stdev(float(r["BA"]) for r in items) if len(items) > 1 else 0.0,
            "mean_TPR": statistics.mean(float(r["TPR"]) for r in items),
            "mean_TNR": statistics.mean(float(r["TNR"]) for r in items),
            "num_runs": len(items),
        })
    return out


def run(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    target_dir = (PROJECT_ROOT / args.target_dataset).resolve() if not args.target_dataset.is_absolute() else args.target_dataset.resolve()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    debug_rows: list[dict] = []
    p_no_trace = None
    if "no_trace" in args.configs or "anchor_blend" in args.configs:
        t0 = time.perf_counter()
        p_no_trace, note, runtime = current_best_probability(args, target_dir)
        rows.append(score_probability(
            "no_trace_current_best_external",
            p_no_trace,
            target_dir,
            output_dir,
            runtime,
            notes=note + "; pretrained artifacts only",
            extra={"config": "no_trace", "random_state": "", "window_size": "", "model_type": ""},
        ))
        print(f"[result] no_trace BA={rows[-1]['BA']:.6f}", file=sys.stderr)
    for rs in args.random_states:
        for model_type in args.model_types:
            if "anchor_old_fake" in args.configs or "anchor_blend" in args.configs:
                blocks = load_dataset_blocks(args.train_datasets, args.benchmark_datasets, include_benchmark=False)
                for w in args.window_sizes:
                    do_blend = "anchor_blend" in args.configs
                    r, d = train_anchor_transfer(
                        blocks, target_dir, output_dir, "anchor_old_fake", w, model_type, rs,
                        args.negative_ratio, args.max_pairs_per_dataset,
                        p_no_trace=p_no_trace if do_blend else None,
                        betas=args.betas if do_blend else (),
                    )
                    rows.extend(r); debug_rows.extend(d)
                    print(f"[result] anchor_old_fake w={w} {model_type} rs={rs} BA={r[0]['BA']:.6f}", file=sys.stderr)
            if "anchor_old_fake_benchmark" in args.configs or "trace_encoder_old_fake_benchmark" in args.configs:
                blocks_b = load_dataset_blocks(args.train_datasets, args.benchmark_datasets, include_benchmark=True)
                if "anchor_old_fake_benchmark" in args.configs:
                    for w in args.window_sizes:
                        r, d = train_anchor_transfer(
                            blocks_b, target_dir, output_dir, "anchor_old_fake_benchmark", w, model_type, rs,
                            args.negative_ratio, args.max_pairs_per_dataset,
                        )
                        rows.extend(r); debug_rows.extend(d)
                        print(f"[result] anchor_old_fake_benchmark w={w} {model_type} rs={rs} BA={r[0]['BA']:.6f}", file=sys.stderr)
                if "trace_encoder_old_fake_benchmark" in args.configs:
                    for w in args.window_sizes:
                        r, d = train_trace_encoder_transfer(args, blocks_b, target_dir, output_dir, w, model_type, rs, p_no_trace)
                        rows.extend(r); debug_rows.extend(d)
                        if r:
                            print(f"[result] trace_encoder w={w} {model_type} rs={rs} BA={float(r[0].get('BA',0)):.6f}", file=sys.stderr)
    fields = [
        "method", "config", "window_size", "model_type", "random_state", "beta",
        "BA", "TPR", "TNR", "num_cases", "k", "num_pred_clusters", "runtime_sec",
        "feature_dim", "train_pairs", "train_pos_pairs", "train_neg_pairs",
        "trace_encoded_train", "trace_encoded_target", "target_located_by",
        "pred_path", "prob_path", "notes", "train_stats",
    ]
    _write_csv(output_dir / "results.csv", rows, fields)
    _write_csv(output_dir / "summary.csv", summarize(rows), ["method", "mean_BA", "std_BA", "mean_TPR", "mean_TNR", "num_runs"])
    if debug_rows:
        debug_fields = sorted({key for row in debug_rows for key in row})
        _write_csv(output_dir / "anchor_debug.csv", debug_rows, debug_fields)
    build_error_report(rows, target_dir, output_dir)
    return rows, debug_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="External trace transfer to sanitized target (experimental).")
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/external_trace_transfer_exp"))
    p.add_argument("--train-datasets", nargs="+", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--benchmark-datasets", nargs="+", type=Path, default=DEFAULT_BENCH)
    p.add_argument("--target-dataset", type=Path, default=DEFAULT_TARGET)
    p.add_argument("--configs", nargs="+", default=["no_trace", "anchor_old_fake", "anchor_old_fake_benchmark"])
    p.add_argument("--window-sizes", nargs="+", type=int, default=[64])
    p.add_argument("--model-types", nargs="+", choices=("logistic", "gbdt"), default=["gbdt"])
    p.add_argument("--betas", nargs="+", type=float, default=[0.25, 0.50, 0.75, 1.00])
    p.add_argument("--random-state", type=int, default=None)
    p.add_argument("--random-states", nargs="+", type=int, default=None)
    p.add_argument("--negative-ratio", type=float, default=3.0)
    p.add_argument("--max-pairs-per-dataset", type=int, default=50000)
    p.add_argument("--rich-model-root", type=Path, default=Path("/tmp/input_signal_5seed_top/models/llm_dual_struct_det_summary_dim64"))
    p.add_argument("--model-tag", default="llm_dual_struct_det_summary_dim64")
    p.add_argument("--ensemble-model-dir", type=Path, default=Path("/tmp/pairwise_llm_exp_full/models"))
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--llm-reduce-dim", type=int, default=64)
    p.add_argument("--trace-reduce-dim", type=int, default=32)
    p.add_argument("--predict-batch-size", type=int, default=100000)
    p.add_argument("--alpha", type=float, default=0.88)
    p.add_argument("--rich-temp", type=float, default=1.15)
    p.add_argument("--ensemble-temp", type=float, default=1.00)
    p.add_argument("--no-trace-seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--trace-encoder-dir", type=Path, default=Path("/tmp/trace_transformer_encoders/trace_encoder_official"))
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = p.parse_args(argv)
    if args.random_states is None:
        args.random_states = [args.random_state if args.random_state is not None else 0]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows, _debug = run(args)
    print("\n| method | BA | TPR | TNR | clusters | notes |")
    print("|---|---:|---:|---:|---:|---|")
    for row in rows:
        print(f"| {row.get('method','')} | {float(row.get('BA',0)):.6f} | {float(row.get('TPR',0)):.6f} | {float(row.get('TNR',0)):.6f} | {row.get('num_pred_clusters','')} | {str(row.get('notes','')).replace('|','/')} |")
    print(f"\nResults: {args.output_dir / 'results.csv'}")
    print(f"Summary: {args.output_dir / 'summary.csv'}")
    print(f"Error analysis: {args.output_dir / 'error_analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
