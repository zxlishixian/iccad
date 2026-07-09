#!/usr/bin/env python3
"""Graph-aware clustering and multi-view embedding ablations.

Experimental only.  The formal predictor is not modified.  Stage 1 can reuse
saved P_same matrices from existing LODO runs and replace only the final
clustering step.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import graph_clustering as gc
import official_style_features as osf
import pairwise_llm_features as plf
import regr_fail_bucketing as rfb
import train_pairwise_llm as tpl
from run_experiments import pairwise_scores, read_gold
from run_official_full_retrain_experiments import write_csv, write_pred


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    Path("old_fake_dataset/first_batch_dataset"),
    Path("old_fake_dataset/stage2_dataset_working"),
    Path("old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("official_format_fake_dataset/stable_official_like_multitest_v1"),
    Path("official_format_fake_dataset/directed_cross_v2"),
    Path("test_case/problem/benchmark_set_1"),
    Path("test_case/problem/benchmark_set_2"),
]

VIEW_CONFIGS = {
    "dual": ["features", "summary"],
    "tri_event": ["features", "summary", "event"],
    "tri_object": ["features", "summary", "object"],
    "quad_event_object": ["features", "summary", "event", "object"],
    "quad_event_object_context": ["features", "summary", "event", "object", "context"],
}


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _nonempty_conflict(a: str, b: str) -> float:
    return float(bool(a and b and a != b))


def _read_case_texts(input_csv: Path) -> list[dict]:
    rows, fields = rfb.read_csv_rows(input_csv)
    sim_col = rfb.pick_column(fields, "sim")
    regr_col = rfb.pick_column(fields, "regr")
    case_col = osf.explicit_case_col(fields)
    out: list[dict] = []
    for idx, row in enumerate(rows):
        case_id = osf.infer_case_id(input_csv, row, fields, idx)
        sim_path = rfb.resolve_log_path(input_csv, row.get(sim_col) if sim_col else None)
        regr_path = rfb.resolve_log_path(input_csv, row.get(regr_col) if regr_col else None)
        sim_text, sim_status = rfb.read_log_sample(sim_path)
        regr_text, regr_status = rfb.read_log_sample(regr_path)
        out.append({
            "case_id": case_id,
            "sim_text": sim_text,
            "regr_text": regr_text,
            "sim_status": sim_status,
            "regr_status": regr_status,
        })
    return out


def conflict_matrix_from_records(records: Sequence[osf.OfficialCaseRecord]) -> np.ndarray:
    n = len(records)
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        ai = records[i].info
        for j in range(i + 1, n):
            bi = records[j].info
            value = 0.0
            value += 0.45 * _nonempty_conflict(str(ai.get("primary_type", "")), str(bi.get("primary_type", "")))
            value += 0.35 * _nonempty_conflict(str(ai.get("mismatch_type", "")), str(bi.get("mismatch_type", "")))
            value += 0.35 * _nonempty_conflict(str(ai.get("op_pair", "")), str(bi.get("op_pair", "")))
            value += 0.25 * _nonempty_conflict(str(ai.get("primary_signature", "")), str(bi.get("primary_signature", "")))
            value += 0.20 * _nonempty_conflict(str(ai.get("register_name", "")), str(bi.get("register_name", "")))
            matrix[i, j] = matrix[j, i] = min(1.0, value)
    return matrix


def _find_prob_file(
    prob_dirs: Sequence[Path],
    dataset_name: str,
    model_arch: str,
    source_config: str,
    source_graph_tag: str,
    seed: int,
) -> Path | None:
    filename = f"{dataset_name}_{model_arch}_{source_config}_{source_graph_tag}_seed{seed}.npy"
    for root in prob_dirs:
        candidates = [
            root / "probs" / filename,
            root / filename,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def _pair_bug_counts(labels: Sequence[str], pred: Sequence[str]) -> tuple[list[dict], list[dict], list[dict]]:
    by_bucket: dict[str, Counter] = defaultdict(Counter)
    by_bug: dict[str, Counter] = defaultdict(Counter)
    for gold, bucket in zip(labels, pred):
        by_bucket[str(bucket)][str(gold)] += 1
        by_bug[str(gold)][str(bucket)] += 1
    mixed = []
    for bucket, counts in by_bucket.items():
        total = sum(counts.values())
        top_bug, top_count = counts.most_common(1)[0]
        mixed.append({
            "kind": "mixed_cluster",
            "bucket": bucket,
            "total": total,
            "top_label": top_bug,
            "top_count": top_count,
            "purity": top_count / total if total else 0.0,
            "labels": json.dumps(dict(counts), sort_keys=True),
        })
    fragmented = []
    for bug, counts in by_bug.items():
        total = sum(counts.values())
        top_bucket, top_count = counts.most_common(1)[0]
        fragmented.append({
            "kind": "fragmented_bug",
            "bug": bug,
            "total": total,
            "num_buckets": len(counts),
            "top_bucket": top_bucket,
            "top_count": top_count,
            "coverage": top_count / total if total else 0.0,
            "buckets": json.dumps(dict(counts), sort_keys=True),
        })
    fp_pairs: Counter[tuple[str, str]] = Counter()
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if pred[i] == pred[j] and labels[i] != labels[j]:
                a, b = sorted((str(labels[i]), str(labels[j])))
                fp_pairs[(a, b)] += 1
    false_merges = [
        {
            "kind": "false_merge_pair",
            "bug_a": a,
            "bug_b": b,
            "pair_count": count,
        }
        for (a, b), count in fp_pairs.most_common(20)
    ]
    return (
        sorted(mixed, key=lambda row: (float(row["purity"]), -int(row["total"])))[:20],
        sorted(fragmented, key=lambda row: (-int(row["num_buckets"]), float(row["coverage"])))[:20],
        false_merges,
    )


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(
            str(row["graph_method"]),
            str(row["view_config"]),
            str(row["view_fusion"]),
            str(row["source_config"]),
        )].append(row)
    out: list[dict] = []
    for (graph_method, view_config, view_fusion, source_config), values in groups.items():
        dataset_means = {}
        for dataset in sorted({str(v["dataset"]) for v in values}):
            dataset_rows = [v for v in values if v["dataset"] == dataset]
            dataset_means[dataset] = float(np.mean([float(v["BA"]) for v in dataset_rows]))
        bas = list(dataset_means.values())
        out.append({
            "graph_method": graph_method,
            "view_config": view_config,
            "view_fusion": view_fusion,
            "source_config": source_config,
            "mean_BA": float(np.mean(bas)) if bas else 0.0,
            "std_BA": float(np.std(bas)) if bas else 0.0,
            "worst_BA": float(np.min(bas)) if bas else 0.0,
            "mean_TPR": float(np.mean([float(v["TPR"]) for v in values])) if values else 0.0,
            "mean_TNR": float(np.mean([float(v["TNR"]) for v in values])) if values else 0.0,
            "runtime_mean": float(np.mean([float(v["runtime_sec"]) for v in values])) if values else 0.0,
            "datasets": len(dataset_means),
            "runs": len(values),
            "dataset_means": json.dumps(dataset_means, sort_keys=True),
        })
    return sorted(out, key=lambda row: (row["mean_BA"], row["worst_BA"], row["mean_TNR"]), reverse=True)


def _signal_lines(text: str, max_lines: int = 8) -> list[str]:
    lines = rfb.select_lines(text, mode="signal_window")
    compact = []
    for line in lines:
        stripped = re.sub(r"\s+", " ", line.strip())
        if stripped and stripped not in compact:
            compact.append(stripped[:220])
        if len(compact) >= max_lines:
            break
    return compact


def build_view_docs(dataset: Path, view_config: str, max_examples: int = 20) -> list[dict]:
    records = osf.build_case_records(dataset.name, dataset / "input.csv", gold_csv=None)
    texts = _read_case_texts(dataset / "input.csv")
    rows: list[dict] = []
    for record, text in list(zip(records, texts))[:max_examples]:
        info = record.info
        views: dict[str, str] = {}
        views["event"] = "\n".join([
            "EVENT_SEQUENCE:",
            f"primary_type: {record.primary_type}",
            f"mismatch_type: {record.mismatch_type}",
            f"root_tags: {', '.join(sorted(record.root_tags))}",
            f"has_timeout: {'timeout' in (text['sim_text'] + text['regr_text']).lower()}",
            f"has_debug: {'debug' in (text['sim_text'] + text['regr_text']).lower()}",
            f"has_irq: {bool(re.search(r'irq|interrupt|HANDLING_IRQ', text['sim_text'] + text['regr_text'], re.I))}",
        ])
        views["object"] = "\n".join([
            "OBJECTS:",
            f"pc_region: {info.get('pc_region', '')}",
            f"op_pair: {info.get('op_pair', '')}",
            f"ibex_opcode: {info.get('ibex_opcode', '')}",
            f"spike_opcode: {info.get('spike_opcode', '')}",
            f"register: {info.get('register_name', '')}",
            f"fatal_file: {info.get('fatal_file', '')}",
            f"source_file: {info.get('error_source_file', '')}",
            f"uvm_component: {info.get('uvm_component', '')}",
        ])
        views["context"] = "\n".join([
            "SIM_CONTEXT:",
            *[f"- {line}" for line in _signal_lines(text["sim_text"], 5)],
            "REGR_CONTEXT:",
            *[f"- {line}" for line in _signal_lines(text["regr_text"], 6)],
        ])
        enabled = ["features", "summary"]
        if "event" in view_config:
            enabled.append("event")
        if "object" in view_config:
            enabled.append("object")
        if "context" in view_config:
            enabled.append("context")
        rows.append({
            "dataset": dataset.name,
            "case_id": record.case_id,
            "view_config": view_config,
            "enabled_views": enabled,
            "event_doc": views["event"],
            "object_doc": views["object"],
            "context_doc": views["context"],
        })
    return rows


def views_for_config(view_config: str) -> list[str]:
    if view_config not in VIEW_CONFIGS:
        raise ValueError(f"unknown view_config: {view_config}")
    return list(VIEW_CONFIGS[view_config])


def build_all_view_documents(dataset: Path) -> dict[str, list[str]]:
    records = osf.build_case_records(dataset.name, dataset / "input.csv", gold_csv=None)
    texts = _read_case_texts(dataset / "input.csv")
    docs = {"event": [], "object": [], "context": []}
    for record, text in zip(records, texts):
        info = record.info
        joined = text["sim_text"] + "\n" + text["regr_text"]
        event_lines = [
            "EVENT_SEQUENCE:",
            f"primary_type: {record.primary_type}",
            f"mismatch_type: {record.mismatch_type}",
            f"root_tags: {', '.join(sorted(record.root_tags))}",
            f"anchor_source: {record.anchor.source}",
            f"anchor_tags: {', '.join(sorted(record.anchor.anchor_tags))}",
            f"has_timeout: {'timeout' in joined.lower()}",
            f"has_debug: {'debug' in joined.lower()}",
            f"has_irq: {bool(re.search(r'irq|interrupt|HANDLING_IRQ', joined, re.I))}",
            f"has_csr: {bool(re.search(r'csr|mcause|mstatus|dcsr', joined, re.I))}",
            f"has_illegal: {bool(re.search(r'illegal', joined, re.I))}",
        ]
        object_lines = [
            "OBJECTS:",
            f"pc_region: {info.get('pc_region', '')}",
            f"op_pair: {info.get('op_pair', '')}",
            f"ibex_opcode: {info.get('ibex_opcode', '')}",
            f"spike_opcode: {info.get('spike_opcode', '')}",
            f"register: {info.get('register_name', '')}",
            f"fatal_file: {info.get('fatal_file', '')}",
            f"error_source_file: {info.get('error_source_file', '')}",
            f"uvm_component: {info.get('uvm_component', '')}",
            f"uvm_testname: {info.get('uvm_testname', '') or record.test_name}",
            f"dut_pc: {record.anchor.dut_pc}",
            f"iss_pc: {record.anchor.iss_pc}",
            f"target_register: {getattr(record.anchor, 'target_register', getattr(record.anchor, 'mismatch_register', ''))}",
        ]
        context_lines = [
            "SIM_CONTEXT:",
            *[f"- {line}" for line in _signal_lines(text["sim_text"], 6)],
            "REGR_CONTEXT:",
            *[f"- {line}" for line in _signal_lines(text["regr_text"], 8)],
        ]
        docs["event"].append("\n".join(event_lines))
        docs["object"].append("\n".join(object_lines))
        docs["context"].append("\n".join(context_lines))
    return docs


def make_embedding_args(args: argparse.Namespace) -> argparse.Namespace:
    return plf._make_llm_args(
        llm_mode="embedding",
        llm_fusion="concat",
        llm_weight=4.0,
        llm_doc_style="features",
        llm_doc_max_features=args.llm_doc_max_features,
        llm_cache_dir=args.llm_cache_dir,
        llm_batch_size=args.llm_batch_size,
        llm_timeout_sec=args.llm_timeout_sec,
        svd_dim=args.svd_dim,
        llm_dual=True,
    )


def fetch_view_embeddings(docs: Sequence[str], args: argparse.Namespace, view_name: str) -> np.ndarray:
    if not docs:
        return np.zeros((0, 0), dtype=np.float32)
    if rfb.load_llm_embedding_config() is None:
        print(f"[view] LLM embedding config missing; {view_name} view uses zero vectors", flush=True)
        return np.zeros((len(docs), 0), dtype=np.float32)
    llm_args = make_embedding_args(args)
    llm_args.llm_dual = False
    try:
        mat, model_name = rfb.fetch_llm_embeddings(list(docs), llm_args)
        arr = np.asarray(mat, dtype=np.float32)
        from sklearn.preprocessing import Normalizer
        arr = Normalizer(copy=False).fit_transform(arr)
        print(f"[view] {view_name} model={model_name} embedding_dim={arr.shape[1]} docs={len(docs)}", flush=True)
        return arr.astype(np.float32, copy=False)
    except Exception as exc:
        print(f"[view] {view_name} embedding failed ({exc}); using zeros", flush=True)
        return np.zeros((len(docs), 0), dtype=np.float32)


def fit_apply_reduced_matrix(raw: np.ndarray, train_indices: Sequence[int], dim: int, seed: int) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    if dim <= 0 or raw.size == 0 or raw.shape[1] == 0:
        return np.zeros((raw.shape[0], 0), dtype=np.float32)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    reducer, _ = plf._fit_reducer_for_matrix(raw[train_indices], int(dim), seed)
    return plf._apply_reducer_to_matrix(raw, reducer, int(dim)).astype(np.float32, copy=False)


def relation_block_with_scalars(mat: np.ndarray, i: int, j: int) -> np.ndarray:
    if mat.size == 0 or mat.shape[1] == 0:
        return np.zeros(0, dtype=np.float32)
    a = mat[i]
    b = mat[j]
    dot = float(np.dot(a, b))
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    cosine = dot / denom
    euclidean = float(np.linalg.norm(a - b))
    return np.concatenate([
        np.abs(a - b),
        a * b,
        np.asarray([cosine, euclidean], dtype=np.float32),
    ]).astype(np.float32, copy=False)


def build_multiview_pair_feature_matrix(
    features: list[plf.LLMCaseFeature],
    view_mats: dict[str, np.ndarray],
    view_names: Sequence[str],
    pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    mats: dict[str, np.ndarray] = dict(view_mats)
    if "features" in view_names:
        mats["features"] = np.vstack([f.effective_llm_vec for f in features]).astype(np.float32)
    if "summary" in view_names:
        mats["summary"] = np.vstack([f.effective_llm_summary_vec for f in features]).astype(np.float32)
    if not pairs:
        sample = build_multiview_pair_feature_matrix(features, view_mats, view_names, [(0, 0)]) if features else np.zeros((1, 0), dtype=np.float32)
        return np.zeros((0, sample.shape[1]), dtype=np.float32)
    sample_blocks = []
    i0, j0 = pairs[0]
    for view in view_names:
        sample_blocks.append(relation_block_with_scalars(mats.get(view, np.zeros((len(features), 0), dtype=np.float32)), i0, j0))
    sample_blocks.append(plf.build_structured_pair_feature_vector(features[i0], features[j0]))
    sample_blocks.append(plf.build_det_scalar_summary_vector(features[i0], features[j0]))
    dim = int(sum(len(block) for block in sample_blocks))
    out = np.empty((len(pairs), dim), dtype=np.float32)
    for row, (i, j) in enumerate(pairs):
        blocks = [relation_block_with_scalars(mats.get(view, np.zeros((len(features), 0), dtype=np.float32)), i, j) for view in view_names]
        blocks.append(plf.build_structured_pair_feature_vector(features[i], features[j]))
        blocks.append(plf.build_det_scalar_summary_vector(features[i], features[j]))
        out[row] = np.concatenate(blocks).astype(np.float32, copy=False)
    return out


def train_view_model(X: np.ndarray, y: np.ndarray, model_type: str, seed: int) -> dict:
    if model_type == "logistic":
        return plf.train_logistic_model(X, y, random_state=seed)
    if model_type == "gbdt":
        return plf.train_gbdt_model(X, y, random_state=seed)
    raise ValueError(f"unknown view model type: {model_type}")


def predict_view_probabilities(model_pkg: dict, X: np.ndarray, pairs: Sequence[tuple[int, int]], n: int) -> np.ndarray:
    model = model_pkg["model"]
    scaler = model_pkg.get("scaler")
    X_eff = scaler.transform(X) if scaler is not None else X
    if hasattr(model, "predict_proba"):
        flat = model.predict_proba(X_eff)[:, 1].astype(np.float32)
    else:
        flat = np.clip(model.predict(X_eff).astype(np.float32), 1e-6, 1.0 - 1e-6)
    prob = np.eye(n, dtype=np.float32)
    for (i, j), value in zip(pairs, flat):
        prob[i, j] = prob[j, i] = float(value)
    return prob


def sample_lodo_train_pairs(
    features: list[plf.LLMCaseFeature],
    slices: Sequence[dict],
    holdout_name: str,
    args: argparse.Namespace,
    seed: int,
) -> tuple[list[tuple[int, int]], np.ndarray, list[dict]]:
    all_pairs: list[tuple[int, int]] = []
    all_y: list[np.ndarray] = []
    stats: list[dict] = []
    for ds_idx, ds in enumerate(slices):
        if ds["name"] == holdout_name:
            continue
        start, stop = ds["start"], ds["stop"]
        local_pairs, y, pair_stats = tpl.sample_pairs(
            features[start:stop],
            ds["labels"],
            negative_ratio=args.view_negative_ratio,
            hard_negative_ratio=args.view_hard_negative_ratio,
            hard_positive_ratio=args.view_hard_positive_ratio,
            max_train_pairs=args.view_max_pairs_per_dataset,
            random_state=seed * 1009 + ds_idx * 97 + 31,
            positive_sampling="diverse",
            negative_sampling="confusable",
        )
        all_pairs.extend((i + start, j + start) for i, j in local_pairs)
        all_y.append(y)
        item = dict(pair_stats)
        item.update({"dataset": ds["name"], "pairs": len(y)})
        stats.append(item)
    return all_pairs, np.concatenate(all_y).astype(np.float32), stats


def run_view_stage(args: argparse.Namespace, datasets: Sequence[Path]) -> tuple[list[dict], list[dict]]:
    inputs = [dataset / "input.csv" for dataset in datasets]
    llm_args = make_embedding_args(args)
    features, _ = plf.build_llm_case_features_for_inputs(inputs, parser=args.parser, svd_dim=args.svd_dim, llm_args=llm_args)

    slices: list[dict] = []
    offset = 0
    for dataset in datasets:
        labels = read_gold(osf.gold_path(dataset))
        cases = osf.read_cases(dataset / "input.csv")
        slices.append({"name": dataset.name, "path": dataset, "start": offset, "stop": offset + len(labels), "labels": labels, "cases": cases})
        offset += len(labels)

    needed_custom_views = sorted({view for config in args.embedding_view_configs for view in views_for_config(config) if view not in {"features", "summary"}})
    raw_custom: dict[str, np.ndarray] = {}
    if needed_custom_views:
        docs_by_view = {view: [] for view in needed_custom_views}
        for dataset in datasets:
            dataset_docs = build_all_view_documents(dataset)
            for view in needed_custom_views:
                docs_by_view[view].extend(dataset_docs[view])
        for view in needed_custom_views:
            raw_custom[view] = fetch_view_embeddings(docs_by_view[view], args, view)

    rows: list[dict] = []
    diagnostics_rows: list[dict] = []
    for seed in args.seeds:
        for holdout in slices:
            train_indices = [idx for ds in slices if ds["name"] != holdout["name"] for idx in range(ds["start"], ds["stop"])]
            train_features = [features[idx] for idx in train_indices]
            hold_indices = list(range(holdout["start"], holdout["stop"]))
            hold_features = [features[idx] for idx in hold_indices]
            feature_reducer = plf.fit_llm_reducer(train_features, args.view_dim, random_state=seed)
            summary_reducer = plf.fit_llm_summary_reducer(train_features, args.view_dim, random_state=seed + 17)
            plf.apply_llm_reducer(hold_features, feature_reducer, args.view_dim)
            plf.apply_llm_summary_reducer(hold_features, summary_reducer, args.view_dim)
            reduced_custom = {view: fit_apply_reduced_matrix(raw, train_indices, args.view_dim, seed + 101) for view, raw in raw_custom.items()}
            train_pairs, y, pair_stats = sample_lodo_train_pairs(features, slices, holdout["name"], args, seed)
            hold_pairs = osf.all_pairs(len(hold_features))
            hold_conflict = conflict_matrix_from_records(osf.build_case_records(holdout["name"], holdout["path"] / "input.csv", gold_csv=None))
            k = len(set(holdout["labels"]))
            for view_config in args.embedding_view_configs:
                view_names = views_for_config(view_config)
                train_X = build_multiview_pair_feature_matrix(features, reduced_custom, view_names, train_pairs)
                hold_custom = {view: mat[hold_indices] for view, mat in reduced_custom.items()}
                hold_X = build_multiview_pair_feature_matrix(hold_features, hold_custom, view_names, hold_pairs)
                for model_type in args.view_model_types:
                    t0 = time.perf_counter()
                    model_pkg = train_view_model(train_X, y, model_type, seed)
                    prob = predict_view_probabilities(model_pkg, hold_X, hold_pairs, len(hold_features))
                    result = gc.cluster_probability_graph(
                        prob,
                        k,
                        args.view_graph_method,
                        conflict_matrix=hold_conflict,
                        signed_conflict_penalty=args.signed_conflict_penalty,
                        signed_max_iter=args.signed_max_iter,
                        signed_keep_k=args.signed_keep_k,
                    )
                    runtime = time.perf_counter() - t0
                    pred_path = args.output_dir / "preds" / f"{holdout['name']}_{view_config}_{model_type}_{args.view_graph_method}_seed{seed}.csv"
                    pred = write_pred(pred_path, holdout["cases"], result.labels)
                    prob_path = args.output_dir / "probs" / f"{holdout['name']}_{view_config}_{model_type}_{args.view_graph_method}_seed{seed}.npy"
                    prob_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(prob_path, prob)
                    ba, tpr, tnr = pairwise_scores(holdout["labels"], pred)
                    row = {
                        "dataset": holdout["name"],
                        "seed": seed,
                        "source_config": f"view_{model_type}",
                        "source_prob_path": "",
                        "graph_method": args.view_graph_method,
                        "view_config": view_config,
                        "view_fusion": "concat",
                        "BA": ba,
                        "TPR": tpr,
                        "TNR": tnr,
                        "k": k,
                        "cases": len(holdout["labels"]),
                        "num_clusters": len(set(pred)),
                        "num_merges": result.num_merges,
                        "num_splits": result.num_splits,
                        "runtime_sec": runtime,
                        "pred_path": str(pred_path),
                        "prob_path": str(prob_path),
                        "notes": json.dumps({"views": view_names, "pair_stats": pair_stats[:3]}, sort_keys=True),
                    }
                    rows.append(row)
                    mixed, fragmented, false_merges = _pair_bug_counts(holdout["labels"], pred)
                    for rank, diag in enumerate(mixed + fragmented + false_merges, 1):
                        diagnostics_rows.append({
                            "dataset": holdout["name"],
                            "seed": seed,
                            "source_config": f"view_{model_type}",
                            "graph_method": args.view_graph_method,
                            "view_config": view_config,
                            "rank": rank,
                            **diag,
                        })
                    print(
                        f"[view] dataset={holdout['name']} seed={seed} config={view_config} "
                        f"model={model_type} graph={args.view_graph_method} BA={ba:.6f} "
                        f"TPR={tpr:.6f} TNR={tnr:.6f}",
                        flush=True,
                    )
    return rows, diagnostics_rows


def predict_view_probabilities_flat(model_pkg: dict, X: np.ndarray) -> np.ndarray:
    model = model_pkg["model"]
    scaler = model_pkg.get("scaler")
    X_eff = scaler.transform(X) if scaler is not None else X
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_eff)[:, 1].astype(np.float32)
    return np.clip(model.predict(X_eff).astype(np.float32), 1e-6, 1.0 - 1e-6)


def probability_matrix_from_flat(pairs: Sequence[tuple[int, int]], values: Sequence[float], n: int) -> np.ndarray:
    prob = np.eye(n, dtype=np.float32)
    for (i, j), value in zip(pairs, values):
        prob[i, j] = prob[j, i] = float(value)
    return prob


def pair_gate_feature_matrix(
    features: list[plf.LLMCaseFeature],
    pairs: Sequence[tuple[int, int]],
    p_base: np.ndarray,
    p_expert: np.ndarray,
) -> np.ndarray:
    rows = np.empty((len(pairs), 18), dtype=np.float32)
    eps = 1e-6
    for row, (i, j) in enumerate(pairs):
        pb = float(np.clip(p_base[row], eps, 1.0 - eps))
        pe = float(np.clip(p_expert[row], eps, 1.0 - eps))
        fb = plf.build_structured_pair_feature_vector(features[i], features[j])
        ds = plf.build_det_scalar_summary_vector(features[i], features[j])
        ent_b = -(pb * np.log(pb) + (1.0 - pb) * np.log(1.0 - pb))
        ent_e = -(pe * np.log(pe) + (1.0 - pe) * np.log(1.0 - pe))
        rows[row] = np.asarray([
            pb,
            pe,
            pe - pb,
            abs(pe - pb),
            min(pb, pe),
            max(pb, pe),
            ent_b,
            ent_e,
            abs(ent_e - ent_b),
            float(pb > 0.5),
            float(pe > 0.5),
            float((pb > 0.5) != (pe > 0.5)),
            float(fb[11]) if len(fb) > 11 else 0.0,  # same primary signature
            float(fb[12]) if len(fb) > 12 else 0.0,  # same primary type
            float(fb[16]) if len(fb) > 16 else 0.0,  # mismatch conflict
            float(fb[17]) if len(fb) > 17 else 0.0,  # primary conflict
            float(ds[0]) if len(ds) > 0 else 0.0,    # det cosine
            float(ds[2]) if len(ds) > 2 else 0.0,    # token jaccard
        ], dtype=np.float32)
    return rows


def train_pair_gate(gate_X: np.ndarray, y: np.ndarray, p_base: np.ndarray, p_expert: np.ndarray, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y, dtype=np.float32)
    base_err = np.abs(p_base.astype(np.float32) - y)
    expert_err = np.abs(p_expert.astype(np.float32) - y)
    target = (expert_err + 1e-6 < base_err).astype(np.int32)
    if len(np.unique(target)) < 2:
        return {"kind": "constant", "value": float(np.mean(target)) if len(target) else 0.0}
    weights = np.clip(np.abs(base_err - expert_err), 0.05, 1.0).astype(np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(gate_X)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    model.fit(Xs, target, sample_weight=weights)
    return {"kind": "logistic", "model": model, "scaler": scaler, "target_mean": float(np.mean(target)), "weight_mean": float(np.mean(weights))}


def predict_pair_gate(gate_pkg: dict, gate_X: np.ndarray) -> np.ndarray:
    if gate_pkg.get("kind") == "constant":
        return np.full(len(gate_X), float(gate_pkg.get("value", 0.0)), dtype=np.float32)
    Xs = gate_pkg["scaler"].transform(gate_X)
    return gate_pkg["model"].predict_proba(Xs)[:, 1].astype(np.float32)


def train_eval_gate_for_holdout(
    features: list[plf.LLMCaseFeature],
    reduced_custom: dict[str, np.ndarray],
    slices: Sequence[dict],
    holdout: dict,
    base_view: str,
    expert_view: str,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed * 100003 + len(holdout["labels"]))
    train_pairs, y, pair_stats = sample_lodo_train_pairs(features, slices, holdout["name"], args, seed)
    y = np.asarray(y, dtype=np.float32)
    pos = np.flatnonzero(y > 0.5)
    neg = np.flatnonzero(y < 0.5)
    gate_idx_parts = []
    for idxs in (pos, neg):
        if len(idxs):
            take = max(1, int(round(len(idxs) * args.gate_validation_fraction)))
            gate_idx_parts.append(rng.choice(idxs, size=min(len(idxs), take), replace=False))
    gate_idx = np.unique(np.concatenate(gate_idx_parts)) if gate_idx_parts else np.arange(0, min(len(y), args.gate_min_val_pairs))
    if len(gate_idx) < args.gate_min_val_pairs and len(y) > len(gate_idx):
        extra = rng.choice(np.setdiff1d(np.arange(len(y)), gate_idx), size=min(args.gate_min_val_pairs - len(gate_idx), len(y) - len(gate_idx)), replace=False)
        gate_idx = np.unique(np.concatenate([gate_idx, extra]))
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[gate_idx] = False
    model_idx = np.flatnonzero(train_mask)
    if len(model_idx) < 100:
        model_idx = np.arange(len(y))
    base_views = views_for_config(base_view)
    expert_views = views_for_config(expert_view)
    X_base_all = build_multiview_pair_feature_matrix(features, reduced_custom, base_views, train_pairs)
    X_expert_all = build_multiview_pair_feature_matrix(features, reduced_custom, expert_views, train_pairs)
    base_model = train_view_model(X_base_all[model_idx], y[model_idx], args.gate_model_type, seed)
    expert_model = train_view_model(X_expert_all[model_idx], y[model_idx], args.gate_model_type, seed + 53)
    gate_pairs = [train_pairs[int(idx)] for idx in gate_idx]
    p_base_gate = predict_view_probabilities_flat(base_model, X_base_all[gate_idx])
    p_expert_gate = predict_view_probabilities_flat(expert_model, X_expert_all[gate_idx])
    gate_X = pair_gate_feature_matrix(features, gate_pairs, p_base_gate, p_expert_gate)
    gate_pkg = train_pair_gate(gate_X, y[gate_idx], p_base_gate, p_expert_gate, seed)

    hold_indices = list(range(holdout["start"], holdout["stop"]))
    hold_features = [features[idx] for idx in hold_indices]
    hold_custom = {view: mat[hold_indices] for view, mat in reduced_custom.items()}
    hold_pairs = osf.all_pairs(len(hold_features))
    hold_X_base = build_multiview_pair_feature_matrix(hold_features, hold_custom, base_views, hold_pairs)
    hold_X_expert = build_multiview_pair_feature_matrix(hold_features, hold_custom, expert_views, hold_pairs)
    p_base_hold = predict_view_probabilities_flat(base_model, hold_X_base)
    p_expert_hold = predict_view_probabilities_flat(expert_model, hold_X_expert)
    hold_gate_X = pair_gate_feature_matrix(hold_features, hold_pairs, p_base_hold, p_expert_hold)
    gate = predict_pair_gate(gate_pkg, hold_gate_X)
    p_final = (1.0 - gate) * p_base_hold + gate * p_expert_hold
    prob = probability_matrix_from_flat(hold_pairs, p_final, len(hold_features))
    debug = {
        "gate_kind": gate_pkg.get("kind", "unknown"),
        "gate_mean": float(np.mean(gate)) if len(gate) else 0.0,
        "gate_p25": float(np.quantile(gate, 0.25)) if len(gate) else 0.0,
        "gate_p75": float(np.quantile(gate, 0.75)) if len(gate) else 0.0,
        "gate_val_pairs": int(len(gate_idx)),
        "gate_target_mean": float(gate_pkg.get("target_mean", 0.0)),
        "p_base_mean": float(np.mean(p_base_hold)) if len(p_base_hold) else 0.0,
        "p_expert_mean": float(np.mean(p_expert_hold)) if len(p_expert_hold) else 0.0,
        "p_final_mean": float(np.mean(p_final)) if len(p_final) else 0.0,
        "pair_stats": pair_stats[:3],
    }
    return {"prob": prob, "debug": debug}, pair_stats


def run_view_gate_stage(args: argparse.Namespace, datasets: Sequence[Path]) -> tuple[list[dict], list[dict], list[dict]]:
    inputs = [dataset / "input.csv" for dataset in datasets]
    llm_args = make_embedding_args(args)
    features, _ = plf.build_llm_case_features_for_inputs(inputs, parser=args.parser, svd_dim=args.svd_dim, llm_args=llm_args)
    slices: list[dict] = []
    offset = 0
    for dataset in datasets:
        labels = read_gold(osf.gold_path(dataset))
        cases = osf.read_cases(dataset / "input.csv")
        slices.append({"name": dataset.name, "path": dataset, "start": offset, "stop": offset + len(labels), "labels": labels, "cases": cases})
        offset += len(labels)

    needed_custom_views = sorted({view for config in args.gate_expert_view_configs for view in views_for_config(config) if view not in {"features", "summary"}})
    raw_custom: dict[str, np.ndarray] = {}
    if needed_custom_views:
        docs_by_view = {view: [] for view in needed_custom_views}
        for dataset in datasets:
            dataset_docs = build_all_view_documents(dataset)
            for view in needed_custom_views:
                docs_by_view[view].extend(dataset_docs[view])
        for view in needed_custom_views:
            raw_custom[view] = fetch_view_embeddings(docs_by_view[view], args, view)

    rows: list[dict] = []
    diagnostics_rows: list[dict] = []
    gate_debug_rows: list[dict] = []
    for seed in args.seeds:
        for holdout in slices:
            train_indices = [idx for ds in slices if ds["name"] != holdout["name"] for idx in range(ds["start"], ds["stop"])]
            train_features = [features[idx] for idx in train_indices]
            hold_indices = list(range(holdout["start"], holdout["stop"]))
            hold_features = [features[idx] for idx in hold_indices]
            feature_reducer = plf.fit_llm_reducer(train_features, args.view_dim, random_state=seed)
            summary_reducer = plf.fit_llm_summary_reducer(train_features, args.view_dim, random_state=seed + 17)
            plf.apply_llm_reducer(hold_features, feature_reducer, args.view_dim)
            plf.apply_llm_summary_reducer(hold_features, summary_reducer, args.view_dim)
            reduced_custom = {view: fit_apply_reduced_matrix(raw, train_indices, args.view_dim, seed + 101) for view, raw in raw_custom.items()}
            hold_conflict = conflict_matrix_from_records(osf.build_case_records(holdout["name"], holdout["path"] / "input.csv", gold_csv=None))
            k = len(set(holdout["labels"]))
            for expert_view in args.gate_expert_view_configs:
                t0 = time.perf_counter()
                outcome, _pair_stats = train_eval_gate_for_holdout(
                    features, reduced_custom, slices, holdout,
                    args.gate_base_view_config, expert_view, args, seed,
                )
                prob = outcome["prob"]
                debug = outcome["debug"]
                result = gc.cluster_probability_graph(
                    prob, k, args.view_graph_method,
                    conflict_matrix=hold_conflict,
                    signed_conflict_penalty=args.signed_conflict_penalty,
                    signed_max_iter=args.signed_max_iter,
                    signed_keep_k=args.signed_keep_k,
                )
                runtime = time.perf_counter() - t0
                pred_path = args.output_dir / "preds" / f"{holdout['name']}_gate_{expert_view}_{args.view_graph_method}_seed{seed}.csv"
                pred = write_pred(pred_path, holdout["cases"], result.labels)
                prob_path = args.output_dir / "probs" / f"{holdout['name']}_gate_{expert_view}_{args.view_graph_method}_seed{seed}.npy"
                prob_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(prob_path, prob)
                ba, tpr, tnr = pairwise_scores(holdout["labels"], pred)
                row = {
                    "dataset": holdout["name"],
                    "seed": seed,
                    "source_config": f"view_gate_{args.gate_model_type}",
                    "source_prob_path": "",
                    "graph_method": args.view_graph_method,
                    "view_config": f"gate_{args.gate_base_view_config}_to_{expert_view}",
                    "view_fusion": "learned_gate",
                    "BA": ba,
                    "TPR": tpr,
                    "TNR": tnr,
                    "k": k,
                    "cases": len(holdout["labels"]),
                    "num_clusters": len(set(pred)),
                    "num_merges": result.num_merges,
                    "num_splits": result.num_splits,
                    "runtime_sec": runtime,
                    "pred_path": str(pred_path),
                    "prob_path": str(prob_path),
                    "notes": json.dumps(debug, sort_keys=True),
                }
                rows.append(row)
                gate_debug_rows.append({"dataset": holdout["name"], "seed": seed, "expert_view": expert_view, **debug, "BA": ba, "TPR": tpr, "TNR": tnr})
                mixed, fragmented, false_merges = _pair_bug_counts(holdout["labels"], pred)
                for rank, diag in enumerate(mixed + fragmented + false_merges, 1):
                    diagnostics_rows.append({
                        "dataset": holdout["name"],
                        "seed": seed,
                        "source_config": f"view_gate_{args.gate_model_type}",
                        "graph_method": args.view_graph_method,
                        "view_config": f"gate_{args.gate_base_view_config}_to_{expert_view}",
                        "rank": rank,
                        **diag,
                    })
                print(
                    f"[gate] dataset={holdout['name']} seed={seed} expert={expert_view} "
                    f"graph={args.view_graph_method} BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f} "
                    f"gate_mean={debug['gate_mean']:.3f}", flush=True,
                )
    return rows, diagnostics_rows, gate_debug_rows


def run_graph_stage(args: argparse.Namespace, datasets: Sequence[Path]) -> tuple[list[dict], list[dict], list[dict]]:
    prob_dirs = [resolve(path) for path in args.source_prob_dirs]
    result_rows: list[dict] = []
    diagnostics_rows: list[dict] = []
    trajectory_rows: list[dict] = []
    for dataset in datasets:
        cases = osf.read_cases(dataset / "input.csv")
        gold = read_gold(osf.gold_path(dataset))
        records = osf.build_case_records(dataset.name, dataset / "input.csv", gold_csv=None)
        conflict = conflict_matrix_from_records(records)
        k = len(set(gold))
        for seed in args.seeds:
            for source_config in args.source_configs:
                prob_path = _find_prob_file(
                    prob_dirs,
                    dataset.name,
                    args.source_model_arch,
                    source_config,
                    args.source_graph_tag,
                    seed,
                )
                if prob_path is None:
                    print(f"[graph] missing prob dataset={dataset.name} seed={seed} config={source_config}", flush=True)
                    continue
                prob = np.load(prob_path).astype(np.float32)
                if prob.shape[0] != len(gold):
                    print(f"[graph] skip shape mismatch {prob_path}: prob={prob.shape} gold={len(gold)}", flush=True)
                    continue
                for graph_method in args.graph_methods:
                    t0 = time.perf_counter()
                    result = gc.cluster_probability_graph(
                        prob,
                        k,
                        graph_method,
                        conflict_matrix=conflict,
                        merge_top_m=args.merge_top_m,
                        merge_threshold=args.merge_threshold,
                        merge_conflict_threshold=args.merge_conflict_threshold,
                        merge_internal_threshold=args.merge_internal_threshold,
                        merge_max_merges=args.merge_max_merges,
                        mknn_k=args.mknn_k,
                        mknn_threshold=args.mknn_threshold,
                        signed_conflict_penalty=args.signed_conflict_penalty,
                        signed_max_iter=args.signed_max_iter,
                        signed_keep_k=args.signed_keep_k,
                    )
                    runtime = time.perf_counter() - t0
                    pred_path = (
                        args.output_dir / "preds" /
                        f"{dataset.name}_{source_config}_{graph_method}_seed{seed}.csv"
                    )
                    pred = write_pred(pred_path, cases, result.labels)
                    ba, tpr, tnr = pairwise_scores(gold, pred)
                    row = {
                        "dataset": dataset.name,
                        "seed": seed,
                        "source_config": source_config,
                        "source_prob_path": str(prob_path),
                        "graph_method": graph_method,
                        "view_config": "dual",
                        "view_fusion": "concat",
                        "BA": ba,
                        "TPR": tpr,
                        "TNR": tnr,
                        "k": k,
                        "cases": len(gold),
                        "num_clusters": len(set(pred)),
                        "num_merges": result.num_merges,
                        "num_splits": result.num_splits,
                        "runtime_sec": runtime,
                        "pred_path": str(pred_path),
                        "prob_path": "",
                        "notes": "graph_recluster_existing_prob",
                    }
                    result_rows.append(row)
                    mixed, fragmented, false_merges = _pair_bug_counts(gold, pred)
                    for rank, diag in enumerate(mixed + fragmented + false_merges, 1):
                        diagnostics_rows.append({
                            "dataset": dataset.name,
                            "seed": seed,
                            "source_config": source_config,
                            "graph_method": graph_method,
                            "rank": rank,
                            **diag,
                        })
                    for step, traj in enumerate(result.trajectory):
                        trajectory_rows.append({
                            "dataset": dataset.name,
                            "seed": seed,
                            "source_config": source_config,
                            "graph_method": graph_method,
                            "step": step,
                            **traj,
                        })
                    print(
                        f"[graph] dataset={dataset.name} seed={seed} source={source_config} "
                        f"method={graph_method} BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f} "
                        f"clusters={len(set(pred))}",
                        flush=True,
                    )
    return result_rows, diagnostics_rows, trajectory_rows


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graph-aware clustering + multi-view embedding experiments")
    p.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--source-prob-dirs", nargs="+", type=Path, default=[
        Path("/tmp/eight_dataset_gated_hardpos_tmux_gpu4_s0_2"),
        Path("/tmp/eight_dataset_gated_hardpos_tmux_gpu5_s3_4"),
        Path("/tmp/eight_dataset_gated_hardpos_gpu4_s0_2"),
        Path("/tmp/eight_dataset_gated_hardpos_gpu5_s3_4"),
    ])
    p.add_argument("--source-model-arch", default="gated_mlp")
    p.add_argument("--source-configs", nargs="+", default=["balanced"])
    p.add_argument("--source-graph-tag", default="graph0")
    p.add_argument("--graph-methods", nargs="+", default=[
        "agglomerative_avg",
        "agglomerative_complete",
        "conservative_merge",
        "mutual_knn_cc",
        "signed_graph_greedy",
    ])
    p.add_argument("--embedding-view-configs", nargs="+", default=[
        "dual",
        "tri_event",
        "tri_object",
        "quad_event_object",
        "quad_event_object_context",
    ])
    p.add_argument("--merge-top-m", type=int, default=5)
    p.add_argument("--merge-threshold", type=float, default=0.75)
    p.add_argument("--merge-conflict-threshold", type=float, default=0.20)
    p.add_argument("--merge-internal-threshold", type=float, default=0.55)
    p.add_argument("--merge-max-merges", type=int, default=2)
    p.add_argument("--mknn-k", type=int, default=5)
    p.add_argument("--mknn-threshold", type=float, default=0.65)
    p.add_argument("--signed-conflict-penalty", type=float, default=1.0)
    p.add_argument("--signed-max-iter", type=int, default=20)
    p.add_argument("--signed-keep-k", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-graph-stage", action="store_true")
    p.add_argument("--doc-examples-per-dataset", type=int, default=4)
    p.add_argument("--run-view-stage", action="store_true")
    p.add_argument("--view-model-types", nargs="+", default=["gbdt"])
    p.add_argument("--view-graph-method", default="agglomerative_avg")
    p.add_argument("--view-dim", type=int, default=64)
    p.add_argument("--view-max-pairs-per-dataset", type=int, default=60000)
    p.add_argument("--view-negative-ratio", type=float, default=2.0)
    p.add_argument("--view-hard-negative-ratio", type=float, default=0.5)
    p.add_argument("--view-hard-positive-ratio", type=float, default=0.5)
    p.add_argument("--parser", default="drain")
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--llm-doc-max-features", type=int, default=80)
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--llm-batch-size", type=int, default=64)
    p.add_argument("--llm-timeout-sec", type=float, default=20.0)
    p.add_argument("--run-view-gate-stage", action="store_true")
    p.add_argument("--gate-base-view-config", default="dual")
    p.add_argument("--gate-expert-view-configs", nargs="+", default=["quad_event_object_context"])
    p.add_argument("--gate-model-type", default="gbdt", choices=["gbdt", "logistic"])
    p.add_argument("--gate-validation-fraction", type=float, default=0.20)
    p.add_argument("--gate-min-val-pairs", type=int, default=512)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [resolve(path) for path in args.datasets]
    missing = [str(ds) for ds in datasets if not (ds / "input.csv").exists()]
    if missing:
        raise FileNotFoundError(f"missing datasets: {missing}")

    doc_rows: list[dict] = []
    for dataset in datasets:
        for view_config in args.embedding_view_configs:
            if view_config == "dual":
                continue
            doc_rows.extend(build_view_docs(dataset, view_config, args.doc_examples_per_dataset))
    write_jsonl(args.output_dir / "embedding_doc_examples.jsonl", doc_rows)

    result_rows: list[dict] = []
    diagnostics_rows: list[dict] = []
    trajectory_rows: list[dict] = []
    if not args.skip_graph_stage:
        result_rows, diagnostics_rows, trajectory_rows = run_graph_stage(args, datasets)
        if result_rows:
            result_fields = [
                "dataset", "seed", "source_config", "source_prob_path",
                "graph_method", "view_config", "view_fusion",
                "BA", "TPR", "TNR", "k", "cases", "num_clusters",
                "num_merges", "num_splits", "runtime_sec", "pred_path",
            ]
            write_csv(args.output_dir / "results.csv", result_rows, result_fields)
            summary = summarize(result_rows)
            summary_fields = [
                "graph_method", "view_config", "view_fusion", "source_config",
                "mean_BA", "std_BA", "worst_BA", "mean_TPR", "mean_TNR",
                "runtime_mean", "datasets", "runs", "dataset_means",
            ]
            write_csv(args.output_dir / "summary.csv", summary, summary_fields)
            write_csv(args.output_dir / "graph_ablation_summary.csv", summary, summary_fields)
            if diagnostics_rows:
                write_csv(args.output_dir / "cluster_diagnostics.csv", diagnostics_rows, sorted({k for row in diagnostics_rows for k in row}))
            if trajectory_rows:
                write_csv(args.output_dir / "cluster_trajectory.csv", trajectory_rows, sorted({k for row in trajectory_rows for k in row}))
            view_rows = [{
                "view_config": view,
                "status": "evaluated" if view == "dual" else "doc_examples_only_requires_retrain",
                "notes": "Stage 1 holds embeddings fixed; Stage 2 will train multi-view features.",
            } for view in args.embedding_view_configs]
            write_csv(args.output_dir / "view_ablation_summary.csv", view_rows, ["view_config", "status", "notes"])
            print("\n| rank | graph | source | mean BA | worst BA | TPR | TNR | runs |")
            print("|---:|---|---|---:|---:|---:|---:|---:|")
            for rank, row in enumerate(summary[:20], 1):
                print(
                    f"| {rank} | {row['graph_method']} | {row['source_config']} | "
                    f"{row['mean_BA']:.4f} | {row['worst_BA']:.4f} | "
                    f"{row['mean_TPR']:.4f} | {row['mean_TNR']:.4f} | {row['runs']} |"
                )
    if args.run_view_stage:
        view_rows, view_diagnostics = run_view_stage(args, datasets)
        result_rows.extend(view_rows)
        diagnostics_rows.extend(view_diagnostics)
        if result_rows:
            result_fields = [
                "dataset", "seed", "source_config", "source_prob_path",
                "graph_method", "view_config", "view_fusion",
                "BA", "TPR", "TNR", "k", "cases", "num_clusters",
                "num_merges", "num_splits", "runtime_sec", "pred_path",
                "prob_path", "notes",
            ]
            write_csv(args.output_dir / "results.csv", result_rows, result_fields)
            summary = summarize(result_rows)
            summary_fields = [
                "graph_method", "view_config", "view_fusion", "source_config",
                "mean_BA", "std_BA", "worst_BA", "mean_TPR", "mean_TNR",
                "runtime_mean", "datasets", "runs", "dataset_means",
            ]
            write_csv(args.output_dir / "summary.csv", summary, summary_fields)
            graph_summary = [row for row in summary if row["view_config"] == "dual" and not str(row["source_config"]).startswith("view_")]
            view_summary = [row for row in summary if str(row["source_config"]).startswith("view_")]
            if graph_summary:
                write_csv(args.output_dir / "graph_ablation_summary.csv", graph_summary, summary_fields)
            if view_summary:
                write_csv(args.output_dir / "view_ablation_summary.csv", view_summary, summary_fields)
            if diagnostics_rows:
                write_csv(args.output_dir / "cluster_diagnostics.csv", diagnostics_rows, sorted({k for row in diagnostics_rows for k in row}))

    if args.run_view_gate_stage:
        gate_rows, gate_diagnostics, gate_debug_rows = run_view_gate_stage(args, datasets)
        result_rows.extend(gate_rows)
        diagnostics_rows.extend(gate_diagnostics)
        if result_rows:
            result_fields = [
                "dataset", "seed", "source_config", "source_prob_path",
                "graph_method", "view_config", "view_fusion",
                "BA", "TPR", "TNR", "k", "cases", "num_clusters",
                "num_merges", "num_splits", "runtime_sec", "pred_path",
                "prob_path", "notes",
            ]
            write_csv(args.output_dir / "results.csv", result_rows, result_fields)
            summary = summarize(result_rows)
            summary_fields = [
                "graph_method", "view_config", "view_fusion", "source_config",
                "mean_BA", "std_BA", "worst_BA", "mean_TPR", "mean_TNR",
                "runtime_mean", "datasets", "runs", "dataset_means",
            ]
            write_csv(args.output_dir / "summary.csv", summary, summary_fields)
            gate_summary = [row for row in summary if str(row["source_config"]).startswith("view_gate_")]
            if gate_summary:
                write_csv(args.output_dir / "gate_ablation_summary.csv", gate_summary, summary_fields)
            if diagnostics_rows:
                write_csv(args.output_dir / "cluster_diagnostics.csv", diagnostics_rows, sorted({k for row in diagnostics_rows for k in row}))
            if gate_debug_rows:
                write_csv(args.output_dir / "gate_debug.csv", gate_debug_rows, sorted({k for row in gate_debug_rows for k in row}))

    manifest = {
        "datasets": [str(ds) for ds in datasets],
        "seeds": args.seeds,
        "source_prob_dirs": [str(resolve(path)) for path in args.source_prob_dirs],
        "source_model_arch": args.source_model_arch,
        "source_configs": args.source_configs,
        "graph_methods": args.graph_methods,
        "embedding_view_configs": args.embedding_view_configs,
        "run_view_stage": args.run_view_stage,
        "view_model_types": args.view_model_types,
        "view_graph_method": args.view_graph_method,
        "view_dim": args.view_dim,
        "run_view_gate_stage": args.run_view_gate_stage,
        "gate_base_view_config": args.gate_base_view_config,
        "gate_expert_view_configs": args.gate_expert_view_configs,
        "formal_predictor_modified": False,
        "notes": "Stage 1 evaluates graph clustering on existing dual-view P_same matrices.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

