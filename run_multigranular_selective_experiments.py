#!/usr/bin/env python3
"""LODO experiments for multi-granular evidence and selective experts.

Experimental only. Gold/golden files are used by this runner for supervised
training and evaluation. The formal regr_fail_bucketing.py entry point is not
modified and continues to ignore gold, meta, and trace by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import completion_case_features as ccf
import multigranular_features as mgf
import pairwise_llm_features as plf
import run_unified_multidataset_experiments as ume
import selective_expert as se
import signed_graph_clustering as sgc
import trace_anchor as ta
from run_experiments import pairwise_scores
from run_official_full_retrain_experiments import write_csv, write_pred

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = ume.DEFAULT_DATASETS


def all_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def predict_flat(model_pkg: dict, X: np.ndarray, batch_size: int) -> np.ndarray:
    import torch
    if len(X) == 0:
        return np.zeros(0, dtype=np.float32)
    scaled = model_pkg["scaler"].transform(X).astype(np.float32)
    values: list[np.ndarray] = []
    model_pkg["model"].eval()
    with torch.no_grad():
        for start in range(0, len(scaled), batch_size):
            xb = torch.from_numpy(scaled[start:start + batch_size]).to(model_pkg["device"])
            values.append(torch.sigmoid(model_pkg["model"](xb)).cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def probability_from_flat(n: int, pairs: Sequence[tuple[int, int]], values: Sequence[float]) -> np.ndarray:
    prob = np.eye(n, dtype=np.float32)
    for (i, j), value in zip(pairs, values):
        prob[i, j] = prob[j, i] = float(value)
    return prob


def build_matrix(
    features: list[plf.LLMCaseFeature],
    evidence: list[mgf.CaseEvidence],
    pairs: Sequence[tuple[int, int]],
    fine: bool,
    event_order: bool,
    local_embeddings: bool,
) -> np.ndarray:
    base = plf.build_rich_pair_feature_matrix(
        features, pairs, feature_mode="llm_dual_struct_det_summary"
    )
    if not fine:
        return base
    granular = mgf.build_multigranular_pair_feature_matrix(
        evidence, pairs,
        include_event_order=event_order,
        include_local_embeddings=local_embeddings,
    )
    return np.concatenate([base, granular], axis=1).astype(np.float32)


def dataset_probabilities(
    model_pkg: dict,
    features: list[plf.LLMCaseFeature],
    evidence: list[mgf.CaseEvidence],
    slices: Sequence[ume.DatasetSlice],
    fine: bool,
    args: argparse.Namespace,
    aux_builder=None,
) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for ds in slices:
        local_pairs = all_pairs(ds.stop - ds.start)
        global_pairs = [(i + ds.start, j + ds.start) for i, j in local_pairs]
        X = build_matrix(
            features, evidence, global_pairs, fine,
            not args.no_event_order, not args.no_local_embeddings,
        )
        if aux_builder is not None:
            aux = aux_builder(global_pairs)
            X = np.concatenate([X, aux], axis=1).astype(np.float32)
        values = predict_flat(model_pkg, X, args.predict_batch_size)
        output[ds.name] = {
            "pairs": local_pairs,
            "global_pairs": global_pairs,
            "values": values,
            "prob": probability_from_flat(len(ds.labels), local_pairs, values),
        }
    return output


def difficulty_for_slices(
    probabilities: dict[str, dict],
    slices: Sequence[ume.DatasetSlice],
    seed: int,
    budget: float,
) -> tuple[dict[str, se.DifficultyResult], set[int]]:
    result: dict[str, se.DifficultyResult] = {}
    selected_global: set[int] = set()
    for ds in slices:
        prob = probabilities[ds.name]["prob"]
        labels = plf.cluster_from_probability(prob, len(set(ds.labels)))
        item = se.compute_case_difficulty(
            prob, labels, len(set(ds.labels)),
            view_probabilities=[prob, ume.graph_refine_probability(prob, 0.10)],
            budget=budget, random_state=seed * 1009 + ds.start,
        )
        result[ds.name] = item
        selected_global.update(ds.start + int(i) for i in np.flatnonzero(item.selected))
    return result, selected_global


def expert_aux_builder(
    mode: str,
    trace_features,
    completion_features,
    selected_global: set[int],
):
    trace_dim = ta.anchor_trace_pair_feature_dim() if "trace" in mode else 0
    completion_dim = ccf.completion_pair_feature_dim() if "completion" in mode else 0

    def build(pairs: Sequence[tuple[int, int]]) -> np.ndarray:
        blocks: list[np.ndarray] = []
        if trace_dim:
            blocks.append(ta.build_anchor_trace_pair_feature_matrix(trace_features, pairs))
        if completion_dim:
            blocks.append(ccf.build_completion_pair_feature_matrix(completion_features, pairs))
        aux = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(pairs), 0), dtype=np.float32)
        available = np.asarray(
            [i in selected_global or j in selected_global for i, j in pairs], dtype=bool
        )
        aux[~available] = 0.0
        return aux.astype(np.float32)

    return build, trace_dim + completion_dim


def local_gate_rows(
    ds: ume.DatasetSlice,
    info_base: dict,
    info_expert: dict,
    difficulty: se.DifficultyResult,
    evidence: Sequence[mgf.CaseEvidence],
    selected_global: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_evidence = evidence[ds.start:ds.stop]
    conflict = mgf.build_conflict_matrix(local_evidence)
    selected = np.asarray([ds.start + i in selected_global for i in range(len(ds.labels))])
    X = se.build_gate_feature_matrix(
        info_base["pairs"], info_base["prob"], info_expert["prob"],
        difficulty, conflict, selected,
    )
    availability = np.asarray(
        [selected[i] or selected[j] for i, j in info_base["pairs"]], dtype=np.float32
    )
    y = np.asarray(
        [float(ds.labels[i] == ds.labels[j]) for i, j in info_base["pairs"]],
        dtype=np.float32,
    )
    return X, availability, y


def train_selective_expert(
    mode: str,
    args: argparse.Namespace,
    seed: int,
    features,
    evidence,
    slices,
    pair_data,
    fine_pkg,
    fine_train_probs,
    difficulty,
    selected_global,
    input_csvs,
):
    trace_features = None
    completion_features = None
    trace_debug: list[dict] = []
    completion_debug: list[dict] = []
    if "trace" in mode:
        trace_features, trace_debug = ta.build_anchor_trace_case_features(
            input_csvs, window_size=args.trace_window, selected_indices=selected_global
        )
    if "completion" in mode:
        completion_features, completion_debug = ccf.build_completion_case_features(
            input_csvs, cache_dir=args.completion_cache_dir,
            strict=False, selected_indices=selected_global,
        )
    aux_builder, aux_dim = expert_aux_builder(
        mode, trace_features, completion_features, selected_global
    )
    base_X = build_matrix(
        features, evidence, pair_data.pairs, True,
        not args.no_event_order, not args.no_local_embeddings,
    )
    aux_X = aux_builder(pair_data.pairs)
    if args.modality_dropout > 0.0 and aux_X.shape[1]:
        rng = np.random.default_rng(seed + 31013)
        available = se.selective_pair_mask(
            pair_data.pairs, [i in selected_global for i in range(len(features))]
        )
        dropped = available & (rng.random(len(aux_X)) < args.modality_dropout)
        aux_X = aux_X.copy()
        aux_X[dropped] = 0.0
    X = np.concatenate([base_X, aux_X], axis=1).astype(np.float32)
    teacher = predict_flat(fine_pkg, base_X, args.predict_batch_size)
    easy = ~se.selective_pair_mask(pair_data.pairs, [i in selected_global for i in range(len(features))])
    expert_pkg = ume.train_unified_model(
        X, pair_data, args, "rank_trans", seed + 10000,
        teacher_probs=teacher, easy_mask=easy,
    )
    expert_probs = dataset_probabilities(
        expert_pkg, features, evidence, slices, True, args, aux_builder=aux_builder
    )

    val_global = set(int(x) for x in expert_pkg["val_idx"])
    val_pair_keys = {tuple(pair_data.pairs[idx]): idx for idx in val_global}
    gate_X: list[np.ndarray] = []
    gate_y: list[np.ndarray] = []
    gate_pb: list[np.ndarray] = []
    gate_pe: list[np.ndarray] = []
    for ds in slices:
        Xg, availability, y = local_gate_rows(
            ds, fine_train_probs[ds.name], expert_probs[ds.name], difficulty[ds.name],
            evidence, selected_global,
        )
        keep = np.asarray([
            (i + ds.start, j + ds.start) in val_pair_keys
            for i, j in fine_train_probs[ds.name]["pairs"]
        ], dtype=bool)
        keep &= availability > 0
        if np.any(keep):
            gate_X.append(Xg[keep]); gate_y.append(y[keep])
            gate_pb.append(fine_train_probs[ds.name]["values"][keep])
            gate_pe.append(expert_probs[ds.name]["values"][keep])
    if gate_X:
        gate_pkg = se.train_gate(
            np.concatenate(gate_X), np.concatenate(gate_y),
            np.concatenate(gate_pb), np.concatenate(gate_pe), seed,
        )
    else:
        gate_pkg = {"model_type": "constant", "value": 0.0}
    return expert_pkg, gate_pkg, aux_builder, trace_debug, completion_debug, aux_dim


def evaluate_probability(
    args, ds, method, prob, evidence, seed, clusterer, k_policy, runtime, notes,
):
    conflict = mgf.build_conflict_matrix(evidence[ds.start:ds.stop])
    clustered = sgc.cluster_probability(
        prob, len(set(ds.labels)), clusterer, k_policy,
        conflict_matrix=conflict, random_state=seed,
    )
    ba, tpr, tnr = pairwise_scores(ds.labels, clustered.labels)
    stem = f"{ds.name}_{method}_seed{seed}"
    pred_path = args.output_dir / "preds" / f"{stem}.csv"
    prob_path = args.output_dir / "probs" / f"{stem}.npy"
    write_pred(pred_path, ds.cases, clustered.labels)
    prob_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(prob_path, prob)
    trajectory_rows = []
    for row in clustered.trajectory:
        trajectory_rows.append({"dataset": ds.name, "method": method, "seed": seed, **row})
    return {
        "seed": seed, "dataset": ds.name, "method": method,
        "clusterer": clusterer, "k_policy": k_policy,
        "reference_k": len(set(ds.labels)), "selected_k": clustered.selected_k,
        "cases": len(ds.labels), "BA": ba, "TPR": tpr, "TNR": tnr,
        "runtime_sec": runtime, "pred_path": str(pred_path),
        "prob_path": str(prob_path), "notes": notes,
    }, trajectory_rows


def run_fold(args, datasets: list[Path], holdout_index: int, seed: int):
    start_time = time.perf_counter()
    holdout = datasets[holdout_index]
    ordered = [ds for idx, ds in enumerate(datasets) if idx != holdout_index] + [holdout]
    slices = ume.build_slices(ordered)
    train_slices, target = slices[:-1], slices[-1]
    input_csvs = [ds / "input.csv" for ds in ordered]

    llm_args = plf._make_llm_args(
        llm_mode="embedding", llm_doc_style="features",
        llm_cache_dir=args.llm_cache_dir, svd_dim=args.svd_dim, llm_dual=True,
    )
    features, _ = plf.build_llm_case_features_for_inputs(
        input_csvs, svd_dim=args.svd_dim, llm_args=llm_args,
    )
    train_stop = target.start
    llm_reducer = plf.fit_llm_reducer(features[:train_stop], args.llm_reduce_dim, seed)
    summary_reducer = plf.fit_llm_summary_reducer(features[:train_stop], args.llm_reduce_dim, seed)
    plf.apply_llm_reducer(features, llm_reducer, args.llm_reduce_dim)
    plf.apply_llm_summary_reducer(features, summary_reducer, args.llm_reduce_dim)

    evidence, evidence_debug = mgf.build_case_evidence(input_csvs, context_radius=2)
    if not args.no_local_embeddings:
        try:
            model_name = mgf.embed_local_documents(
                evidence, args.local_embedding_cache_dir,
                batch_size=args.llm_batch_size, timeout_sec=args.llm_timeout_sec,
            )
            print(f"[local-embedding] model={model_name} dim={len(evidence[0].sim_local_vec)}", flush=True)
        except Exception as exc:
            print(f"[warning] local embeddings unavailable: {type(exc).__name__}: {exc}", flush=True)
            args.no_local_embeddings = True
    if not args.no_local_embeddings:
        reducers = mgf.fit_local_reducers(evidence[:train_stop], args.local_reduce_dim, seed)
        mgf.apply_local_reducers(evidence, reducers)

    pair_data = ume.build_pair_data(features[:train_stop], train_slices, args, seed)
    rows: list[dict] = []
    trajectories: list[dict] = []
    difficulty_rows: list[dict] = []
    completion_rows: list[dict] = []
    trace_rows: list[dict] = []

    current_pkg = fine_pkg = None
    if "current_base" in args.variants:
        X_current = build_matrix(features[:train_stop], evidence[:train_stop], pair_data.pairs, False, True, False)
        current_pkg = ume.train_unified_model(X_current, pair_data, args, "rank_trans", seed)
        target_pairs = all_pairs(len(target.labels))
        global_target_pairs = [(i + target.start, j + target.start) for i, j in target_pairs]
        X_target = build_matrix(features, evidence, global_target_pairs, False, True, False)
        prob = probability_from_flat(len(target.labels), target_pairs, predict_flat(current_pkg, X_target, args.predict_batch_size))
        row, traj = evaluate_probability(args, target, "current_base", prob, evidence, seed, "average", "fixed", time.perf_counter()-start_time, "global dual embedding base")
        rows.append(row); trajectories += traj

    need_fine = any(v != "current_base" for v in args.variants)
    if not need_fine:
        return rows, trajectories, difficulty_rows, completion_rows, trace_rows, evidence_debug
    X_fine = build_matrix(
        features[:train_stop], evidence[:train_stop], pair_data.pairs, True,
        not args.no_event_order, not args.no_local_embeddings,
    )
    fine_pkg = ume.train_unified_model(X_fine, pair_data, args, "rank_trans", seed + 100)
    fine_train_probs = dataset_probabilities(
        fine_pkg, features[:train_stop], evidence[:train_stop], train_slices, True, args
    )
    target_pairs = all_pairs(len(target.labels))
    target_global_pairs = [(i + target.start, j + target.start) for i, j in target_pairs]
    X_target_fine = build_matrix(
        features, evidence, target_global_pairs, True,
        not args.no_event_order, not args.no_local_embeddings,
    )
    target_fine_prob = probability_from_flat(
        len(target.labels), target_pairs,
        predict_flat(fine_pkg, X_target_fine, args.predict_batch_size),
    )
    all_probs = dict(fine_train_probs)
    all_probs[target.name] = {"pairs": target_pairs, "global_pairs": target_global_pairs,
                              "values": target_fine_prob[np.triu_indices(len(target.labels), 1)],
                              "prob": target_fine_prob}
    difficulty, selected_global = difficulty_for_slices(all_probs, slices, seed, args.completion_budget)
    for ds in slices:
        d = difficulty[ds.name]
        for i in range(len(ds.labels)):
            difficulty_rows.append({
                "seed": seed, "holdout": target.name, "dataset": ds.name,
                "case_id": ds.cases[i], "difficulty": d.scores[i],
                "entropy": d.entropy[i], "margin": d.margin[i],
                "instability": d.instability[i], "selected": bool(d.selected[i]),
            })

    if "fine_base" in args.variants:
        row, traj = evaluate_probability(args, target, "fine_base", target_fine_prob, evidence, seed, args.clusterer, args.k_policy, time.perf_counter()-start_time, "multi-granular sim/regr")
        rows.append(row); trajectories += traj
    if "fine_signed" in args.variants:
        row, traj = evaluate_probability(args, target, "fine_signed", target_fine_prob, evidence, seed, "signed_graph", "fixed", time.perf_counter()-start_time, "multi-granular plus signed graph fixed k")
        rows.append(row); trajectories += traj

    requested_modes = []
    if "selective_trace" in args.variants: requested_modes.append("trace")
    if "selective_completion" in args.variants: requested_modes.append("completion")
    if any(v in args.variants for v in ("full", "full_adaptive")): requested_modes.append("trace_completion")
    if args.expert_mode != "auto":
        requested_modes = [args.expert_mode] if args.expert_mode != "none" else []
    for mode in dict.fromkeys(requested_modes):
        expert_pkg, gate_pkg, aux_builder, tdebug, cdebug, aux_dim = train_selective_expert(
            mode, args, seed, features[:train_stop], evidence[:train_stop], train_slices,
            pair_data, fine_pkg, fine_train_probs,
            {ds.name: difficulty[ds.name] for ds in train_slices},
            {i for i in selected_global if i < train_stop}, input_csvs[:-1],
        )
        for item in tdebug:
            item.update({"seed": seed, "holdout": target.name, "expert_mode": mode, "split": "train"})
        trace_rows += tdebug
        for item in cdebug:
            item.update({"seed": seed, "holdout": target.name, "expert_mode": mode, "split": "train"})
        completion_rows += cdebug
        # Build target-only selective evidence with local indices, avoiding calls for easy cases.
        selected_target_local = {i for i in range(len(target.labels)) if target.start + i in selected_global}
        target_trace = target_completion = None
        if "trace" in mode:
            target_trace, td = ta.build_anchor_trace_case_features(
                [input_csvs[-1]], args.trace_window, selected_indices=selected_target_local
            )
            for item in td: item.update({"seed": seed, "holdout": target.name, "expert_mode": mode, "split": "holdout"})
            trace_rows += td
        if "completion" in mode:
            target_completion, cd = ccf.build_completion_case_features(
                [input_csvs[-1]], args.completion_cache_dir, strict=False,
                selected_indices=selected_target_local,
            )
            for item in cd: item.update({"seed": seed, "holdout": target.name, "expert_mode": mode, "split": "holdout"})
            completion_rows += cd
        target_aux, _ = expert_aux_builder(mode, target_trace, target_completion, selected_target_local)
        X_expert_target = np.concatenate([X_target_fine, target_aux(target_pairs)], axis=1)
        target_expert_prob = probability_from_flat(
            len(target.labels), target_pairs,
            predict_flat(expert_pkg, X_expert_target, args.predict_batch_size),
        )
        selected_mask = difficulty[target.name].selected
        conflict = mgf.build_conflict_matrix(evidence[target.start:target.stop])
        gate_X = se.build_gate_feature_matrix(
            target_pairs, target_fine_prob, target_expert_prob,
            difficulty[target.name], conflict, selected_mask,
        )
        availability = np.asarray([selected_mask[i] or selected_mask[j] for i,j in target_pairs], dtype=np.float32)
        gate_values = se.predict_gate(gate_pkg, gate_X, availability)
        if args.gate_mode == "fixed":
            gate_values = availability * float(args.fixed_gate)
        final_prob = se.fuse_probability_matrix(target_fine_prob, target_expert_prob, target_pairs, gate_values)
        method = f"selective_{mode}"
        row, traj = evaluate_probability(args, target, method, final_prob, evidence, seed, "average", "fixed", time.perf_counter()-start_time, f"selected_cases={int(selected_mask.sum())};aux_dim={aux_dim};gate={gate_pkg['model_type']}")
        rows.append(row); trajectories += traj
        if mode == "trace_completion" and "full_adaptive" in args.variants:
            row, traj = evaluate_probability(args, target, "full_adaptive", final_prob, evidence, seed, "signed_graph", "adaptive", time.perf_counter()-start_time, f"selected_cases={int(selected_mask.sum())};gate={gate_pkg['model_type']}")
            rows.append(row); trajectories += traj
    return rows, trajectories, difficulty_rows, completion_rows, trace_rows, evidence_debug


def summarize(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows: grouped[str(row["method"])].append(row)
    out = []
    for method, values in grouped.items():
        by_dataset: dict[str, list[float]] = defaultdict(list)
        for row in values: by_dataset[str(row["dataset"])].append(float(row["BA"]))
        ds_mean = {k: float(np.mean(v)) for k,v in by_dataset.items()}
        official = [v for k,v in ds_mean.items() if k in {"benchmark_set_1","benchmark_set_2"}]
        fake = [v for k,v in ds_mean.items() if k not in {"benchmark_set_1","benchmark_set_2"}]
        out.append({
            "method": method, "mean_BA": float(np.mean(list(ds_mean.values()))),
            "official_mean_BA": float(np.mean(official)) if official else float("nan"),
            "fake_mean_BA": float(np.mean(fake)) if fake else float("nan"),
            "min_dataset_BA": min(ds_mean.values()),
            "mean_TPR": float(np.mean([float(x["TPR"]) for x in values])),
            "mean_TNR": float(np.mean([float(x["TNR"]) for x in values])),
            "mean_selected_k": float(np.mean([float(x["selected_k"]) for x in values])),
            "runs": len(values),
        })
    return sorted(out, key=lambda x:(x["official_mean_BA"],x["mean_BA"]), reverse=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experimental multi-granular selective expert LODO")
    p.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    p.add_argument("--holdouts", nargs="+", default=["all"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--variants", nargs="+", default=["current_base","fine_base","fine_signed","selective_trace","selective_completion","full","full_adaptive"], choices=("current_base","fine_base","fine_signed","selective_trace","selective_completion","full","full_adaptive"))
    p.add_argument("--clusterer", choices=("average","spectral","signed_graph"), default="average")
    p.add_argument("--k-policy", choices=("fixed","adaptive"), default="fixed")
    p.add_argument("--expert-mode", choices=("auto","none","trace","completion","trace_completion"), default="auto")
    p.add_argument("--completion-budget", type=float, default=0.15)
    p.add_argument("--modality-dropout", type=float, default=0.30)
    p.add_argument("--gate-mode", choices=("learned","fixed"), default="learned")
    p.add_argument("--fixed-gate", type=float, default=0.50)
    p.add_argument("--gate-weight", type=float, default=0.20, help="Recorded gate BCE coefficient; the gate is optimized separately.")
    p.add_argument("--trace-window", type=int, default=64)
    p.add_argument("--completion-cache-dir", type=Path, default=Path("/tmp/regr_fail_completion_cache_selective"))
    p.add_argument("--no-event-order", action="store_true")
    p.add_argument("--no-local-embeddings", action="store_true")
    p.add_argument("--local-reduce-dim", type=int, default=32)
    p.add_argument("--local-embedding-cache-dir", type=Path, default=Path("/tmp/regr_fail_local_embedding_cache"))
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--llm-batch-size", type=int, default=32)
    p.add_argument("--llm-timeout-sec", type=float, default=120.0)
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--llm-reduce-dim", type=int, default=64)
    p.add_argument("--device", choices=("auto","cpu","cuda"), default="auto")
    p.add_argument("--negative-ratio", type=float, default=2.0)
    p.add_argument("--hard-negative-ratio", type=float, default=0.6)
    p.add_argument("--hard-positive-ratio", type=float, default=0.25)
    p.add_argument("--max-pairs-per-dataset", type=int, default=30000)
    p.add_argument("--max-aux-per-dataset", type=int, default=10000)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--representation-dim", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--aux-batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=2e-4)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--ranking-weight", type=float, default=0.20)
    p.add_argument("--ranking-margin", type=float, default=0.5)
    p.add_argument("--transitivity-weight", type=float, default=0.05)
    p.add_argument("--distillation-weight", type=float, default=0.30)
    p.add_argument("--domain-weight", type=float, default=0.0)
    p.add_argument("--domain-grl-scale", type=float, default=0.0)
    p.add_argument("--validation-fraction", type=float, default=0.10)
    p.add_argument("--early-stop-patience", type=int, default=6)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--predict-batch-size", type=int, default=100000)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [ume.resolve(x) for x in args.datasets]
    if args.holdouts == ["all"]:
        holdout_indices = list(range(len(datasets)))
    else:
        lookup = {x.name:i for i,x in enumerate(datasets)}
        holdout_indices = [lookup[x] for x in args.holdouts]
    rows=[]; trajectories=[]; difficulties=[]; completions=[]; traces=[]; evidence=[]
    for seed in args.seeds:
        for holdout_index in holdout_indices:
            print(f"[LODO] seed={seed} holdout={datasets[holdout_index].name}", flush=True)
            values = run_fold(args, datasets, holdout_index, seed)
            for target, source in zip((rows,trajectories,difficulties,completions,traces,evidence), values):
                target.extend(source)
            if rows: write_csv(args.output_dir/"results.partial.csv", rows, list(rows[0]))
    if rows: write_csv(args.output_dir/"results.csv", rows, list(rows[0]))
    summary = summarize(rows)
    if summary: write_csv(args.output_dir/"summary.csv", summary, list(summary[0]))
    if trajectories: write_csv(args.output_dir/"cluster_trajectory.csv", trajectories, list(trajectories[0]))
    if difficulties: write_csv(args.output_dir/"difficulty_debug.csv", difficulties, list(difficulties[0]))
    if completions: write_csv(args.output_dir/"completion_debug.csv", completions, sorted({k for x in completions for k in x}))
    if traces: write_csv(args.output_dir/"trace_debug.csv", traces, sorted({k for x in traces for k in x}))
    if evidence: write_csv(args.output_dir/"evidence_debug.csv", evidence, list(evidence[0]))
    # Ablation-compatible flat output and case-level error analysis.
    if rows: write_csv(args.output_dir/"ablation.csv", rows, list(rows[0]))
    report = ["# Multi-granular selective expert error analysis", ""]
    dataset_by_name = {x.name: x for x in datasets}
    for item in summary:
        report.append(f"- {item['method']}: official mean BA={item['official_mean_BA']:.4f}, fake mean BA={item['fake_mean_BA']:.4f}, TPR={item['mean_TPR']:.4f}, TNR={item['mean_TNR']:.4f}")
    for row in rows:
        ds = ume.build_slices([dataset_by_name[str(row["dataset"]) ]])[0]
        with Path(str(row["pred_path"])).open(newline="", encoding="utf-8-sig") as f:
            pred_rows = list(csv.DictReader(f))
        pred = [str(x.get("bucket", "")) for x in pred_rows]
        fp=[]; fn=[]
        for i,j in all_pairs(len(ds.labels)):
            same_gold = ds.labels[i] == ds.labels[j]
            same_pred = pred[i] == pred[j]
            if same_pred and not same_gold: fp.append((ds.cases[i],ds.cases[j],ds.labels[i],ds.labels[j]))
            if same_gold and not same_pred: fn.append((ds.cases[i],ds.cases[j],ds.labels[i]))
        fragments = defaultdict(set)
        purity = defaultdict(list)
        for gold_label, bucket in zip(ds.labels,pred):
            fragments[gold_label].add(bucket); purity[bucket].append(gold_label)
        report.extend(["", f"## {row['dataset']} / {row['method']} / seed {row['seed']}", f"BA={float(row['BA']):.4f}, FP pairs={len(fp)}, FN pairs={len(fn)}", "Top fragmented bugs: " + ", ".join(f"{k}:{len(v)}" for k,v in sorted(fragments.items(), key=lambda x:len(x[1]), reverse=True)[:5]), "Top FP pairs: " + "; ".join(f"{a}-{b} ({ga}/{gb})" for a,b,ga,gb in fp[:5]), "Top FN pairs: " + "; ".join(f"{a}-{b} ({g})" for a,b,g in fn[:5])])
    (args.output_dir/"error_analysis.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    manifest={"datasets":[str(x) for x in datasets],"seeds":args.seeds,"variants":args.variants,"formal_predictor_modified":False,"completion_budget":args.completion_budget,"trace_window":args.trace_window,"modality_dropout":args.modality_dropout,"gate_mode":args.gate_mode,"loss_weights":{"classification":1.0,"ranking":args.ranking_weight,"transitivity":args.transitivity_weight,"distillation":args.distillation_weight,"gate":args.gate_weight}}
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("\n| method | official BA | fake BA | overall BA | TPR | TNR |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in summary:
        print(f"| {row['method']} | {row['official_mean_BA']:.4f} | {row['fake_mean_BA']:.4f} | {row['mean_BA']:.4f} | {row['mean_TPR']:.4f} | {row['mean_TNR']:.4f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
