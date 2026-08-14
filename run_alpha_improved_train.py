#!/usr/bin/env python3
"""Improved alpha: alpha's pairwise ensemble + hierarchical trace + SupCon + connectivity + soft-k.

Composition (full retrain, no LODO):
  rich     = TriLog two-tower (hierarchical trace residual + SupCon contrastive loss)
  ensemble = logistic / gbdt / mlp on summary21 features (alpha's original ensemble)
  final    = alpha * rich + (1-alpha) * ensemble   (calibrated blend)
  cluster  = soft-k correlation clustering (graph_clustering.cluster_with_fallback)

Train on --train-datasets, evaluate on --eval-datasets.  Does NOT touch
alpha_test_submission; this is a new, separately-trained model.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

import graph_clustering as gc
import official_style_features as osf
import pairwise_llm_features as plf
import run_graph_multiview_experiments as gm
import theta_trace_features as ttf
import theta_trilog_model as ttm
from run_experiments import pairwise_scores, read_gold
from run_official_full_retrain_experiments import (
    resolve, train_one, write_csv, write_pred,
)

ENSEMBLE_TYPES = ("logistic", "gbdt", "mlp")
ENSEMBLE_WEIGHTS = (0.20, 0.40, 0.40)
OFFICIAL_NAMES = {"benchmark_set_1", "benchmark_set_2"}

# Failure-mode + instruction-family signature (Priority 1).  The regr.log's
# Mismatch section records the divergence instruction between the RTL and the
# reference model; the instruction family (MDU/ALU/shift/...) and the presence
# or absence of a mismatch are strong bucketing signals that a whole-log LLM
# embedding dilutes.
FAMILY_OPS = {
    "MDU":   {"mul", "mulh", "mulhsu", "mulhu", "div", "divu", "rem", "remu"},
    "Shift": {"sll", "srl", "sra", "slli", "srli", "srai"},
    "ALU":   {"add", "addi", "sub", "or", "ori", "and", "andi", "xor", "xori", "lui", "auipc", "slt", "slti", "sltu", "sltiu", "nop"},
    "CSR":   {"csrrw", "csrrs", "csrrc", "csrrwi", "csrrsi", "csrrci"},
    "Branch": {"beq", "bne", "blt", "bge", "bltu", "bgeu"},
    "Mem":   {"lb", "lh", "lw", "ld", "lbu", "lhu", "lwu", "sb", "sh", "sw", "sd"},
    "Jump":  {"jal", "jalr"},
    "System": {"ecall", "ebreak", "fence", "mret", "sret", "wfi"},
}
FAMILY_LIST = ["MDU", "Shift", "ALU", "CSR", "Branch", "Mem", "Jump", "System", "Compressed", "Other", "None"]
FAMILY_IDX = {f: i for i, f in enumerate(FAMILY_LIST)}


def _family_of(insn: str | None) -> str:
    if not insn:
        return "None"
    i = insn.lower().strip()
    for fam, ops in FAMILY_OPS.items():
        if i in ops:
            return fam
    if i.startswith("c.") or i == "c":
        return "Compressed"
    return "Other"


def parse_failure_signature(regr_text: str) -> tuple[bool, str]:
    """Return (is_mismatch, family) from a regr.log text."""
    import re
    is_mismatch = ("Mismatch" in regr_text) or bool(re.search(r"pc\[", regr_text))
    m = re.search(r"pc\[[0-9a-fx]+\]\s+([A-Za-z][\w.]*)", regr_text)
    insn = m.group(1) if m else None
    return is_mismatch, _family_of(insn)


# Common RISC-V opcodes (RV32IMC + system), for the rich semantic signature.
OPCODE_VOCAB = [
    "lui", "auipc", "jal", "jalr",
    "beq", "bne", "blt", "bge", "bltu", "bgeu",
    "lb", "lh", "lw", "lbu", "lhu", "sb", "sh", "sw",
    "addi", "slti", "sltiu", "xori", "ori", "andi", "slli", "srli", "srai",
    "add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and",
    "mul", "mulh", "mulhsu", "mulhu", "div", "divu", "rem", "remu",
    "csrrw", "csrrs", "csrrc", "csrrwi", "csrrsi", "csrrci",
    "ecall", "ebreak", "mret", "fence",
    "c.addi", "c.add", "c.sub", "c.xor", "c.or", "c.and", "c.li", "c.lui",
    "c.slli", "c.srli", "c.srai", "c.lw", "c.sw", "c.j", "c.jr", "c.beqz", "c.bnez",
]
OPCODE_IDX = {op: i for i, op in enumerate(OPCODE_VOCAB)}
REGISTER_VOCAB = ["x%d" % i for i in range(32)]
REGISTER_ABI = {"zero": "x0", "ra": "x1", "sp": "x2", "gp": "x3", "tp": "x4",
                "t0": "x5", "t1": "x6", "t2": "x7", "s0": "x8", "fp": "x8", "s1": "x9",
                "a0": "x10", "a1": "x11", "a2": "x12", "a3": "x13", "a4": "x14", "a5": "x15",
                "a6": "x16", "a7": "x17", "s2": "x18", "s3": "x19", "s4": "x20", "s5": "x21",
                "s6": "x22", "s7": "x23", "s8": "x24", "s9": "x25", "s10": "x26", "s11": "x27",
                "t3": "x28", "t4": "x29", "t5": "x30", "t6": "x31"}
REGISTER_IDX = {r: i for i, r in enumerate(REGISTER_VOCAB)}


def parse_rich_signature(regr_text: str) -> tuple[str, str, str, int]:
    """Extract (divergence_type, opcode, register, pc_bucket) from a regr.log.

    Divergence type is one of 'mismatch' | 'test_fail' | 'other'.  The opcode and
    destination register come from the FIRST mismatch line (the root divergence),
    which is the most distribution-robust bug fingerprint available.
    """
    import re
    has_mismatch = "Mismatch" in regr_text or bool(re.search(r"pc\[", regr_text))
    has_failed = "FAILED" in regr_text or "FAIL" in regr_text
    if has_mismatch:
        dtype = "mismatch"
    elif has_failed:
        dtype = "test_fail"
    else:
        dtype = "other"

    # first mismatch line: pc[ADDR] OPCODE  RD,...
    m = re.search(r"pc\[([0-9a-fA-Fx]+)\]\s+([A-Za-z][\w.]*)\s+([a-zA-Z0-9]+)", regr_text)
    opcode = "None"
    register = "None"
    pc_bucket = -1
    if m:
        opcode = m.group(2).lower()
        reg_raw = m.group(3).lower().split(",")[0].split(":")[0].strip()
        register = REGISTER_ABI.get(reg_raw, reg_raw if reg_raw.startswith("x") else "None")
        pc = int(m.group(1), 16)
        pc_bucket = (pc >> 16) & 0xFF  # 16-bit bucket of the PC
    return dtype, opcode, register, pc_bucket


def rich_signature_features(signatures: Sequence[tuple[str, str, str, int]]) -> np.ndarray:
    """Build a deterministic one-hot feature matrix for rich signatures."""
    n_op = len(OPCODE_VOCAB)
    n_reg = len(REGISTER_VOCAB)
    n_type = 3
    type_idx = {"mismatch": 0, "test_fail": 1, "other": 2}
    feats = np.zeros((len(signatures), n_type + n_op + n_reg + 1), dtype=np.float32)
    for i, (dtype, opcode, register, pc_bucket) in enumerate(signatures):
        feats[i, type_idx.get(dtype, 2)] = 1.0
        if opcode in OPCODE_IDX:
            feats[i, n_type + OPCODE_IDX[opcode]] = 1.0
        if register in REGISTER_IDX:
            feats[i, n_type + n_op + REGISTER_IDX[register]] = 1.0
        if pc_bucket >= 0:
            feats[i, n_type + n_op + n_reg] = float(pc_bucket) / 255.0
    return feats


def extract_mismatch_snippet(regr_text: str, max_pairs: int = 2) -> str:
    """Extract a short SEMANTIC snippet (the divergence) from a regr.log.

    For a mismatch case this is the first few ``ibex/spike`` divergence lines
    (the actual bug manifestation), NOT the UVM/VCS boilerplate.  For a
    test-fail case it is the ``<test> : [FAILED]`` line.
    """
    lines = [ln.strip() for ln in regr_text.splitlines() if ln.strip()]
    snippet: list[str] = []
    for ln in lines:
        if ln.startswith("ibex") or ln.startswith("spike"):
            snippet.append(ln)
            if len(snippet) >= max_pairs * 2:
                break
    if snippet:
        return " | ".join(snippet)
    for ln in lines:
        if "[FAILED]" in ln or "FAILED" in ln:
            return ln
    return lines[0] if lines else ""


def extract_mismatch_snippets(dataset: Path, max_pairs: int = 2) -> list[str]:
    """Return per-case mismatch snippet strings in input.csv order."""
    cases = osf.read_cases(dataset / "input.csv")
    out = []
    for case_id in cases:
        r = _find_regr(dataset, case_id)
        txt = r.read_text(errors="ignore") if r else ""
        out.append(extract_mismatch_snippet(txt, max_pairs=max_pairs))
    return out


def extract_test_name(regr_text: str) -> str:
    """Extract the failing test name (e.g. 'riscv_interrupt_csr_test') from regr.log.

    The test name is a strong bucketing signal: it labels the functional area the
    test exercises (csr / interrupt / debug / mmu / ...), which correlates with the
    root-cause bug.  Prefer the ``<test>.N : [FAILED]`` line; fall back to any
    ``*_test`` token.
    """
    import re
    m = re.search(r"([A-Za-z0-9_]+_test)\.\d+\s*:\s*\[?FAILED", regr_text)
    if m:
        return m.group(1)
    m = re.search(r"([A-Za-z0-9_]+_test)", regr_text)
    if m:
        return m.group(1)
    return ""


def extract_test_names(dataset: Path) -> list[str]:
    """Return per-case failing test name strings in input.csv order."""
    cases = osf.read_cases(dataset / "input.csv")
    out = []
    for case_id in cases:
        r = _find_regr(dataset, case_id)
        txt = r.read_text(errors="ignore") if r else ""
        out.append(extract_test_name(txt))
    return out


def _find_regr(dataset: Path, case_id: str) -> Path | None:
    case_id = str(case_id).strip()
    # ``read_cases`` returns the bare Case column ('1') for official-format
    # datasets but the directory is 'case_1'; normalize the bare numeric form.
    if case_id.isdigit():
        case_id = f"case_{case_id}"
    for sub in ("", "cases"):
        base = dataset / sub / case_id / "regr.log" if sub else dataset / case_id / "regr.log"
        if base.exists():
            return base
    for h in dataset.rglob("regr.log"):
        if h.parent.name == case_id or case_id in h.parts:
            return h
    return None


def extract_failure_signatures(dataset: Path) -> list[tuple[bool, str]]:
    """Return per-case (is_mismatch, family) in input.csv order."""
    cases = osf.read_cases(dataset / "input.csv")
    sigs = []
    for case_id in cases:
        r = _find_regr(dataset, case_id)
        txt = r.read_text(errors="ignore") if r else ""
        sigs.append(parse_failure_signature(txt))
    return sigs


def extract_rich_signatures(dataset: Path) -> list[tuple[str, str, str, int]]:
    """Return per-case (divergence_type, opcode, register, pc_bucket) in input.csv order."""
    cases = osf.read_cases(dataset / "input.csv")
    sigs = []
    for case_id in cases:
        r = _find_regr(dataset, case_id)
        txt = r.read_text(errors="ignore") if r else ""
        sigs.append(parse_rich_signature(txt))
    return sigs


def build_signature_pair_features(signatures: Sequence[tuple[bool, str]], pairs: list[tuple[int, int]]) -> np.ndarray:
    """Pair features: same-family bit + family one-hot for both cases."""
    nf = len(FAMILY_LIST)
    feats = np.zeros((len(pairs), 1 + 2 * nf), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        mi, fi = signatures[i]
        mj, fj = signatures[j]
        feats[k, 0] = 1.0 if fi == fj else 0.0
        feats[k, 1 + FAMILY_IDX[fi]] = 1.0
        feats[k, 1 + nf + FAMILY_IDX[fj]] = 1.0
    return feats


def clone_args(args: argparse.Namespace, **updates) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def build_all_features(args, datasets):
    """Build base (LLM case) features + hierarchical trace features for all datasets."""
    llm_args = gm.make_embedding_args(args)
    base_features: list[plf.LLMCaseFeature] = []
    trace_features: list[ttf.HierarchicalTraceFeature] = []
    labels: list[str] = []
    cases: list[str] = []
    signatures: list[tuple[bool, str]] = []
    slices: list[dict] = []
    offset = 0
    for dataset in datasets:
        ep, _ = plf.build_llm_case_features_for_inputs(
            [dataset / "input.csv"], parser=args.parser, svd_dim=args.svd_dim, llm_args=llm_args
        )
        gold = read_gold(osf.gold_path(dataset))
        cs = osf.read_cases(dataset / "input.csv")
        if not (len(ep) == len(gold) == len(cs)):
            raise RuntimeError(f"feature/label mismatch for {dataset.name}")
        tr, _ = ttf.build_hierarchical_trace_features(
            dataset / "input.csv",
            cache_dir=args.trace_cache_dir,
            segment_count=args.trace_segment_count,
            chunk_size=args.trace_chunk_size,
            anchor_sizes=args.trace_anchor_sizes,
        )
        sig = extract_failure_signatures(dataset)
        base_features.extend(ep)
        trace_features.extend(tr)
        labels.extend(gold)
        cases.extend(cs)
        signatures.extend(sig)
        slices.append({"name": dataset.name, "start": offset, "stop": offset + len(gold)})
        offset += len(gold)
        print(f"[load] dataset={dataset.name} cases={len(gold)} trace_ok={sum(f.has_trace for f in tr)}", flush=True)
    return base_features, trace_features, labels, cases, signatures, slices


def build_custom_views(args, datasets):
    """Build the event/object/context custom-view embeddings for all cases."""
    documents = {name: [] for name in ("event", "object", "context")}
    for dataset in datasets:
        docs = gm.build_all_view_documents(dataset)
        for name in documents:
            documents[name].extend(docs[name])
    return {name: gm.fetch_view_embeddings(docs, args, name) for name, docs in documents.items()}


def train_rich_trilog(args, base_features, trace_features, raw_custom, signatures, slices, labels):
    """Train the TriLog two-tower (trace + SupCon + failure-signature) on all training data."""
    n = len(base_features)
    train_indices = np.arange(n, dtype=np.int64)
    feature_reducer = plf.fit_llm_reducer(base_features, args.view_dim, random_state=args.seed)
    summary_reducer = plf.fit_llm_summary_reducer(base_features, args.view_dim, random_state=args.seed + 17)
    plf.apply_llm_reducer(base_features, feature_reducer, args.view_dim)
    plf.apply_llm_summary_reducer(base_features, summary_reducer, args.view_dim)

    custom_reducers: dict[str, object] = {}
    reduced_custom: dict[str, np.ndarray] = {}
    for pos, (name, raw) in enumerate(raw_custom.items()):
        reducer, _ = plf._fit_reducer_for_matrix(
            np.asarray(raw, dtype=np.float32)[train_indices], args.view_dim, args.seed + 101 + pos * 13
        )
        custom_reducers[name] = reducer
        reduced_custom[name] = plf._apply_reducer_to_matrix(
            np.asarray(raw, dtype=np.float32), reducer, args.view_dim
        ).astype(np.float32, copy=False)

    trace_bundle, trace_matrices = ttf.fit_transform_trace_views(
        trace_features, train_indices, seed=args.seed,
        global_struct_dim=args.trace_global_struct_dim, global_text_dim=args.trace_global_text_dim,
        anchor_struct_dim=args.trace_anchor_struct_dim, anchor_text_dim=args.trace_anchor_text_dim,
        residual_struct_dim=args.trace_residual_struct_dim, residual_text_dim=args.trace_residual_text_dim,
    )

    # Sample within-dataset pairs from all training data (full retrain, no fold holdout).
    pairs, y, weight = tpl_sample_pairs(base_features, labels, slices, args, args.seed)

    view_names = gm.views_for_config("quad_event_object_context")
    base = gm.build_multiview_pair_feature_matrix(base_features, reduced_custom, view_names, pairs)
    trace = ttf.build_trace_pair_feature_components(trace_features, trace_matrices, pairs)["residual"]
    sig_feats = build_signature_pair_features(signatures, pairs)
    matrix = np.hstack([base, trace, sig_feats]).astype(np.float32, copy=False)

    rich_args = argparse.Namespace(
        random_state=args.seed, device=args.device, epochs=args.rich_epochs,
        batch_size=args.rich_batch_size, lr=args.lr, weight_decay=args.weight_decay,
        dropout=args.dropout, early_stop_patience=args.early_stop_patience,
        focal_gamma=args.focal_gamma, fusion=args.fusion,
        supcon_weight=args.supcon_weight, supcon_temperature=args.supcon_temperature,
    )
    w = np.asarray(weight, dtype=np.float32)
    w = w / max(float(np.mean(w)), 1e-12)
    package = ttm.train_trilog_pair_model(matrix, np.asarray(y, dtype=np.float32), w, int(base.shape[1]), rich_args)

    preprocess = {
        "feature_reducer": feature_reducer,
        "summary_reducer": summary_reducer,
        "custom_reducers": custom_reducers,
        "trace_bundle": trace_bundle,
        "view_names": view_names,
    }
    return {"model_pkg": package, "preprocess": preprocess, "base_dim": int(base.shape[1])}


def tpl_sample_pairs(base_features, labels, slices, args, seed):
    """Sample within-dataset pairs (connectivity-aware), mapped to global indices.

    Returns (pairs, y, weight, pair_bug): ``pair_bug`` is the integer bug id of
    the pair's shared bug for same-bug pairs, and -1 for different-bug pairs.
    """
    import train_pairwise_llm as tpl
    all_pairs: list[tuple[int, int]] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    bugs: list[np.ndarray] = []
    n_ds = max(1, len(slices))
    per_ds_budget = max(1, args.max_train_pairs // n_ds)
    rng_seed = seed * 1009 + 17
    bug_to_id: dict[str, int] = {}

    def bid(b: str) -> int:
        if b not in bug_to_id:
            bug_to_id[b] = len(bug_to_id)
        return bug_to_id[b]

    for sl in slices:
        start, stop = sl["start"], sl["stop"]
        local_features = base_features[start:stop]
        local_labels = labels[start:stop]
        pairs, y, _stats = tpl.sample_pairs(
            local_features, local_labels,
            negative_ratio=args.rich_negative_ratio,
            hard_negative_ratio=args.rich_hard_negative_ratio,
            hard_positive_ratio=args.rich_hard_positive_ratio,
            max_train_pairs=per_ds_budget,
            random_state=rng_seed + start,
            positive_sampling="diverse",
            negative_sampling="confusable",
            connectivity_positive_fraction=args.connectivity_positive_fraction,
        )
        all_pairs.extend([(i + start, j + start) for i, j in pairs])
        ys.append(y)
        ws.append(np.ones(len(y), dtype=np.float32))
        bugs.append(np.array([bid(str(local_labels[i])) if yy > 0.5 else -1 for (i, _j), yy in zip(pairs, y)], dtype=np.int64))
    return all_pairs, np.concatenate(ys), np.concatenate(ws), np.concatenate(bugs)


def train_complete(args, train_datasets):
    base_features, trace_features, labels, cases, signatures, slices = build_all_features(args, train_datasets)
    raw_custom = build_custom_views(args, train_datasets)
    rich = train_rich_trilog(args, base_features, trace_features, raw_custom, signatures, slices, labels)

    ensemble = []
    for model_type in ENSEMBLE_TYPES:
        ens_args = clone_args(
            args, model_type=model_type, feature_mode="summary21", llm_reduce_dim=0,
            mlp_arch="shallow", loss="bce", epochs=args.ensemble_epochs,
            batch_size=args.ensemble_batch_size, negative_ratio=2.0,
            hard_negative_ratio=0.5, hard_positive_ratio=0.5,
            positive_sampling="det_low", negative_sampling="det_high",
        )
        trained = train_one(ens_args, train_datasets, f"{args.tag}_ensemble_{model_type}", args.seed)
        ensemble.append(trained)
    return {"rich": rich, "ensemble": ensemble}


def predict_rich(args, rich, dataset):
    """Predict pairwise probabilities with the TriLog rich model on one dataset."""
    llm_args = gm.make_embedding_args(args)
    features, _ = plf.build_llm_case_features_for_inputs(
        [dataset / "input.csv"], parser=args.parser, svd_dim=args.svd_dim, llm_args=llm_args
    )
    n = len(features)
    trace_features, _ = ttf.build_hierarchical_trace_features(
        dataset / "input.csv", cache_dir=args.trace_cache_dir,
        segment_count=args.trace_segment_count, chunk_size=args.trace_chunk_size,
        anchor_sizes=args.trace_anchor_sizes,
    )
    pre = rich["preprocess"]
    plf.apply_llm_reducer(features, pre["feature_reducer"], args.view_dim)
    plf.apply_llm_summary_reducer(features, pre["summary_reducer"], args.view_dim)

    docs = gm.build_all_view_documents(dataset)
    raw_custom = {name: gm.fetch_view_embeddings(docs[name], args, name) for name in ("event", "object", "context")}
    reduced_custom = {}
    for name, reducer in pre["custom_reducers"].items():
        raw = raw_custom.get(name, np.zeros((n, 0), dtype=np.float32))
        reduced_custom[name] = (
            plf._apply_reducer_to_matrix(raw, reducer, args.view_dim).astype(np.float32, copy=False)
            if reducer is not None and raw.shape[1] > 0 else np.zeros((n, 0), dtype=np.float32)
        )
    trace_matrices = ttf.apply_trace_reducers(pre["trace_bundle"], trace_features)
    pairs = osf.all_pairs(n)
    base = gm.build_multiview_pair_feature_matrix(features, reduced_custom, pre["view_names"], pairs)
    trace = ttf.build_trace_pair_feature_components(trace_features, trace_matrices, pairs)["residual"]
    sig = extract_failure_signatures(dataset)
    sig_feats = build_signature_pair_features(sig, pairs)
    matrix = np.hstack([base, trace, sig_feats]).astype(np.float32, copy=False)
    flat = ttm.predict_trilog_pair_model(rich["model_pkg"], matrix)
    prob = np.eye(n, dtype=np.float32)
    for (i, j), v in zip(pairs, flat):
        prob[i, j] = prob[j, i] = float(v)
    return prob


def evaluate(args, trained, eval_datasets):
    rich = trained["rich"]
    ensemble_pkgs = [item["model_pkg"] for item in trained["ensemble"]]
    ensemble_llm_args = plf._make_llm_args(
        llm_mode="embedding", llm_doc_style="features", llm_cache_dir=args.llm_cache_dir,
        svd_dim=args.svd_dim, llm_dual=False,
    )
    rows = []
    for ds in eval_datasets:
        gold = read_gold(osf.gold_path(ds))
        k = len(set(gold))
        cases = osf.read_cases(ds / "input.csv")

        p_rich_raw = predict_rich(args, rich, ds)
        ensemble_features, _ = plf.build_llm_case_features(ds / "input.csv", svd_dim=args.svd_dim, llm_args=ensemble_llm_args)
        p_ensemble_raw = plf.predict_probability_matrix_ensemble(
            ensemble_pkgs, list(ENSEMBLE_WEIGHTS), ensemble_features,
            ensemble_mode="prob_average", batch_size=args.predict_batch_size,
        )
        p_final = args.alpha * p_rich_raw + (1.0 - args.alpha) * p_ensemble_raw
        p_final = (p_final + p_final.T) * 0.5
        np.fill_diagonal(p_final, 1.0)

        clustered = gc.cluster_with_fallback(
            p_final.astype(np.float32), k, cannot_link_weight=args.cannot_link_weight
        )
        pred = write_pred(args.output_dir / "preds" / f"{ds.name}_seed{args.seed}.csv", cases, clustered.labels)
        ba, tpr, tnr = pairwise_scores(gold, pred)
        row = {"dataset": ds.name, "BA": ba, "TPR": tpr, "TNR": tnr,
               "k": k, "cases": len(gold), "clusters": clustered.num_clusters}
        rows.append(row)
        print(f"[eval] {ds.name} BA={ba:.4f} TPR={tpr:.4f} TNR={tnr:.4f} clusters={clustered.num_clusters}/{k}", flush=True)
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Improved alpha: TriLog(trace+SupCon) + alpha ensemble + soft-k")
    p.add_argument("--train-datasets", nargs="+", type=Path, required=True)
    p.add_argument("--eval-datasets", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tag", default="alpha_improved")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--parser", default="drain")
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--view-dim", type=int, default=64)
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--trace-cache-dir", type=Path, default=Path("/tmp/theta_trilog_trace_cache"))
    p.add_argument("--llm-doc-max-features", type=int, default=80)
    p.add_argument("--llm-batch-size", type=int, default=64)
    p.add_argument("--llm-timeout-sec", type=float, default=60.0)
    p.add_argument("--embedding-expected-dim", type=int, default=768)
    p.add_argument("--trace-segment-count", type=int, default=16)
    p.add_argument("--trace-chunk-size", type=int, default=512)
    p.add_argument("--trace-anchor-sizes", nargs="+", type=int, default=[32, 64, 128])
    p.add_argument("--trace-global-struct-dim", type=int, default=48)
    p.add_argument("--trace-global-text-dim", type=int, default=48)
    p.add_argument("--trace-anchor-struct-dim", type=int, default=48)
    p.add_argument("--trace-anchor-text-dim", type=int, default=48)
    p.add_argument("--trace-residual-struct-dim", type=int, default=48)
    p.add_argument("--trace-residual-text-dim", type=int, default=48)
    p.add_argument("--fusion", default="concat")
    p.add_argument("--supcon-weight", type=float, default=0.1)
    p.add_argument("--supcon-temperature", type=float, default=0.1)
    p.add_argument("--connectivity-positive-fraction", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=0.88)
    p.add_argument("--cannot-link-weight", type=float, default=100.0)
    p.add_argument("--predict-batch-size", type=int, default=100000)
    p.add_argument("--rich-epochs", type=int, default=40)
    p.add_argument("--rich-batch-size", type=int, default=4096)
    p.add_argument("--ensemble-epochs", type=int, default=40)
    p.add_argument("--ensemble-batch-size", type=int, default=4096)
    p.add_argument("--rich-negative-ratio", type=float, default=4.0)
    p.add_argument("--rich-hard-negative-ratio", type=float, default=0.8)
    p.add_argument("--rich-hard-positive-ratio", type=float, default=0.5)
    p.add_argument("--max-train-pairs", type=int, default=300000)
    p.add_argument("--max-pairs-per-dataset", type=int, default=120000)
    p.add_argument("--official-pair-weight", type=float, default=1.0)
    p.add_argument("--fake-pair-weight", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", default="auto")
    p.add_argument("--hidden-dims", nargs="+", type=int, default=None)
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--no-llm", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_datasets = [resolve(x) for x in args.train_datasets]
    eval_datasets = [resolve(x) for x in args.eval_datasets]
    t0 = time.perf_counter()
    trained = train_complete(args, train_datasets)
    rows = evaluate(args, trained, eval_datasets)
    fields = ["dataset", "BA", "TPR", "TNR", "k", "cases", "clusters"]
    write_csv(args.output_dir / "results.csv", rows, fields)
    manifest = {
        "seed": args.seed, "alpha": args.alpha, "supcon_weight": args.supcon_weight,
        "connectivity_positive_fraction": args.connectivity_positive_fraction,
        "fusion": args.fusion, "train_datasets": [str(x) for x in train_datasets],
        "ensemble_weights": list(ENSEMBLE_WEIGHTS),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] total={time.perf_counter() - t0:.1f}s results -> {args.output_dir / 'results.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
