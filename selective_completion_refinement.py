#!/usr/bin/env python3
"""Selective Completion Post-hoc Refinement.

Post-processes an existing pairwise probability matrix with a small number of
completion LLM calls on the hardest cases only. Completion evidence provides
conservative boost/veto signals that adjust uncertain same-bug pair edges,
followed by reclustering.

Does NOT: train models, call completion on all cases, output buckets directly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


# ── Config loading ────────────────────────────────────────────────────────

def load_completion_config() -> dict | None:
    """Read completion config from LLM_MODEL_CONFIG (same as completion_case_features)."""
    raw = os.getenv("LLM_MODEL_CONFIG", "").strip()
    if not raw:
        return None
    try:
        import yaml
        data = yaml.safe_load(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    section = data.get("completion") or data.get("chat") or data.get("generation")
    if not isinstance(section, dict):
        return None
    cfg = section.get("config") if isinstance(section.get("config"), dict) else section
    model = section.get("model_name") or cfg.get("model")
    base_url = cfg.get("base_url")
    api_key = cfg.get("api_key")
    api_key_env = cfg.get("api_key_env")
    if not api_key and api_key_env:
        api_key = os.getenv(str(api_key_env), "")
    if not model or not base_url:
        return None
    return {
        "model": str(model),
        "base_url": str(base_url),
        "api_key": str(api_key or "dummy"),
        "timeout": float(cfg.get("timeout", 60.0)),
        "max_tokens": int(cfg.get("max_tokens", 320)),
    }


# ── Hard case selection ───────────────────────────────────────────────────

def select_hard_cases(
    prob_matrix: np.ndarray,
    labels: np.ndarray,
    infos: Sequence[dict],
    selection_mode: str = "hybrid",
    max_cases: int = 5,
    n: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select the hardest cases for completion.

    Returns (selected_indices, difficulty_scores, entropy_scores, margin_scores).
    """
    if n is None:
        n = prob_matrix.shape[0]
    max_cases = min(max_cases, n)

    # ── Entropy score: entropy of P values in uncertain zone ──
    entropy_scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        probs = prob_matrix[i]
        uncertain = probs[(probs >= 0.3) & (probs <= 0.7)]
        uncertain = uncertain[~np.isclose(uncertain, 1.0)]  # exclude self (diag=1)
        if len(uncertain) > 0:
            # Binary entropy: -p*log(p) - (1-p)*log(1-p), max at p=0.5
            entropies = -uncertain * np.log2(np.clip(uncertain, 1e-10, 1.0)) \
                        - (1 - uncertain) * np.log2(np.clip(1 - uncertain, 1e-10, 1.0))
            entropy_scores[i] = float(np.mean(entropies))

    # ── Margin score: gap between best and second-best cluster affinity ──
    margin_scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        cluster_affinities = defaultdict(list)
        for j in range(n):
            if i != j:
                cluster_affinities[str(labels[j])].append(float(prob_matrix[i, j]))
        if len(cluster_affinities) >= 2:
            means = sorted([np.mean(v) for v in cluster_affinities.values()], reverse=True)
            if len(means) >= 2:
                margin_scores[i] = 1.0 - (means[0] - means[1])  # smaller margin → higher score
            else:
                margin_scores[i] = 1.0
        else:
            margin_scores[i] = 1.0  # only one cluster → uncertain

    # ── Conflict score: high P but structured disagreement ──
    conflict_scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        conflicts = 0
        high_p_count = 0
        for j in range(n):
            if i == j or prob_matrix[i, j] < 0.5:
                continue
            high_p_count += 1
            ii = infos[i] if i < len(infos) else {}
            ij = infos[j] if j < len(infos) else {}
            for key in ("primary_type", "mismatch_type", "op_pair"):
                av = str(ii.get(key, "") or "").strip().lower()
                bv = str(ij.get(key, "") or "").strip().lower()
                if av and bv and av != bv:
                    conflicts += 1
                    break
        conflict_scores[i] = conflicts / max(1, high_p_count) if high_p_count > 0 else 0.0

    # ── Combined difficulty score ──
    if selection_mode == "entropy_top":
        difficulty = entropy_scores
    elif selection_mode == "margin_top":
        difficulty = margin_scores
    elif selection_mode == "conflict_top":
        difficulty = conflict_scores
    else:  # hybrid
        e_rank = _rank(entropy_scores)
        m_rank = _rank(margin_scores)
        c_rank = _rank(conflict_scores)
        difficulty = 0.45 * e_rank + 0.35 * m_rank + 0.20 * c_rank

    selected = np.argsort(-difficulty)[:max_cases]
    return selected, difficulty[selected], entropy_scores[selected], margin_scores[selected]


def _rank(scores: np.ndarray) -> np.ndarray:
    """Convert scores to 0-1 ranks (higher score → higher rank)."""
    order = np.argsort(scores)
    ranks = np.zeros_like(scores, dtype=np.float64)
    for i, idx in enumerate(order):
        ranks[idx] = i / max(1, len(scores) - 1)
    return ranks


# ── Completion client ─────────────────────────────────────────────────────

def _completion_cache_key(prompt: str, config: dict, schema_version: int = 2) -> str:
    identity = f"{config['model']}|{config['base_url']}|v{schema_version}"
    return hashlib.sha256((identity + "\n" + prompt).encode()).hexdigest()[:16]


def _read_cache(cache_dir: Path, cache_key: str) -> dict | None:
    path = cache_dir / f"{cache_key}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") == "ok":
            return data
        # Error cache: check TTL (1h)
        created = data.get("created_at", 0)
        if created and (time.time() - created) > 3600:
            path.unlink(missing_ok=True)
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(cache_dir: Path, cache_key: str, data: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{cache_key}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@dataclass
class CompletionUsageStats:
    requests: int = 0
    cache_hits: int = 0
    failures: int = 0
    total_latency: float = 0.0
    unknown_count: int = 0
    parse_failures: int = 0

    @property
    def cache_hit_ratio(self) -> float:
        return self.cache_hits / max(1, self.requests)

    @property
    def avg_latency(self) -> float:
        return self.total_latency / max(1, self.requests - self.cache_hits)


def run_selective_completions(
    case_indices: np.ndarray,
    infos: Sequence[dict],
    case_ids: Sequence[str],
    cache_dir: Path,
    config: dict | None = None,
    stats: CompletionUsageStats | None = None,
) -> dict[int, dict]:
    """Run completion LLM calls for selected case indices only.

    Returns dict: case_index → parsed_completion_data (fallback empty dict on failure).
    """
    if config is None:
        config = load_completion_config()
    if config is None:
        print("[completion] no valid completion config; skipping all calls", file=sys.stderr)
        return {}
    if stats is None:
        stats = CompletionUsageStats()

    from openai import OpenAI
    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        timeout=config["timeout"],
    )

    results: dict[int, dict] = {}
    for idx in case_indices:
        stats.requests += 1
        info = infos[idx] if idx < len(infos) else {}
        case_id = case_ids[idx] if idx < len(case_ids) else f"case_{idx}"

        prompt = _build_short_prompt(info)
        cache_key = _completion_cache_key(prompt, config)
        cached = _read_cache(cache_dir, cache_key)

        if cached is not None:
            stats.cache_hits += 1
            results[idx] = cached
            continue

        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "system", "content": "Return strict JSON only. No markdown. Do not infer bucket id or same/different bug."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0, max_tokens=config["max_tokens"],
            )
            text = response.choices[0].message.content or ""
            latency = time.perf_counter() - t0
            stats.total_latency += latency

            parsed = _parse_minimal_json(text)
            parsed["raw_response"] = text[:200]
            parsed["status"] = "ok" if parsed.get("failure_stage") else "parse_partial"
            parsed["created_at"] = time.time()
            _write_cache(cache_dir, cache_key, parsed)
            results[idx] = parsed
        except Exception as exc:
            stats.failures += 1
            error_data = {
                "status": "request_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:200],
                "created_at": time.time(),
            }
            _write_cache(cache_dir, cache_key, error_data)
            results[idx] = error_data

    return results


def _build_short_prompt(info: dict) -> str:
    """Build a compact prompt from a single case's structured info."""
    parts = []
    for key in ("primary_signature", "primary_type", "mismatch_type", "op_pair",
                "failed_reason", "fatal_file"):
        val = info.get(key, "")
        if val:
            parts.append(f"{key}: {val}")
    return (
        "Analyze this RISC-V CPU regression failure mechanism. "
        "Return JSON: {\"failure_stage\":\"...\",\"mechanism\":\"...\","
        "\"trigger\":\"...\",\"symptom\":\"...\",\"confidence\":\"low|medium|high\","
        "\"objects\":{\"opcode\":[],\"register\":[],\"csr\":[],\"pc_region\":[]},"
        "\"evidence_tags\":[],\"conflict_tags\":[],\"behavior_tags\":[]}\n\n"
        + "\n".join(parts[:10])
    )


def _parse_minimal_json(text: str) -> dict:
    import re
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"failure_stage": "", "status": "json_not_found"}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"failure_stage": "", "status": "json_parse_failed"}


# ── Pair Refinement ───────────────────────────────────────────────────────

def refine_probability_matrix(
    prob_base: np.ndarray,
    selected_indices: np.ndarray,
    completion_results: dict[int, dict],
    infos: Sequence[dict],
    labels: Sequence[int] | None = None,
    boost: float = 0.05,
    veto: float = 0.10,
    min_p: float = 0.20,
    max_p: float = 0.80,
    adjust_scope: str = "selected_uncertain_neighbors",
    neighbor_top_m: int = 10,
    max_adj_per_case: int = 20,
    max_adj_total: int = 100,
) -> tuple[np.ndarray, list[dict]]:
    """Apply conservative boost/veto to uncertain pairs.

    adjust_scope:
      - selected_selected: only pairs where both cases are selected
      - selected_uncertain_neighbors: each selected case i adjusts its top-M
        uncertain neighbors j (in same/nearby clusters, P in [min_p, max_p])

    Returns (adjusted probability matrix, list of adjustment records).
    """
    n = prob_base.shape[0]
    prob_adj = prob_base.copy()
    selected_set = set(int(i) for i in selected_indices)
    adjustments: list[dict] = []
    adj_count_per_case: dict[int, int] = defaultdict(int)

    # Build candidate pairs based on scope
    candidate_pairs: list[tuple[int, int, float]] = []  # (i, j, priority_score)

    if adjust_scope == "selected_selected":
        for i in selected_set:
            for j in selected_set:
                if i < j and min_p <= float(prob_base[i, j]) <= max_p:
                    candidate_pairs.append((i, j, 0.0))

    elif adjust_scope == "selected_uncertain_neighbors":
        # Map each unselected case to its cluster
        if labels is None:
            labels = list(range(n))
        case_clusters: dict[int, int] = {}
        for idx in range(n):
            case_clusters[idx] = int(labels[idx]) if idx < len(labels) else -1

        # Find fragment sizes (clusters with few cases = fragments)
        cluster_sizes: dict[int, int] = defaultdict(int)
        for idx in range(n):
            cluster_sizes[case_clusters.get(idx, -1)] += 1
        small_fragment_threshold = max(3, n // 10)
        small_clusters = {c for c, sz in cluster_sizes.items() if sz <= small_fragment_threshold}

        for i in selected_set:
            ci = completion_results.get(i, {})
            if not ci:
                continue
            # Find uncertain neighbors for i
            neighbors = []
            i_cluster = case_clusters.get(i, -1)
            for j in range(n):
                if j in selected_set or j == i:
                    continue
                p = float(prob_base[i, j])
                if p < min_p or p > max_p:
                    continue
                j_cluster = case_clusters.get(j, -1)
                # Priority: same cluster > nearby > small fragment > other
                priority = 0.0
                if j_cluster == i_cluster and i_cluster >= 0:
                    priority = 3.0
                elif j_cluster in small_clusters:
                    priority = 2.0
                else:
                    priority = 1.0
                # Prefer more uncertain (closer to 0.5) within same priority
                uncertainty = 1.0 - abs(p - 0.5) * 2.0  # 0.5→1.0, 0.2/0.8→0.4
                score = priority + uncertainty
                neighbors.append((j, p, score))

            # Sort by score descending, take top_m
            neighbors.sort(key=lambda x: -x[2])
            for j, p, score in neighbors[:neighbor_top_m]:
                candidate_pairs.append((i, j, score))

    # Apply boost/veto to candidate pairs
    for i, j, _score in candidate_pairs:
        if len(adjustments) >= max_adj_total:
            break
        if adj_count_per_case.get(i, 0) >= max_adj_per_case:
            continue

        p = float(prob_base[i, j])
        ci = completion_results.get(i, {})
        cj = completion_results.get(j, {})

        # For selected-unselected: only selected side has completion
        if adjust_scope != "selected_selected":
            if not ci:
                continue
            # Use structured info for unselected side
            j_info = infos[j] if j < len(infos) else {}
            cj = {
                "mechanism": str(j_info.get("mismatch_type", "") or ""),
                "trigger": str(j_info.get("failed_reason", "") or ""),
                "failure_stage": "",
                "evidence_tags": [],
                "conflict_tags": [],
                "objects": {"opcode": [str(j_info.get("op_pair", "") or "")]},
            }

        same_mech = _same(str(ci.get("mechanism", "")), str(cj.get("mechanism", "")))
        same_trig = _same(str(ci.get("trigger", "")), str(cj.get("trigger", "")))
        mech_conflict = _conflict(str(ci.get("mechanism", "")), str(cj.get("mechanism", "")))
        trig_conflict = _conflict(str(ci.get("trigger", "")), str(cj.get("trigger", "")))

        # Tag jaccard
        tags_i = set(_listify(ci.get("evidence_tags", [])) + _listify(ci.get("conflict_tags", [])))
        tags_j = set(_listify(cj.get("evidence_tags", [])) + _listify(cj.get("conflict_tags", [])))
        tag_jac = len(tags_i & tags_j) / max(1, len(tags_i | tags_j)) if tags_i or tags_j else 0.0

        rule = "none"
        new_p = p

        if (same_mech and same_trig and tag_jac >= 0.3
                and not mech_conflict and not trig_conflict):
            new_p = min(p + boost, 0.99)
            rule = "boost"

        if mech_conflict or trig_conflict:
            new_p = max(p - veto, 0.01)
            rule = "veto"

        if rule != "none":
            prob_adj[i, j] = prob_adj[j, i] = new_p
            adj_count_per_case[i] = adj_count_per_case.get(i, 0) + 1
            adjustments.append({
                "case_i": i, "case_j": j,
                "P_base": p, "P_adj": new_p,
                "delta": new_p - p, "rule": rule,
                "same_mechanism": int(same_mech),
                "same_trigger": int(same_trig),
                "tag_jaccard": round(tag_jac, 4),
            })

    return prob_adj, adjustments


def _same(a: str, b: str) -> bool:
    return bool(a and b and str(a).strip().lower() == str(b).strip().lower())


def _conflict(a: str, b: str) -> bool:
    return bool(a and b and str(a).strip().lower() != str(b).strip().lower())


def _listify(val) -> list:
    if isinstance(val, (list, tuple)):
        return [str(v) for v in val if str(v).strip()]
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return []


# ── Reclustering ──────────────────────────────────────────────────────────

def recluster(prob: np.ndarray, k: int) -> list[int]:
    from sklearn.cluster import AgglomerativeClustering
    n = prob.shape[0]
    k = max(1, min(k, n))
    if k == n:
        return list(range(n))
    distance = 1.0 - prob.astype(np.float64)
    np.fill_diagonal(distance, 0.0)
    try:
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
    return model.fit_predict(distance).tolist()


# ── Top-level refinement entry point ──────────────────────────────────────

@dataclass
class RefinementResult:
    prob_base: np.ndarray
    prob_refined: np.ndarray
    labels_base: list[int]
    labels_refined: list[int]
    selected_indices: np.ndarray
    difficulty_scores: np.ndarray
    adjustments: list[dict]
    completion_results: dict[int, dict]
    completion_stats: CompletionUsageStats
    runtime_sec: float


def refine(
    prob_base: np.ndarray,
    k: int,
    infos: Sequence[dict],
    case_ids: Sequence[str],
    cache_dir: Path,
    selection_mode: str = "hybrid",
    max_cases: int = 5,
    boost: float = 0.05,
    veto: float = 0.10,
    adjust_scope: str = "selected_uncertain_neighbors",
    neighbor_top_m: int = 15,
    config: dict | None = None,
) -> RefinementResult:
    """Run the full selective completion refinement pipeline.

    1. Cluster base probability → initial labels
    2. Select hard cases
    3. Call completion LLM for selected cases
    4. Adjust uncertain pair probabilities
    5. Recluster with adjusted matrix

    Returns RefinementResult with all diagnostics.
    """
    t0 = time.perf_counter()
    n = prob_base.shape[0]

    # Base clustering
    labels_base = recluster(prob_base, k)

    # Select hard cases
    selected, difficulty, entropy_s, margin_s = select_hard_cases(
        prob_base, np.array(labels_base), infos,
        selection_mode=selection_mode, max_cases=max_cases,
    )

    # Completion calls
    cache_root = Path(cache_dir)
    stats = CompletionUsageStats()
    completion_results = run_selective_completions(
        selected, infos, case_ids, cache_root, config=config, stats=stats,
    )

    # Adjust pairs
    prob_refined, adjustments = refine_probability_matrix(
        prob_base, selected, completion_results, infos,
        labels=labels_base,
        boost=boost, veto=veto,
        adjust_scope=adjust_scope,
        neighbor_top_m=neighbor_top_m,
    )

    # Recluster
    labels_refined = recluster(prob_refined, k)

    runtime = time.perf_counter() - t0
    stats.log = (
        f"[completion] requests={stats.requests} cache_hits={stats.cache_hits} "
        f"hit_ratio={stats.cache_hit_ratio:.2f} avg_latency={stats.avg_latency:.1f}s "
        f"failures={stats.failures} parse_failures={stats.parse_failures} "
        f"unknown={stats.unknown_count}"
    )

    return RefinementResult(
        prob_base=prob_base,
        prob_refined=prob_refined,
        labels_base=labels_base,
        labels_refined=labels_refined,
        selected_indices=selected,
        difficulty_scores=difficulty,
        adjustments=adjustments,
        completion_results=completion_results,
        completion_stats=stats,
        runtime_sec=runtime,
    )
