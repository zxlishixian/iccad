#!/usr/bin/env python3
"""Regression-failure bucketing baseline with switchable parsers/backends.

Pipeline:
  input.csv -> sim.log/regr.log -> SimpleDrain or fixed-depth Drain templates
  -> case-level template/token/count features -> TF-IDF/hash vector
  -> k-means or agglomerative clustering -> output.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

SKLEARN_AVAILABLE = True
try:
    from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.preprocessing import Normalizer, normalize
except ImportError:
    SKLEARN_AVAILABLE = False
    AgglomerativeClustering = MiniBatchKMeans = TruncatedSVD = None
    FeatureHasher = TfidfTransformer = Normalizer = normalize = None


HASH_DIM = 1 << 15
MAX_LOG_BYTES = 256 * 1024
MAX_SELECTED_LINES = 420
RANDOM_SEED = 0
WILDCARD = "<*>"

SIGNAL_RE = re.compile(
    r"uvm_(?:fatal|error|warning)|cosim|mismatch|failed|passed|timeout|"
    r"test failed|test passed|error seen|log-extract|rtl_sim|retired|"
    r"register write|pc mismatch|illegal|exception|trap|interrupt|"
    r"signature|scoreboard",
    re.IGNORECASE,
)

HEX_RE = re.compile(r"\b(?:0x)?[0-9a-fA-F]{6,}\b")
DEC_RE = re.compile(r"(?<![A-Za-z_])\b\d{2,}\b")
PATH_RE = re.compile(r"(?:/[^\s:]+)+")
CASE_RE = re.compile(r"case[_-]?\d+", re.IGNORECASE)
SEED_RE = re.compile(r"seed[_-]?\d+|\bsvseed\s+\d+\b", re.IGNORECASE)
TIME_RE = re.compile(r"@\s*\d+")
LINE_NUM_RE = re.compile(r"^\s*(?:\[E\]\s*)?\d+:\s*")
TOKEN_RE = re.compile(r"[a-z_][a-z0-9_./:+-]*|<[^>]+>|\[[a-z]\]")
PRIMARY_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.:-]+")


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def info(message: str) -> None:
    print(message, file=sys.stderr)


def read_csv_rows(input_csv: Path) -> Tuple[List[dict], List[str]]:
    with input_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def pick_column(fieldnames: Sequence[str], wanted: str) -> str | None:
    normalized = [(name, norm_col(name)) for name in fieldnames]
    aliases = {
        "sim": ("simlog", "rtllog", "simulationlog", "sim"),
        "regr": ("regrlog", "regressionlog", "reportlog", "regr"),
        "trace": ("tracelog", "rtltrace", "cpulog", "trace"),
    }[wanted]
    for name, key in normalized:
        if key in aliases:
            return name
    for name, key in normalized:
        if wanted in key and "log" in key:
            return name
    for name, key in normalized:
        if wanted in key:
            return name
    return None


def resolve_log_path(input_csv: Path, value: str | None) -> Path | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (input_csv.parent / path).resolve()


def read_log_sample(path: Path | None) -> Tuple[str, str]:
    if path is None:
        return "", "missing_path"
    try:
        size = path.stat().st_size
    except OSError:
        return "", "missing_file"
    try:
        with path.open("rb") as f:
            if size <= MAX_LOG_BYTES:
                data = f.read()
            else:
                half = MAX_LOG_BYTES // 2
                head = f.read(half)
                f.seek(max(0, size - half))
                tail = f.read(half)
                data = head + b"\n... <LOG_TRUNCATED_HEAD_TAIL> ...\n" + tail
    except OSError:
        return "", "read_error"
    return data.decode("utf-8", "ignore"), "ok"


def select_lines(text: str) -> List[str]:
    if not text:
        return []
    lines = text.splitlines()
    chosen: List[str] = []
    seen = set()

    def add(line: str) -> None:
        if len(chosen) >= MAX_SELECTED_LINES:
            return
        compact = line.strip()
        if not compact:
            return
        key = compact[:500]
        if key not in seen:
            chosen.append(compact)
            seen.add(key)

    for line in lines:
        if SIGNAL_RE.search(line):
            add(line)
    for line in lines[:40]:
        add(line)
    for line in lines[-80:]:
        add(line)
    return chosen


def path_to_token(match: re.Match[str]) -> str:
    raw = match.group(0)
    base = raw.rstrip("/").rsplit("/", 1)[-1].lower()
    if re.search(r"\.(svh?|v|log|tcl|bin)$", base):
        return base
    return "<path>"


def normalize_log_line(line: str, preserve_basenames: bool) -> str:
    """Normalize dynamic tokens while preserving useful EDA error vocabulary."""
    s = line.strip().lower()
    s = LINE_NUM_RE.sub("", s)
    s = PATH_RE.sub(path_to_token if preserve_basenames else "<path>", s)
    s = CASE_RE.sub("<case>", s)
    s = SEED_RE.sub("<seed>", s)
    s = TIME_RE.sub("@ <time>", s)
    s = HEX_RE.sub("<hex>", s)
    s = DEC_RE.sub("<num>", s)
    s = re.sub(r"\b\d+\b", "<n>", s)
    s = re.sub(r"['\"]", "", s)
    s = re.sub(r"[^a-z0-9_+./:<>\[\]-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def line_tokens(template: str) -> Iterable[str]:
    for token in TOKEN_RE.findall(template):
        if len(token) > 1:
            yield token


def sanitize_primary_part(text: str, max_len: int = 120) -> str:
    text = PATH_RE.sub(path_to_token, text)
    text = HEX_RE.sub(" ", text)
    text = DEC_RE.sub(" ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.upper()
    text = PRIMARY_TOKEN_RE.sub("_", text).strip("_")
    text = re.sub(r"_+", "_", text)
    return text[:max_len].strip("_") or "UNKNOWN"


def sanitize_primary_token(*parts: str, max_len: int = 160) -> str:
    token = "_".join(part.strip("_") for part in parts if part and part.strip("_"))
    token = PRIMARY_TOKEN_RE.sub("_", token).strip("_")
    token = re.sub(r"_+", "_", token)
    return token[:max_len].strip("_") or "PRIMARY_UNKNOWN_FAILURE"


def extract_sv_basename(line: str) -> str:
    matches = re.findall(r"([^/\s:()]+\.svh?|[^/\s:()]+\.v)(?:\(\d+\))?", line, flags=re.IGNORECASE)
    if matches:
        return matches[-1]
    path_matches = re.findall(r"(/[^\s:()]+(?:\.svh?|\.v))(?:\(\d+\))?", line, flags=re.IGNORECASE)
    if path_matches:
        return path_matches[-1].rsplit("/", 1)[-1]
    return "unknown_source"


def message_after_uvm_context(line: str) -> str:
    # UVM messages usually place the human-readable reason after the final [component] tag.
    if "]" in line:
        return line.rsplit("]", 1)[-1].strip()
    parts = line.split(":", 1)
    return parts[-1].strip() if parts else line.strip()


def extract_opcode_pair(lines: Sequence[str]) -> str | None:
    ibex_op = None
    spike_op = None
    op_re = re.compile(r"\b(ibex|dut|rtl|spike|iss)\[\d+\]\s*:\s*pc\[[^\]]+\]\s+([a-z0-9.]+)", re.IGNORECASE)
    for line in lines:
        match = op_re.search(line)
        if not match:
            continue
        side = match.group(1).lower()
        op = match.group(2).lower()
        if side in {"ibex", "dut", "rtl"} and ibex_op is None:
            ibex_op = op
        elif side in {"spike", "iss"} and spike_op is None:
            spike_op = op
    if ibex_op and spike_op:
        return sanitize_primary_token("PRIMARY_REGR_OPPAIR", ibex_op, spike_op)
    return None


def extract_primary_signature(
    sim_info: dict,
    regr_info: dict,
    sim_lines: List[str],
    regr_lines: List[str],
) -> List[str]:
    # Primary signature tokens are deterministic sim/regr failure summaries and
    # are part of the default baseline. They do not require gold labels.
    del sim_info, regr_info
    tokens: List[str] = []

    for severity in ("UVM_FATAL", "UVM_ERROR"):
        for line in sim_lines:
            if severity not in line.upper():
                continue
            source = extract_sv_basename(line)
            message = sanitize_primary_part(message_after_uvm_context(line))
            tokens.append(sanitize_primary_token("PRIMARY", severity, source, message))
            return tokens

    joined_regr = "\n".join(regr_lines)
    lower_regr = joined_regr.lower()
    if "mismatch" in lower_regr:
        if "pc mismatch" in lower_regr:
            tokens.append("PRIMARY_REGR_PC_MISMATCH")
        if "register write data mismatch" in lower_regr:
            tokens.append("PRIMARY_REGR_REGISTER_WRITE_DATA_MISMATCH")
        if "memory mismatch" in lower_regr or "store" in lower_regr and "mismatch" in lower_regr:
            tokens.append("PRIMARY_REGR_MEMORY_MISMATCH")
        if "instruction mismatch" in lower_regr or "retired" in lower_regr:
            tokens.append("PRIMARY_REGR_INSTRUCTION_MISMATCH")
        if "cosim mismatch" in lower_regr or "co-sim" in lower_regr:
            tokens.append("PRIMARY_REGR_COSIM_MISMATCH")
        if not tokens:
            tokens.append("PRIMARY_REGR_MISMATCH_GENERIC")
        op_pair = extract_opcode_pair(regr_lines)
        if op_pair:
            tokens.append(op_pair)
        return tokens

    for line in regr_lines + sim_lines:
        upper = line.upper()
        if "[FAILED]" in upper or "TEST FAILED" in upper or "FAILED" in upper:
            reason = sanitize_primary_part(line)
            return [sanitize_primary_token("PRIMARY_FAILED", reason)]

    return ["PRIMARY_UNKNOWN_FAILURE"]


@dataclass
class LogCluster:
    cluster_id: int
    template_tokens: List[str]
    size: int = 0


@dataclass
class DrainNode:
    children: Dict[str, "DrainNode"] = field(default_factory=dict)
    clusters: List[LogCluster] = field(default_factory=list)


class SimpleDrainParser:
    """Existing simple Drain-like parser: exact normalized line templates."""

    def __init__(self) -> None:
        self.template_to_id: Dict[str, int] = {}
        self.clusters: Dict[int, LogCluster] = {}

    def add_line(self, line: str) -> int:
        tokens = self.tokenize(line)
        template = " ".join(tokens)
        if template not in self.template_to_id:
            cluster_id = len(self.template_to_id)
            self.template_to_id[template] = cluster_id
            self.clusters[cluster_id] = LogCluster(cluster_id, tokens, 0)
        cluster = self.clusters[self.template_to_id[template]]
        cluster.size += 1
        return cluster.cluster_id

    def tokenize(self, line: str) -> List[str]:
        return list(line_tokens(line))

    def get_template(self, cluster_id: int) -> str:
        cluster = self.clusters.get(cluster_id)
        return " ".join(cluster.template_tokens) if cluster else ""

    @property
    def num_templates(self) -> int:
        return len(self.clusters)


class FixedDepthDrainParser:
    def __init__(
        self,
        depth: int = 4,
        sim_threshold: float = 0.45,
        max_children: int = 100,
    ) -> None:
        self.depth = max(3, depth)
        self.sim_threshold = sim_threshold
        self.max_children = max_children
        self.root = DrainNode()
        self.clusters: Dict[int, LogCluster] = {}
        self.clusters_by_length: Dict[int, List[LogCluster]] = defaultdict(list)

    def add_line(self, line: str) -> int:
        """Add one normalized log line and return template/cluster id."""
        tokens = self.tokenize(line)
        if not tokens:
            tokens = ["<empty>"]
        candidates = self.tree_search(tokens)
        if not candidates:
            candidates = self.clusters_by_length.get(len(tokens), [])
        cluster = self.fast_match(candidates, tokens)
        if cluster is not None:
            cluster.template_tokens = self.merge_template(cluster.template_tokens, tokens)
            cluster.size += 1
            return cluster.cluster_id

        cluster_id = len(self.clusters)
        cluster = LogCluster(cluster_id=cluster_id, template_tokens=list(tokens), size=1)
        self.clusters[cluster_id] = cluster
        self.clusters_by_length[len(tokens)].append(cluster)
        self.add_cluster_to_tree(cluster)
        return cluster_id

    def tokenize(self, line: str) -> List[str]:
        return list(line_tokens(line))

    def tree_search(self, tokens: List[str]) -> List[LogCluster]:
        """Return candidate clusters from parse tree leaf."""
        node = self.root.children.get(str(len(tokens)))
        if not node:
            return []
        for depth_idx in range(min(self.depth - 2, len(tokens))):
            token = tokens[depth_idx]
            if token in node.children:
                node = node.children[token]
            elif WILDCARD in node.children:
                node = node.children[WILDCARD]
            else:
                return []
        return node.clusters

    def add_cluster_to_tree(self, cluster: LogCluster) -> None:
        tokens = cluster.template_tokens
        node = self.root.children.setdefault(str(len(tokens)), DrainNode())
        for depth_idx in range(min(self.depth - 2, len(tokens))):
            token = tokens[depth_idx]
            if token == WILDCARD or (len(node.children) >= self.max_children and token not in node.children):
                key = WILDCARD
            else:
                key = token
            node = node.children.setdefault(key, DrainNode())
        node.clusters.append(cluster)

    def fast_match(self, clusters: List[LogCluster], tokens: List[str]) -> LogCluster | None:
        best_cluster = None
        best_similarity = -1.0
        best_wildcards = 10**9
        best_size = -1
        for cluster in clusters:
            similarity, wildcard_count = self.sequence_distance(cluster.template_tokens, tokens)
            if similarity < self.sim_threshold:
                continue
            key = (similarity, -wildcard_count, cluster.size)
            best_key = (best_similarity, -best_wildcards, best_size)
            if key > best_key:
                best_cluster = cluster
                best_similarity = similarity
                best_wildcards = wildcard_count
                best_size = cluster.size
        return best_cluster

    def sequence_distance(self, template_tokens: List[str], tokens: List[str]) -> Tuple[float, int]:
        """
        Return:
          similarity: matched_token_count / len(tokens)
          wildcard_count: number of <*> in template
        """
        if len(template_tokens) != len(tokens) or not tokens:
            return 0.0, template_tokens.count(WILDCARD)
        matched = 0
        wildcard_count = 0
        for tmpl, token in zip(template_tokens, tokens):
            if tmpl == WILDCARD:
                matched += 1
                wildcard_count += 1
            elif tmpl == token:
                matched += 1
        return matched / len(tokens), wildcard_count

    def merge_template(self, old_template: List[str], tokens: List[str]) -> List[str]:
        return [old if old == token else WILDCARD for old, token in zip(old_template, tokens)]

    def get_template(self, cluster_id: int) -> str:
        cluster = self.clusters.get(cluster_id)
        return " ".join(cluster.template_tokens) if cluster else ""

    @property
    def num_templates(self) -> int:
        return len(self.clusters)


def make_parser(args: argparse.Namespace) -> SimpleDrainParser | FixedDepthDrainParser:
    if args.parser == "drain":
        return FixedDepthDrainParser(
            depth=args.drain_depth,
            sim_threshold=args.drain_st,
            max_children=args.drain_max_children,
        )
    return SimpleDrainParser()


def extract_status_features(prefix: str, text: str, feats: Counter) -> None:
    lower = text.lower()
    for key in ("uvm_fatal", "uvm_error", "uvm_warning", "mismatch", "failed", "passed", "timeout"):
        count = lower.count(key)
        if count:
            feats[f"{prefix}:count:{key}"] += min(count, 20)

    patterns = [
        ("reg_write_mismatch", r"register write data mismatch to\s+x\d+"),
        ("pc_mismatch", r"\bpc mismatch\b"),
        ("sync_trap_mismatch", r"synchronous trap"),
        ("test_pass_verdict", r"risc-v uvm test passed"),
        ("test_fail_verdict", r"risc-v uvm test failed"),
        ("cosim_matched", r"co-simulation matched"),
        ("error_seen_rtl", r"error seen in ['\"]?rtl_sim\.log"),
        ("no_failing_tests", r"no failing tests"),
    ]
    for name, pat in patterns:
        if re.search(pat, lower):
            feats[f"{prefix}:flag:{name}"] += 5


def case_id_from_row(row: dict, idx: int, columns: Sequence[str]) -> str:
    for col in columns:
        val = str(row.get(col, "") or "")
        m = CASE_RE.search(val)
        if m:
            return m.group(0).lower().replace("-", "_")
    for value in row.values():
        m = CASE_RE.search(str(value or ""))
        if m:
            return m.group(0).lower().replace("-", "_")
    return f"case_{idx + 1:06d}"


def collect_case_inputs(
    input_csv: Path,
    rows: Sequence[dict],
    sim_col: str | None,
    regr_col: str | None,
    parser_kind: str,
) -> Tuple[List[Counter], List[List[Tuple[str, str]]]]:
    base_features: List[Counter] = []
    normalized_lines: List[List[Tuple[str, str]]] = []
    preserve_basenames = parser_kind == "drain"

    for idx, row in enumerate(rows):
        feats: Counter = Counter()
        lines_for_case: List[Tuple[str, str]] = []
        selected_by_prefix: Dict[str, List[str]] = {"sim": [], "regr": []}
        info_by_prefix: Dict[str, dict] = {"sim": {}, "regr": {}}
        used_cols = [c for c in (sim_col, regr_col) if c]
        cid = case_id_from_row(row, idx, used_cols)
        feats[f"case_shape:{re.sub(r'\\d', '0', cid)}"] += 1

        for prefix, col in (("sim", sim_col), ("regr", regr_col)):
            path = resolve_log_path(input_csv, row.get(col) if col else None)
            text, status = read_log_sample(path)
            selected = select_lines(text)
            selected_by_prefix[prefix] = selected
            info_by_prefix[prefix] = {"path": str(path) if path else "", "status": status}
            feats[f"{prefix}:file_status:{status}"] += 3
            if path is not None:
                feats[f"{prefix}:basename:{path.name.lower()}"] += 1
            extract_status_features(prefix, text, feats)

            for line in selected:
                normalized_line = normalize_log_line(line, preserve_basenames=preserve_basenames)
                if normalized_line:
                    lines_for_case.append((prefix, normalized_line))

        for primary in extract_primary_signature(
            info_by_prefix["sim"],
            info_by_prefix["regr"],
            selected_by_prefix["sim"],
            selected_by_prefix["regr"],
        ):
            feats[primary] += 1

        base_features.append(feats)
        normalized_lines.append(lines_for_case)
    return base_features, normalized_lines


def build_feature_counters(
    args: argparse.Namespace,
    base_features: Sequence[Counter],
    normalized_lines: Sequence[List[Tuple[str, str]]],
    token_weights: Dict[str, float] | None = None,
    token_weight_mode: str = "none",
) -> Tuple[List[Counter], int]:
    parser = make_parser(args)
    case_template_ids: List[List[Tuple[str, int]]] = []

    for lines_for_case in normalized_lines:
        ids_for_case = []
        for prefix, normalized_line in lines_for_case:
            template_id = parser.add_line(normalized_line)
            ids_for_case.append((prefix, template_id))
        case_template_ids.append(ids_for_case)

    feature_counters: List[Counter] = []
    for feats, ids_for_case in zip(base_features, case_template_ids):
        out = Counter(feats)
        out[f"parser:{args.parser}"] += 1
        for prefix, template_id in ids_for_case:
            template = parser.get_template(template_id)
            if not template:
                continue
            out[f"{prefix}:tmpl:{template}"] += 2
            toks = list(line_tokens(template))
            for tok in toks:
                out[f"{prefix}:tok:{tok}"] += 1
            for a, b in zip(toks, toks[1:]):
                out[f"{prefix}:bi:{a}_{b}"] += 1
        feature_counters.append(apply_token_weights(out, token_weights or {}, token_weight_mode))
    return feature_counters, parser.num_templates


def load_token_weights(path: str | Path | None) -> Dict[str, float]:
    if not path:
        return {}
    try:
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        warn(f"could not read token weights {path}: {exc}; continuing without learned weights")
        return {}
    weights = data.get("weights", data if isinstance(data, dict) else {})
    out: Dict[str, float] = {}
    if isinstance(weights, dict):
        for token, weight in weights.items():
            try:
                out[str(token)] = float(weight)
            except (TypeError, ValueError):
                continue
    return out


def learned_repeat_count(weight: float) -> int:
    if weight >= 4.0:
        return 4
    if weight >= 3.0:
        return 3
    if weight >= 2.0:
        return 2
    if weight >= 0.5:
        return 1
    return 0


def token_repeat_count(token: str, weights: Dict[str, float], mode: str) -> int:
    is_primary = token.startswith("PRIMARY_")
    if mode == "none" or not weights:
        return 4 if is_primary else 1
    if token not in weights:
        return 4 if is_primary else 1
    repeat = learned_repeat_count(weights[token])
    if is_primary:
        return max(4, repeat)
    return repeat


def apply_token_weights(tokens: Counter, weights: Dict[str, float], mode: str) -> Counter:
    if mode not in {"repeat", "none"}:
        raise ValueError(f"unknown token weight mode: {mode}")
    weighted: Counter = Counter()
    for token, count in tokens.items():
        repeat = token_repeat_count(str(token), weights, mode)
        if repeat > 0:
            weighted[token] += count * repeat
    return weighted


def stable_hash(text: str, modulo: int = HASH_DIM) -> int:
    digest = hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def fallback_vectorize_features(cases: Sequence[Counter]) -> List[Dict[int, float]]:
    docs: List[Dict[int, float]] = []
    df: Dict[int, int] = defaultdict(int)
    for feats in cases:
        hashed: Dict[int, float] = defaultdict(float)
        for feat, cnt in feats.items():
            hashed[stable_hash(feat)] += 1.0 + math.log1p(float(cnt))
        docs.append(dict(hashed))
        for idx in hashed:
            df[idx] += 1

    n = max(1, len(docs))
    vectors: List[Dict[int, float]] = []
    for doc in docs:
        vec: Dict[int, float] = {}
        norm = 0.0
        for idx, val in doc.items():
            idf = math.log((1.0 + n) / (1.0 + df[idx])) + 1.0
            x = val * idf
            vec[idx] = x
            norm += x * x
        scale = 1.0 / math.sqrt(norm) if norm else 1.0
        vectors.append({idx: val * scale for idx, val in vec.items()})
    return vectors


def vectorize_features(feature_counters: Sequence[Counter]) -> Tuple[Any, Tuple[int, int], bool]:
    if not SKLEARN_AVAILABLE:
        vectors = fallback_vectorize_features(feature_counters)
        return vectors, (len(vectors), HASH_DIM), False
    hasher = FeatureHasher(n_features=HASH_DIM, input_type="dict", alternate_sign=False)
    counts = hasher.transform(feature_counters)
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
    matrix = tfidf.fit_transform(counts)
    matrix = normalize(matrix, norm="l2", copy=False)
    return matrix, matrix.shape, True


def sparse_dot(a: Dict[int, float], b: Dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(i, 0.0) for i, v in a.items())


def normalize_sparse_dict(vec: Dict[int, float]) -> Dict[int, float]:
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm <= 0.0:
        return {}
    return {i: v / norm for i, v in vec.items() if v}


def fallback_kmeans(vectors: Sequence[Dict[int, float]], k: int) -> List[int]:
    n = len(vectors)
    if n == 0:
        return []
    k = max(1, min(k, n))
    if k == n:
        return list(range(n))

    centers = [dict(vectors[0])]
    min_dist = [1.0 - max(0.0, sparse_dot(v, centers[0])) for v in vectors]
    while len(centers) < k:
        candidate = max(range(n), key=lambda i: (min_dist[i], -i))
        centers.append(dict(vectors[candidate]))
        for i, v in enumerate(vectors):
            dist = 1.0 - max(0.0, sparse_dot(v, centers[-1]))
            if dist < min_dist[i]:
                min_dist[i] = dist

    labels = [-1] * n
    rng = random.Random(RANDOM_SEED)
    for _ in range(14):
        changed = False
        for i, vec in enumerate(vectors):
            best = max(range(k), key=lambda c: (sparse_dot(vec, centers[c]), -c))
            if labels[i] != best:
                labels[i] = best
                changed = True

        sums: List[Dict[int, float]] = [defaultdict(float) for _ in range(k)]
        counts = [0] * k
        for label, vec in zip(labels, vectors):
            counts[label] += 1
            for idx, val in vec.items():
                sums[label][idx] += val
        for c in range(k):
            centers[c] = dict(vectors[rng.randrange(n)]) if counts[c] == 0 else normalize_sparse_dict(sums[c])
        if not changed:
            break
    return labels


def to_dense_or_reduced(X: Any, svd_dim: int) -> Any:
    n_samples, n_features = X.shape
    n_components = min(svd_dim, n_samples - 1, n_features - 1)
    if n_components >= 2:
        reduced = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED).fit_transform(X)
        return Normalizer(copy=False).fit_transform(reduced)
    return X.toarray() if hasattr(X, "toarray") else X


def cluster_vectors(X: Any, k: int, method: str = "agglomerative", svd_dim: int = 128, sklearn_input: bool = True) -> List[int]:
    n_samples = X.shape[0] if sklearn_input else len(X)
    if n_samples == 0:
        return []
    k = max(1, min(k, n_samples))
    if k == n_samples:
        return list(range(n_samples))

    if not SKLEARN_AVAILABLE or not sklearn_input:
        if method != "kmeans":
            warn(f"sklearn is unavailable; falling back from {method} to standard-library kmeans")
        return fallback_kmeans(X, k)

    if method == "kmeans":
        model = MiniBatchKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=3,
            max_iter=40,
            batch_size=max(256, min(n_samples, k * 32)),
            random_state=RANDOM_SEED,
            reassignment_ratio=0.01,
        )
        return model.fit_predict(X).tolist()

    if method == "hdbscan":
        try:
            import hdbscan  # type: ignore
        except ImportError:
            warn("hdbscan is unavailable; falling back to agglomerative")
            return cluster_vectors(X, k, method="agglomerative", svd_dim=svd_dim, sklearn_input=sklearn_input)
        reduced = to_dense_or_reduced(X, svd_dim)
        min_cluster_size = max(2, n_samples // max(2, 2 * k))
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1, metric="euclidean")
        labels = model.fit_predict(reduced).tolist()
        next_label = max([label for label in labels if label >= 0], default=-1) + 1
        out = []
        for label in labels:
            if label == -1:
                out.append(next_label)
                next_label += 1
            else:
                out.append(label)
        return out

    # TF-IDF/template features are sparse text-like vectors. Cosine average-linkage
    # agglomerative clustering often works better than k-means for log bucketing
    # because clusters may be non-spherical and uneven in size.
    reduced = to_dense_or_reduced(X, svd_dim)
    try:
        try:
            model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        except TypeError:
            model = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
        return model.fit_predict(reduced).tolist()
    except Exception as exc:
        warn(f"cosine agglomerative failed ({exc}); falling back to ward linkage")
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        return model.fit_predict(reduced).tolist()


def remap_labels(labels: Sequence[int]) -> List[int]:
    mapping: Dict[int, int] = {}
    out = []
    for label in labels:
        if label not in mapping:
            mapping[label] = len(mapping)
        out.append(mapping[label])
    return out


def write_output(path: Path, labels: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bucket"])
        for label in labels:
            writer.writerow([f"bucket_{label:03d}"])


def cluster_precomputed_distance(distance: Any, k: int) -> List[int]:
    if not SKLEARN_AVAILABLE or AgglomerativeClustering is None:
        raise RuntimeError("pairwise_mlp clustering requires scikit-learn")
    n = distance.shape[0]
    if n == 0:
        return []
    k = max(1, min(k, n))
    if k == n:
        return list(range(n))
    try:
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
    return model.fit_predict(distance).tolist()


def run_pairwise_mlp_backend(args: argparse.Namespace, input_csv: Path, effective_k: int) -> tuple[List[int], int, tuple[int, int]]:
    if not args.pairwise_model:
        raise RuntimeError("--cluster pairwise_mlp requires --pairwise-model")
    try:
        import torch
        import numpy as np
        import pairwise_features as pf
    except ImportError as exc:
        raise RuntimeError(f"pairwise_mlp requires PyTorch, NumPy, and pairwise_features: {exc}") from exc

    device = pf.resolve_torch_device(args.pairwise_device)
    checkpoint = torch.load(args.pairwise_model, map_location=device, weights_only=False)
    input_dim = int(checkpoint["input_dim"])
    hidden_dims = checkpoint.get("hidden_dims", (256, 128))
    dropout = float(checkpoint.get("dropout", 0.2))
    architecture = checkpoint.get("architecture", "plain")
    svd_dim = int(checkpoint.get("svd_dim", args.svd_dim))
    model = pf.build_pairwise_mlp_model(input_dim, hidden_dims=hidden_dims, dropout=dropout, architecture=architecture)
    model.load_state_dict(checkpoint["state_dict"])

    features, bundle = pf.build_case_features(
        input_csv,
        parser=args.parser,
        svd_dim=svd_dim,
        token_weights=args.token_weights if args.token_weight_mode != "none" else None,
        token_weight_mode=args.token_weight_mode,
    )
    prob = pf.predict_probability_matrix(
        model,
        features,
        device=device,
        batch_size=args.pairwise_batch_size,
        prob_bias=args.prob_bias,
        prob_temperature=args.prob_temperature,
    )
    prob = pf.calibrate_probability_matrix(
        prob,
        features,
        primary_floor=args.pairwise_primary_floor,
        op_pair_floor=args.pairwise_op_pair_floor,
        mismatch_floor=args.pairwise_mismatch_floor,
        conflict_penalty=args.pairwise_conflict_penalty,
        cosine_gate=args.pairwise_mismatch_cosine_gate,
    )
    distance = 1.0 - prob
    np.fill_diagonal(distance, 0.0)
    labels = cluster_precomputed_distance(distance, effective_k)
    return labels, bundle.template_count, prob.shape


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bucket Ibex regression failures.")
    parser.add_argument("--input", required=True, type=Path, help="Input CSV containing log paths.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV to write buckets.")
    parser.add_argument("--k", required=True, type=int, help="Number of requested buckets.")
    parser.add_argument("--parser", choices=("simple", "drain"), default="drain", help="log template parser")
    parser.add_argument(
        "--cluster",
        choices=("kmeans", "agglomerative", "hdbscan", "pairwise_mlp"),
        default="agglomerative",
        help="clustering backend",
    )
    parser.add_argument("--drain-depth", type=int, default=4, help="fixed-depth Drain tree depth")
    parser.add_argument("--drain-st", type=float, default=0.45, help="Drain token similarity threshold")
    parser.add_argument("--drain-max-children", type=int, default=100, help="Drain max children per tree node")
    parser.add_argument("--svd-dim", type=int, default=128, help="SVD dimension before agglomerative/HDBSCAN")
    parser.add_argument("--token-weights", type=Path, help="optional token_weights.json learned from training data")
    parser.add_argument("--token-weight-mode", choices=("repeat", "none"), default="none")
    parser.add_argument("--cluster-factor", type=float, default=1.0, help="multiply requested k before clustering")
    parser.add_argument(
        "--postprocess",
        choices=("none",),
        default="none",
        help="post-processing mode; reserved for future split_mixed support",
    )
    parser.add_argument("--pairwise-model", type=Path, help="experimental pairwise_mlp checkpoint")
    parser.add_argument("--pairwise-config", type=Path, help="optional pairwise_mlp config JSON")
    parser.add_argument("--pairwise-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--pairwise-batch-size", type=int, default=100000)
    parser.add_argument("--prob-bias", type=float, default=0.0)
    parser.add_argument("--prob-temperature", type=float, default=1.0)
    parser.add_argument("--pairwise-primary-floor", type=float, default=0.70)
    parser.add_argument("--pairwise-op-pair-floor", type=float, default=0.65)
    parser.add_argument("--pairwise-mismatch-floor", type=float, default=0.55)
    parser.add_argument("--pairwise-conflict-penalty", type=float, default=0.05)
    parser.add_argument("--pairwise-mismatch-cosine-gate", type=float, default=0.20)
    parser.add_argument("--strict-pairwise", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    start = time.perf_counter()
    args = parse_args(argv or sys.argv[1:])
    input_csv = args.input.resolve()
    rows, fieldnames = read_csv_rows(input_csv)
    sim_col = pick_column(fieldnames, "sim")
    regr_col = pick_column(fieldnames, "regr")

    if not rows:
        write_output(args.output.resolve(), [])
        return 0
    if sim_col is None and regr_col is None:
        raise SystemExit("input CSV must contain at least a sim.log or regr.log column")

    effective_k = max(1, min(len(rows), round(args.k * args.cluster_factor)))
    token_weights: Dict[str, float] = {}
    token_weights_label = "None"
    if args.token_weights and args.token_weight_mode == "none":
        warn("--token-weights was provided but --token-weight-mode=none, token weights will be ignored.")
    elif args.token_weights:
        token_weights = load_token_weights(args.token_weights)
        token_weights_label = str(args.token_weights)

    info(
        f"[config] parser={args.parser} cluster={args.cluster} "
        f"cluster_factor={args.cluster_factor} token_weight_mode={args.token_weight_mode} "
        f"token_weights={token_weights_label}"
    )
    info("[config] primary_signature=enabled")

    if args.cluster == "pairwise_mlp":
        try:
            labels, template_count, prob_shape = run_pairwise_mlp_backend(args, input_csv, effective_k)
            labels = remap_labels(labels)
            info(f"[data] cases={len(rows)} templates={template_count} vector_shape={prob_shape}")
            info(f"[cluster] requested_k={args.k} effective_k={effective_k} method=pairwise_mlp")
            write_output(args.output.resolve(), labels)
            info(
                f"[output] buckets={len(set(labels))} path={args.output.resolve()} "
                f"runtime_sec={time.perf_counter() - start:.3f}"
            )
            return 0
        except Exception as exc:
            warn(f"pairwise_mlp failed ({exc}); falling back to agglomerative")
            if args.strict_pairwise:
                return 2
            args.cluster = "agglomerative"

    base_features, normalized_lines = collect_case_inputs(input_csv, rows, sim_col, regr_col, args.parser)
    feature_counters, template_count = build_feature_counters(
        args,
        base_features,
        normalized_lines,
        token_weights=token_weights,
        token_weight_mode=args.token_weight_mode,
    )
    X, shape, sklearn_input = vectorize_features(feature_counters)
    info(f"[data] cases={len(rows)} templates={template_count} vector_shape={shape}")
    info(f"[cluster] requested_k={args.k} effective_k={effective_k} method={args.cluster}")
    labels = cluster_vectors(X, effective_k, method=args.cluster, svd_dim=args.svd_dim, sklearn_input=sklearn_input)
    labels = remap_labels(labels)
    write_output(args.output.resolve(), labels)
    # TODO: add split_mixed post-processing for large mixed clusters.
    info(f"[output] buckets={len(set(labels))} path={args.output.resolve()} runtime_sec={time.perf_counter() - start:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
