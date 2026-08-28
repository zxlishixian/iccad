#!/usr/bin/env python3
"""Regression-failure bucketing baseline with switchable parsers/backends.

Pipeline:
  input.csv -> sim.log/regr.log -> SimpleDrain or fixed-depth Drain templates
  -> case-level template/token/count features -> TF-IDF/hash vector
  -> k-means or agglomerative clustering -> output.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict, deque
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
SV_FILE_RE = re.compile(r"([^/\s:()]+\.svh?|[^/\s:()]+\.v)(?:\(\d+\))?", re.IGNORECASE)
REG_RE = re.compile(r"\bx(?:[0-2]?\d|3[01])\b", re.IGNORECASE)
CSR_RE = re.compile(
    r"\b(?:mstatus|misa|medeleg|mideleg|mie|mtvec|mcounteren|mscratch|mepc|"
    r"mcause|mtval|mip|dcsr|dpc|dscratch0?|dscratch1|sstatus|sie|stvec|"
    r"sscratch|sepc|scause|stval|sip|satp)\b",
    re.IGNORECASE,
)
PC_OP_RE = re.compile(r"\b(ibex|dut|rtl|spike|iss)\[\d+\]\s*:\s*pc\[[^\]]+\]\s+([a-z0-9.]+)", re.IGNORECASE)
SIGNAL_LINE_RE = re.compile(
    r"uvm_(?:fatal|error)|cosim|mismatch|failed|timeout|trap|exception|"
    r"interrupt|illegal|register write|pc mismatch",
    re.IGNORECASE,
)
STRUCT_TOKEN_STOPWORDS = {
    "uvm_info",
    "uvm_test_top",
    "test",
    "failed",
    "passed",
    "error",
    "seen",
    "rtl_sim.log",
    "risc-v",
    "uvm",
    "fatal",
    "cosim",
    "mismatch",
    "register",
    "write",
    "data",
    "dut",
    "expected",
    "scoreboard",
    "ibex_cosim_scoreboard.sv",
    "core_ibex_base_test.sv",
    "uvm_test_top.env.cosim_agent.scoreboard",
    "<num>",
    "<hex>",
    "<n>",
    "<time>",
    "<path>",
    "<seed>",
    "<case>",
}


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
        if path.suffix == ".gz":
            half = MAX_LOG_BYTES // 2
            head: List[str] = []
            head_chars = 0
            tail: deque[str] = deque()
            tail_chars = 0
            all_parts: List[str] | None = []
            total_chars = 0
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    total_chars += len(line)
                    if all_parts is not None:
                        all_parts.append(line)
                        if total_chars > MAX_LOG_BYTES:
                            all_parts = None
                    if head_chars < half:
                        take = line[: max(0, half - head_chars)]
                        head.append(take)
                        head_chars += len(take)
                    tail.append(line)
                    tail_chars += len(line)
                    while tail_chars > half and tail:
                        removed = tail.popleft()
                        tail_chars -= len(removed)
            if all_parts is not None:
                return "".join(all_parts), "ok"
            return "".join(head) + "\n... <LOG_TRUNCATED_HEAD_TAIL> ...\n" + "".join(tail), "ok"
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


def select_lines(text: str, mode: str = "default") -> List[str]:
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

    if mode == "signal_window":
        signal_indices = [idx for idx, line in enumerate(lines) if SIGNAL_RE.search(line)]
        for idx in signal_indices:
            for j in range(max(0, idx - 2), min(len(lines), idx + 3)):
                add(lines[j])
        if not signal_indices:
            for line in lines[:30]:
                add(line)
            for line in lines[-50:]:
                add(line)
        return chosen

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


def normalize_log_line_semantic(line: str, preserve_basenames: bool) -> str:
    """Semantic normalization for Drain.

    This keeps stable failure semantics while replacing volatile operands,
    addresses, timestamps, case ids, and exact register ids with broader slots.
    """
    s = line.strip().lower()
    s = LINE_NUM_RE.sub("", s)
    s = PATH_RE.sub(path_to_token if preserve_basenames else "<path>", s)
    s = CASE_RE.sub("<case>", s)
    s = SEED_RE.sub("<seed>", s)
    s = TIME_RE.sub("@ <time>", s)

    phrase_rewrites = [
        (r"register write data mismatch", "register_write_data_mismatch"),
        (r"\bpc mismatch\b", "pc_mismatch"),
        (r"\bcosim mismatch\b", "cosim_mismatch"),
        (r"\bco-simulation matched\b", "cosim_matched"),
        (r"\bsynchronous trap\b", "synchronous_trap"),
        (r"\billegal instruction\b", "illegal_instruction"),
        (r"\btest failed\b", "test_failed"),
        (r"\btest passed\b", "test_passed"),
        (r"\berror seen in\b", "error_seen_in"),
        (r"\bno dret detected\b", "no_dret_detected"),
        (r"\bprivilege mode switch\b", "privilege_mode_switch"),
        (r"\btimeout period\b", "timeout_period"),
    ]
    for pattern, replacement in phrase_rewrites:
        s = re.sub(pattern, replacement, s)

    def reg_slot(match: re.Match[str]) -> str:
        return f"<reg_{reg_class(match.group(0))}>"

    s = REG_RE.sub(reg_slot, s)
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


def sanitize_feature_value(value: str, max_len: int = 96) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_.:+-]+", "_", value).strip("_")
    value = re.sub(r"_+", "_", value)
    return value[:max_len].strip("_")


def reg_class(register: str) -> str:
    try:
        idx = int(register.lower().lstrip("x"))
    except ValueError:
        return "unknown"
    if idx == 0:
        return "zero"
    if idx == 1:
        return "ra"
    if idx == 2:
        return "sp"
    if idx == 3:
        return "gp"
    if idx == 4:
        return "tp"
    if 5 <= idx <= 7 or 28 <= idx <= 31:
        return "temp"
    if 8 <= idx <= 9 or 18 <= idx <= 27:
        return "saved"
    if 10 <= idx <= 17:
        return "arg"
    return "unknown"


def add_struct_feature(feats: Counter, prefix: str, name: str, value: str | None = None, weight: int = 1) -> None:
    if value is None:
        feats[f"{prefix}:struct:{name}"] += weight
        return
    cleaned = sanitize_feature_value(value)
    if cleaned:
        feats[f"{prefix}:struct:{name}:{cleaned}"] += weight


def extract_failure_reason(line: str) -> str:
    if "]" in line:
        return line.rsplit("]", 1)[-1].strip()
    if ":" in line:
        return line.rsplit(":", 1)[-1].strip()
    return line.strip()


def extract_structured_line_tokens(prefix: str, line: str, feats: Counter, limit: int = 14) -> None:
    normalized = normalize_log_line(line, preserve_basenames=True)
    emitted = 0
    for token in line_tokens(normalized):
        if token in STRUCT_TOKEN_STOPWORDS or token.startswith("<"):
            continue
        if len(token) < 3 or token.isdigit():
            continue
        if REG_RE.fullmatch(token) or CSR_RE.fullmatch(token):
            continue
        add_struct_feature(feats, prefix, "sig_tok", token, weight=1)
        emitted += 1
        if emitted >= limit:
            break


def extract_structured_failure_features(prefix: str, text: str, selected: Sequence[str], feats: Counter) -> None:
    """Add deterministic failure-semantics features from sim/regr logs only.

    These features keep stable root-cause hints that Drain may wildcard away:
    mismatch family, failing register class, CSR names, UVM source basename,
    first opcode pair, and compact tokens from the first high-signal lines.
    """
    if not text and not selected:
        return

    lower_text = text.lower()
    joined_selected = "\n".join(selected)
    lower_selected = joined_selected.lower()

    severity_seen = False
    for severity in ("uvm_fatal", "uvm_error", "uvm_warning"):
        if severity in lower_text:
            add_struct_feature(feats, prefix, "uvm_severity", severity, weight=5 if severity != "uvm_warning" else 2)
            severity_seen = True
    if severity_seen:
        for line in selected:
            if re.search(r"uvm_(?:fatal|error|warning)", line, flags=re.IGNORECASE):
                source = extract_sv_basename(line)
                add_struct_feature(feats, prefix, "uvm_source", source, weight=1)
                break

    mismatch_patterns = [
        ("pc_mismatch", r"\bpc mismatch\b"),
        ("register_write_data_mismatch", r"register write data mismatch"),
        ("sync_trap_mismatch", r"synchronous trap"),
        ("memory_mismatch", r"memory mismatch|\bstore\b.*mismatch|\bload\b.*mismatch"),
        ("instruction_mismatch", r"instruction mismatch|retired"),
        ("cosim_mismatch", r"cosim mismatch|co-sim"),
    ]
    mismatch_hit = False
    for name, pattern in mismatch_patterns:
        if re.search(pattern, lower_text):
            add_struct_feature(feats, prefix, "mismatch_type", name, weight=3)
            mismatch_hit = True
    if "mismatch" in lower_text and not mismatch_hit:
        add_struct_feature(feats, prefix, "mismatch_type", "generic", weight=2)

    for reg in sorted({m.group(0).lower() for m in REG_RE.finditer(joined_selected)}):
        add_struct_feature(feats, prefix, "reg_class", reg_class(reg), weight=1)

    reg_targets = re.findall(r"register write data mismatch to\s+(x(?:[0-2]?\d|3[01]))", joined_selected, flags=re.IGNORECASE)
    for reg in sorted({r.lower() for r in reg_targets}):
        add_struct_feature(feats, prefix, "mismatch_reg_class", reg_class(reg), weight=2)

    for csr in sorted({m.group(0).lower() for m in CSR_RE.finditer(joined_selected)}):
        add_struct_feature(feats, prefix, "csr", csr, weight=2)

    trap_keywords = [
        ("illegal_instruction", r"illegal instruction"),
        ("instruction_access_fault", r"instruction access fault"),
        ("load_access_fault", r"load access fault"),
        ("store_access_fault", r"store access fault"),
        ("ecall", r"\becall\b"),
        ("ebreak", r"\bebreak\b|breakpoint"),
        ("interrupt", r"interrupt"),
        ("timeout", r"timeout"),
        ("sync_trap", r"synchronous trap|\bsync trap\b"),
        ("exception", r"exception"),
    ]
    for name, pattern in trap_keywords:
        if re.search(pattern, lower_text):
            add_struct_feature(feats, prefix, "event", name, weight=2)

    side_ops: Dict[str, List[str]] = defaultdict(list)
    for line in selected:
        for side, op in PC_OP_RE.findall(line):
            side_key = "dut" if side.lower() in {"ibex", "dut", "rtl"} else "iss"
            op = op.lower()
            side_ops[side_key].append(op)
            add_struct_feature(feats, prefix, f"{side_key}_op", op, weight=1)
    if side_ops.get("dut") and side_ops.get("iss"):
        pair = f"{side_ops['dut'][0]}_{side_ops['iss'][0]}"
        add_struct_feature(feats, prefix, "op_pair", pair, weight=3)

    for source in sorted({m.group(1).lower() for m in SV_FILE_RE.finditer(joined_selected)}):
        add_struct_feature(feats, prefix, "source_file", source, weight=1)

    signal_lines = [line for line in selected if SIGNAL_LINE_RE.search(line)]
    if signal_lines:
        first = signal_lines[0]
        if "uvm_fatal" in first.lower():
            add_struct_feature(feats, prefix, "first_signal", "uvm_fatal", weight=3)
        elif "uvm_error" in first.lower():
            add_struct_feature(feats, prefix, "first_signal", "uvm_error", weight=3)
        elif "mismatch" in first.lower():
            add_struct_feature(feats, prefix, "first_signal", "mismatch", weight=3)
        elif "failed" in first.lower():
            add_struct_feature(feats, prefix, "first_signal", "failed", weight=2)

    # Avoid adding arbitrary signal-line tokens here: in local validation those
    # exact words improved TNR but fragmented same-bug buckets and hurt TPR.


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


def output_case_values(rows: Sequence[dict], fieldnames: Sequence[str]) -> List[str]:
    """Return case IDs for the official output CSV.

    Official testcase inputs provide a `Case` column and sample solutions emit
    `Case,bucket`.  Older local fake datasets did not require that column, so
    fall back to extracting a stable case id from log paths or, finally, row
    order.
    """
    normalized = [(name, norm_col(name)) for name in fieldnames]
    case_col = next((name for name, key in normalized if key in {"case", "caseid", "id"}), None)
    out: List[str] = []
    for idx, row in enumerate(rows):
        if case_col is not None:
            value = str(row.get(case_col, "") or "").strip()
            if value:
                out.append(value)
                continue
        out.append(case_id_from_row(row, idx, fieldnames))
    return out


def collect_case_inputs(
    input_csv: Path,
    rows: Sequence[dict],
    sim_col: str | None,
    regr_col: str | None,
    parser_kind: str,
    feature_level: str = "baseline",
    normalizer: str = "v1",
    line_mode: str = "default",
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
        # Split out the f-string expression: backslashes inside f-string braces are a
        # SyntaxError on Python < 3.12, breaking the source fallback path on RHEL 8.
        normalized_cid = re.sub(r"\\d", "0", cid)
        feats[f"case_shape:{normalized_cid}"] += 1

        for prefix, col in (("sim", sim_col), ("regr", regr_col)):
            path = resolve_log_path(input_csv, row.get(col) if col else None)
            text, status = read_log_sample(path)
            selected = select_lines(text, mode=line_mode)
            selected_by_prefix[prefix] = selected
            info_by_prefix[prefix] = {"path": str(path) if path else "", "status": status}
            feats[f"{prefix}:file_status:{status}"] += 3
            if path is not None:
                feats[f"{prefix}:basename:{path.name.lower()}"] += 1
            extract_status_features(prefix, text, feats)
            if feature_level == "structured":
                extract_structured_failure_features(prefix, text, selected, feats)

            for line in selected:
                if normalizer == "semantic":
                    normalized_line = normalize_log_line_semantic(line, preserve_basenames=preserve_basenames)
                else:
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


def template_feature_weights(template: str, mode: str) -> Tuple[int, int, int]:
    """Return template/token/bigram weights for a Drain template."""
    if mode != "quality":
        return 2, 1, 1

    t = template.lower()
    low_signal = [
        "wall-clock timeout is set",
        "test_timeout_s",
        "0.00 pass",
        "passed <n> failed",
        "error_seen_in rtl_sim.log",
        "no failing tests",
        "number of uvm_",
        "demoted uvm_",
        "caught uvm_",
        "risc-v uvm test_failed",
    ]
    if any(pattern in t for pattern in low_signal):
        return 0, 0, 0

    high_signal = [
        "uvm_fatal",
        "uvm_error",
        "cosim_mismatch",
        "register_write_data_mismatch",
        "pc_mismatch",
        "synchronous_trap",
        "no_dret_detected",
        "privilege_mode_switch",
        "illegal_instruction",
    ]
    medium_signal = [
        "mismatch",
        "trap",
        "exception",
        "interrupt",
        "timeout_period",
        "scoreboard",
        "ibex_cosim_scoreboard.sv",
    ]
    if any(pattern in t for pattern in high_signal):
        return 4, 2, 2
    if any(pattern in t for pattern in medium_signal):
        return 2, 1, 1
    return 1, 0, 0


def build_feature_counters(
    args: argparse.Namespace,
    base_features: Sequence[Counter],
    normalized_lines: Sequence[List[Tuple[str, str]]],
    token_weights: Dict[str, float] | None = None,
    token_weight_mode: str = "none",
) -> Tuple[List[Counter], int]:
    parser = make_parser(args)
    template_weighting = getattr(args, "template_weighting", "none")
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
            tmpl_weight, tok_weight, bi_weight = template_feature_weights(template, template_weighting)
            if tmpl_weight <= 0 and tok_weight <= 0 and bi_weight <= 0:
                continue
            out[f"{prefix}:tmpl:{template}"] += tmpl_weight
            toks = list(line_tokens(template))
            if tok_weight > 0:
                for tok in toks:
                    out[f"{prefix}:tok:{tok}"] += tok_weight
            if bi_weight > 0:
                for a, b in zip(toks, toks[1:]):
                    out[f"{prefix}:bi:{a}_{b}"] += bi_weight
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


def cluster_vectors(
    X: Any,
    k: int,
    method: str = "agglomerative",
    svd_dim: int = 128,
    sklearn_input: bool = True,
    pre_reduced: bool = False,
) -> List[int]:
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
        reduced = X if pre_reduced else to_dense_or_reduced(X, svd_dim)
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
    reduced = X if pre_reduced else to_dense_or_reduced(X, svd_dim)
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


def write_output(path: Path, labels: Sequence[int], cases: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if cases is None:
        cases = [str(idx + 1) for idx in range(len(labels))]
    if len(cases) != len(labels):
        raise ValueError(f"number of cases ({len(cases)}) does not match labels ({len(labels)})")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Case", "bucket"])
        for case, label in zip(cases, labels):
            writer.writerow([case, f"bucket_{label:03d}"])


def parse_llm_config_without_yaml(raw: str) -> dict:
    """Small fallback parser for the contest's simple LLM_MODEL_CONFIG YAML."""
    lines = raw.splitlines()
    in_embedding = False
    data: Dict[str, str] = {}
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if indent == 0 and stripped.startswith("embedding:"):
            in_embedding = True
            continue
        if indent == 0 and stripped.endswith(":"):
            in_embedding = False
        if not in_embedding or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"“”")
        if key in {"model_name", "api_key", "base_url", "model", "client_library"} and value:
            data[key] = value
    return {"embedding": {"model_name": data.get("model_name"), "config": data}}


def load_llm_embedding_config() -> dict | None:
    raw = os.getenv("LLM_MODEL_CONFIG")
    if not raw:
        return None
    if raw.lstrip().startswith(("/", "./", "~")) or raw.rstrip().endswith((".yaml", ".yml")):
        warn(
            "LLM_MODEL_CONFIG looks like a file path, not YAML content. "
            "Use: export LLM_MODEL_CONFIG=\"$(cat /path/to/config.yaml)\""
        )
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(raw)
    except ImportError:
        cfg = parse_llm_config_without_yaml(raw)
    except Exception as exc:
        warn(f"could not parse LLM_MODEL_CONFIG ({exc}); LLM embeddings disabled")
        return None
    if not isinstance(cfg, dict):
        warn(
            "LLM_MODEL_CONFIG parsed as a plain value, not a YAML mapping. "
            "Make sure the variable contains YAML content, not a file path. "
            "Use: export LLM_MODEL_CONFIG=\"$(cat /path/to/config.yaml)\""
        )
        return None
    emb = cfg.get("embedding") or {}
    if not isinstance(emb, dict):
        return None
    emb_cfg = emb.get("config") or {}
    if not isinstance(emb_cfg, dict):
        return None
    model = emb_cfg.get("model") or emb.get("model_name")
    api_key = emb_cfg.get("api_key")
    base_url = emb_cfg.get("base_url")
    if not model or not api_key or not base_url:
        warn("LLM_MODEL_CONFIG is missing embedding.config model/api_key/base_url; LLM embeddings disabled")
        return None
    return {
        "model": str(model),
        "model_name": str(emb.get("model_name") or model),
        "api_key": str(api_key),
        "base_url": str(base_url),
    }


def llm_feature_priority(feature: str) -> int:
    if feature.startswith("PRIMARY_"):
        return 100
    if ":tmpl:" in feature:
        lower = feature.lower()
        if any(
            key in lower
            for key in (
                "uvm_fatal",
                "uvm_error",
                "cosim",
                "mismatch",
                "pc_mismatch",
                "register_write",
                "synchronous_trap",
                "no_dret",
                "illegal_instruction",
            )
        ):
            return 90
        return 50
    if ":struct:" in feature or ":flag:" in feature:
        return 75
    if ":count:" in feature:
        return 35
    return 0


def format_llm_feature(feature: str) -> str:
    if feature.startswith("PRIMARY_"):
        return f"primary_signature: {feature}"
    if ":tmpl:" in feature:
        prefix, template = feature.split(":tmpl:", 1)
        return f"{prefix} drain_template: {template}"
    if ":struct:" in feature:
        prefix, value = feature.split(":struct:", 1)
        return f"{prefix} structured: {value}"
    if ":flag:" in feature:
        prefix, value = feature.split(":flag:", 1)
        return f"{prefix} flag: {value}"
    if ":count:" in feature:
        prefix, value = feature.split(":count:", 1)
        return f"{prefix} count: {value}"
    return feature


def build_llm_case_documents_features(feature_counters: Sequence[Counter], max_features: int) -> List[str]:
    docs: List[str] = []
    for idx, feats in enumerate(feature_counters):
        candidates = []
        for feature, count in feats.items():
            priority = llm_feature_priority(str(feature))
            if priority <= 0:
                continue
            candidates.append((priority, float(count), str(feature)))
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        lines = ["task: regression failure bucketing case summary"]
        for _, count, feature in candidates[:max_features]:
            lines.append(f"{format_llm_feature(feature)} count={count:g}")
        docs.append("\n".join(lines))
    return docs


def build_llm_case_documents_summary(feature_counters: Sequence[Counter], max_features: int) -> List[str]:
    docs: List[str] = []
    for idx, feats in enumerate(feature_counters):
        primary = []
        templates = []
        flags = []
        structured = []
        counts = []
        for feature, count in feats.items():
            text = str(feature)
            item = (float(count), format_llm_feature(text))
            if text.startswith("PRIMARY_"):
                primary.append(item)
            elif ":tmpl:" in text:
                if llm_feature_priority(text) >= 80:
                    templates.append(item)
            elif ":flag:" in text:
                flags.append(item)
            elif ":struct:" in text:
                structured.append(item)
            elif ":count:" in text:
                counts.append(item)

        def top(items: List[Tuple[float, str]], limit: int) -> List[str]:
            items.sort(key=lambda item: (-item[0], item[1]))
            return [f"- {text} count={count:g}" for count, text in items[:limit]]

        template_limit = max(6, max_features // 2)
        lines = [
            "Regression failure case summary.",
            "Use this text to compare whether two cases likely share the same root-cause design bug.",
            "Primary signatures:",
            *top(primary, 6),
            "High-signal failure templates:",
            *top(templates, template_limit),
            "Failure flags and structured hints:",
            *top(flags + structured, max(8, max_features // 4)),
            "Compact counts:",
            *top(counts, 8),
        ]
        docs.append("\n".join(line for line in lines if line.strip()))
    return docs


def build_llm_case_documents(feature_counters: Sequence[Counter], max_features: int, doc_style: str) -> List[str]:
    if doc_style == "summary":
        return build_llm_case_documents_summary(feature_counters, max_features)
    return build_llm_case_documents_features(feature_counters, max_features)


def cache_key_for_llm_doc(model: str, doc: str) -> str:
    digest = hashlib.blake2b(f"{model}\n{doc}".encode("utf-8", "ignore"), digest_size=16).hexdigest()
    return digest


async def _fetch_embeddings_openai(
    client: Any,
    model: str,
    batch: List[Tuple[int, str, Path]],
    timeout_sec: float,
) -> None:
    """OpenAI-compatible backend (Together AI, Fireworks, Ollama, etc.)."""
    _ = timeout_sec
    resp = await client.embeddings.create(model=model, input=[doc for _, doc, _ in batch])
    ordered = sorted(resp.data, key=lambda item: getattr(item, "index", 0))
    for (idx, _, path), item in zip(batch, ordered):
        yield idx, list(item.embedding), path


async def _fetch_embeddings_huggingface(
    session: Any,
    model: str,
    api_key: str,
    batch: List[Tuple[int, str, Path]],
    timeout_sec: float,
    max_retries: int = 3,
) -> None:
    """HuggingFace Serverless Inference API backend."""
    import aiohttp
    import asyncio as _asyncio
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: Any = {"inputs": [doc for _, doc, _ in batch]}
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_sec)
            async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
                if resp.status == 503 or resp.status == 429:
                    delay = min(2.0 ** (attempt + 1), 30.0)
                    warn(f"HF API {resp.status} (attempt {attempt+1}/{max_retries}), retry in {delay:.0f}s")
                    await _asyncio.sleep(delay)
                    continue
                data: Any = await resp.json()
                if resp.status >= 400:
                    raise RuntimeError(f"HF API error {resp.status}: {data}")
            if isinstance(data, dict) and "error" in data:
                raise RuntimeError(f"HF API returned error: {data['error']}")
            if not isinstance(data, list):
                raise RuntimeError(f"unexpected HF response type: {type(data)}")
            for i, (idx, _, path) in enumerate(batch):
                if i >= len(data):
                    raise RuntimeError(f"HF response missing embedding for index {i}")
                emb = data[i] if isinstance(data[i], list) else list(data[i])
                yield idx, list(emb), path
            return
        except (aiohttp.ClientError, OSError, RuntimeError) as exc:
            if attempt < max_retries - 1:
                delay = min(2.0 ** (attempt + 1), 30.0)
                warn(f"HF request failed ({exc}), retry in {delay:.0f}s")
                await _asyncio.sleep(delay)
            else:
                raise RuntimeError(f"HF API failed after {max_retries} attempts: {exc}") from exc


async def fetch_llm_embeddings_async(
    docs: Sequence[str],
    cfg: dict,
    cache_dir: Path,
    batch_size: int,
    timeout_sec: float,
) -> List[List[float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = cfg["model"]
    base_url = cfg.get("base_url", "")
    api_key = cfg.get("api_key", "")

    cached: Dict[int, List[float]] = {}
    missing: List[Tuple[int, str, Path]] = []
    for idx, doc in enumerate(docs):
        key = cache_key_for_llm_doc(model, doc)
        path = cache_dir / f"{key}.json"
        try:
            with path.open(encoding="utf-8") as f:
                cached[idx] = json.load(f)["embedding"]
        except Exception:
            missing.append((idx, doc, path))

    if not missing:
        return [cached[idx] for idx in range(len(docs))]

    is_hf = "huggingface" in base_url.lower()

    if is_hf:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            for start in range(0, len(missing), batch_size):
                batch = missing[start : start + batch_size]
                async for idx, emb, path in _fetch_embeddings_huggingface(
                    session, model, api_key, batch, timeout_sec
                ):
                    cached[idx] = emb
                    try:
                        with path.open("w", encoding="utf-8") as f:
                            json.dump({"model": model, "embedding": emb}, f)
                    except OSError:
                        pass
    else:
        try:
            from openai import AsyncOpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("openai package is not installed") from exc
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_sec,
            max_retries=1,
        )
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            async for idx, emb, path in _fetch_embeddings_openai(
                client, model, batch, timeout_sec
            ):
                cached[idx] = emb
                try:
                    with path.open("w", encoding="utf-8") as f:
                        json.dump({"model": model, "embedding": emb}, f)
                except OSError:
                    pass

    return [cached[idx] for idx in range(len(docs))]


def _fetch_embeddings_openai_sync(client: Any, model: str, batch: List[Tuple[int, str, Path]], timeout_sec: float):
    """Synchronous OpenAI-compatible backend (nomic, dashscope, ollama, ...).

    Replaces the async ``AsyncOpenAI`` path: repeated ``asyncio.run`` + httpx left a
    dangling ``AsyncClient.aclose()`` task that fired after the loop closed
    ("Event loop is closed") and eventually deadlocked the 8th/9th dataset.  The
    embedding fetch was never actually concurrent (batches are issued one at a
    time), so a plain sync client is strictly simpler and deadlock-free.
    """
    _ = timeout_sec
    resp = client.embeddings.create(model=model, input=[doc for _, doc, _ in batch])
    ordered = sorted(resp.data, key=lambda item: getattr(item, "index", 0))
    for (idx, _, path), item in zip(batch, ordered):
        yield idx, list(item.embedding), path


def _fetch_embeddings_huggingface_sync(model: str, api_key: str, batch: List[Tuple[int, str, Path]], timeout_sec: float, max_retries: int = 3):
    """Synchronous HuggingFace Serverless Inference API backend."""
    import requests
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: Any = {"inputs": [doc for _, doc, _ in batch]}
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout_sec)
            if resp.status_code in (503, 429):
                delay = min(2.0 ** (attempt + 1), 30.0)
                warn(f"HF API {resp.status_code} (attempt {attempt+1}/{max_retries}), retry in {delay:.0f}s")
                time.sleep(delay)
                continue
            data: Any = resp.json()
            if resp.status_code >= 400:
                raise RuntimeError(f"HF API error {resp.status_code}: {data}")
            if isinstance(data, dict) and "error" in data:
                raise RuntimeError(f"HF API returned error: {data['error']}")
            if not isinstance(data, list):
                raise RuntimeError(f"unexpected HF response type: {type(data)}")
            for i, (idx, _, path) in enumerate(batch):
                if i >= len(data):
                    raise RuntimeError(f"HF response missing embedding for index {i}")
                emb = data[i] if isinstance(data[i], list) else list(data[i])
                yield idx, list(emb), path
            return
        except (requests.RequestException, OSError, RuntimeError) as exc:
            if attempt < max_retries - 1:
                delay = min(2.0 ** (attempt + 1), 30.0)
                warn(f"HF request failed ({exc}), retry in {delay:.0f}s")
                time.sleep(delay)
            else:
                raise RuntimeError(f"HF API failed after {max_retries} attempts: {exc}") from exc


def _fetch_llm_embeddings_sync(
    docs: Sequence[str],
    cfg: dict,
    cache_dir: Path,
    batch_size: int,
    timeout_sec: float,
) -> List[List[float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = cfg["model"]
    base_url = cfg.get("base_url", "")
    api_key = cfg.get("api_key", "")

    cached: Dict[int, List[float]] = {}
    missing: List[Tuple[int, str, Path]] = []
    for idx, doc in enumerate(docs):
        key = cache_key_for_llm_doc(model, doc)
        path = cache_dir / f"{key}.json"
        try:
            with path.open(encoding="utf-8") as f:
                cached[idx] = json.load(f)["embedding"]
        except Exception:
            missing.append((idx, doc, path))

    if not missing:
        return [cached[idx] for idx in range(len(docs))]

    is_hf = "huggingface" in base_url.lower()

    if is_hf:
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            for idx, emb, path in _fetch_embeddings_huggingface_sync(model, api_key, batch, timeout_sec):
                cached[idx] = emb
                try:
                    with path.open("w", encoding="utf-8") as f:
                        json.dump({"model": model, "embedding": emb}, f)
                except OSError:
                    pass
    else:
        from openai import OpenAI  # type: ignore
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_sec,
            max_retries=1,
        )
        try:
            for start in range(0, len(missing), batch_size):
                batch = missing[start : start + batch_size]
                for idx, emb, path in _fetch_embeddings_openai_sync(client, model, batch, timeout_sec):
                    cached[idx] = emb
                    try:
                        with path.open("w", encoding="utf-8") as f:
                            json.dump({"model": model, "embedding": emb}, f)
                    except OSError:
                        pass
        finally:
            try:
                client.close()
            except Exception:
                pass

    return [cached[idx] for idx in range(len(docs))]


def fetch_llm_embeddings(
    docs: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[Any, str]:
    cfg = load_llm_embedding_config()
    if cfg is None:
        raise RuntimeError("LLM_MODEL_CONFIG is not set or has no valid embedding endpoint")
    embeddings = _fetch_llm_embeddings_sync(
        docs,
        cfg,
        cache_dir=args.llm_cache_dir,
        batch_size=args.llm_batch_size,
        timeout_sec=args.llm_timeout_sec,
    )
    return embeddings, cfg["model_name"]


def augment_with_llm_embeddings(
    X: Any,
    feature_counters: Sequence[Counter],
    args: argparse.Namespace,
    sklearn_input: bool,
) -> Tuple[Any, Tuple[int, int], bool]:
    if not SKLEARN_AVAILABLE or not sklearn_input:
        raise RuntimeError("LLM embedding augmentation requires scikit-learn vectorization")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("LLM embedding augmentation requires numpy") from exc

    docs = build_llm_case_documents(
        feature_counters,
        max_features=args.llm_doc_max_features,
        doc_style=args.llm_doc_style,
    )
    embeddings, model_name = fetch_llm_embeddings(docs, args)
    base = to_dense_or_reduced(X, args.svd_dim)
    base = Normalizer(copy=False).fit_transform(base)
    llm = np.asarray(embeddings, dtype="float32")
    if llm.ndim != 2 or llm.shape[0] != len(feature_counters):
        raise RuntimeError(f"unexpected embedding matrix shape: {llm.shape}")
    llm = Normalizer(copy=False).fit_transform(llm)
    combined = np.concatenate([base, llm * float(args.llm_weight)], axis=1)
    combined = Normalizer(copy=False).fit_transform(combined)
    info(
        f"[llm] mode=embedding model={model_name} docs={len(docs)} "
        f"embedding_dim={llm.shape[1]} fusion=concat doc_style={args.llm_doc_style} "
        f"weight={args.llm_weight}"
    )
    return combined, combined.shape, True


# ── Trace feature augmentation ────────────────────────────────────────────

def _parse_trace_tail_features(trace_path: Path | None, tail_lines: int = 256) -> dict:
    """Extract compact trace features from the last N instructions.

    Returns dict with scalar features (all 0.0 if trace missing/unparseable).
    """
    if trace_path is None or not trace_path.exists():
        return {"trace_missing": 1.0}

    try:
        if trace_path.suffix == ".gz":
            with gzip.open(trace_path, "rt", encoding="utf-8", errors="replace") as fh:
                lines = [l.rstrip("\n") for l in fh if l.strip()]
        else:
            with open(trace_path, encoding="utf-8", errors="replace") as fh:
                lines = [l.rstrip("\n") for l in fh if l.strip()]
    except (OSError, gzip.BadGzipFile):
        return {"trace_missing": 1.0}

    # Parse tab-separated trace lines
    import re
    tab_re = re.compile(
        r"^\s*\d+\s+\d+\s+(?P<pc>[0-9a-fA-F]{6,16})\s+"
        r"[0-9a-fA-F]+\s+(?P<decoded>.+?)(?:\s{2,}.*)?$",
        re.IGNORECASE,
    )
    opcode_re = re.compile(r"^\s*([a-zA-Z][a-zA-Z0-9_.]*)")

    decoded = []
    pcs = []
    for line in lines[-tail_lines * 3:]:
        m = tab_re.match(line)
        if m:
            decoded.append(m.group("decoded").strip())
            pcs.append(m.group("pc"))

    if not decoded:
        return {"trace_missing": 1.0}

    tail = decoded[-tail_lines:]
    tail_pcs = pcs[-tail_lines:] if len(pcs) >= len(tail) else pcs

    # Classify opcodes
    load_ops = {"lb","lh","lw","lbu","lhu","ld","lwu","c.lw","c.ld","c.lwsp","c.ldsp","flw","fld"}
    store_ops = {"sb","sh","sw","sd","c.sw","c.sd","c.swsp","c.sdsp","fsw","fsd"}
    branch_ops = {"beq","bne","blt","bge","bltu","bgeu","beqz","bnez","c.beqz","c.bnez"}
    jump_ops = {"jal","jalr","c.jal","c.jalr","c.j","c.jr","ret","c.ret"}
    csr_ops = {"csrrw","csrrs","csrrc","csrrwi","csrrsi","csrrci"}
    system_ops = {"ecall","ebreak","mret","sret","dret","wfi","fence","fence.i","sfence.vma"}

    n = len(tail)
    loads = stores = branches = jumps = csrs = systems = compressed = exceptions = 0
    opcodes_seen = set()

    for instr in tail:
        om = opcode_re.match(instr)
        op = om.group(1).lower() if om else ""
        opcodes_seen.add(op)
        if op in load_ops: loads += 1
        elif op in store_ops: stores += 1
        elif op in branch_ops: branches += 1
        elif op in jump_ops: jumps += 1
        elif op in csr_ops: csrs += 1
        elif op in system_ops: systems += 1
        if op.startswith("c."): compressed += 1
        instr_lower = instr.lower()
        if any(w in instr_lower for w in ("exception","trap","interrupt","illegal","fault")):
            exceptions += 1

    # Loop detection: check if last 8 instructions repeat
    has_loop = 0.0
    if n >= 16:
        last8 = tuple(tail[-8:])
        prev8 = tuple(tail[-16:-8])
        if last8 == prev8:
            has_loop = 1.0

    # PC region entropy
    pc_coarse = {pc[:4] for pc in tail_pcs} if tail_pcs else set()
    pc_regions = len(pc_coarse) / max(1, len(set(tail_pcs))) if tail_pcs else 0.0

    # Normalize to [0,1] range
    return {
        "trace_load_ratio": loads / max(1, n),
        "trace_store_ratio": stores / max(1, n),
        "trace_branch_ratio": (branches + jumps) / max(1, n),
        "trace_csr_ratio": csrs / max(1, n),
        "trace_system_ratio": systems / max(1, n),
        "trace_compressed_ratio": compressed / max(1, n),
        "trace_has_loop": has_loop,
        "trace_exception_ratio": exceptions / max(1, n),
        "trace_unique_opcodes": len(opcodes_seen) / max(1, n),
        "trace_pc_regions": pc_regions,
        "trace_log_instructions": __import__("math").log1p(len(decoded)),
        "trace_missing": 0.0,
    }


def augment_with_trace_features(
    X: Any,
    input_csv: Path,
    rows: list[dict],
    fieldnames: list[str],
    args: argparse.Namespace,
    sklearn_input: bool,
) -> Tuple[Any, Tuple[int, int], bool]:
    """Append compact trace features to case vectors.

    Reads trace.log.gz for each case, extracts lightweight scalar features,
    and concatenates them to the existing vector matrix.
    Gracefully handles missing traces (all-zeros).
    """
    if not SKLEARN_AVAILABLE or not sklearn_input:
        warn("[trace] trace augmentation requires scikit-learn; skipping")
        return X, X.shape if hasattr(X, "shape") else (0, 0), sklearn_input

    import numpy as np

    # Find trace column
    trace_col = None
    norm = {"".join(ch for ch in f.lower() if ch.isalnum()): f for f in fieldnames}
    for cand in ("tracelog", "tracelog.gz", "trace", "trace_log", "tracefile"):
        key = "".join(ch for ch in cand.lower() if ch.isalnum())
        if key in norm:
            trace_col = norm[key]
            break
    if not trace_col:
        for f in fieldnames:
            if "trace" in "".join(ch for ch in f.lower() if ch.isalnum()):
                trace_col = f
                break

    if not trace_col:
        info("[trace] no trace column found; skipping trace augmentation")
        return X, X.shape if hasattr(X, "shape") else (0, 0), sklearn_input

    # Extract trace features per case
    trace_feats = []
    for row in rows:
        path_val = row.get(trace_col)
        path = resolve_log_path(input_csv, path_val) if path_val else None
        feats = _parse_trace_tail_features(path)
        trace_feats.append(feats)

    feat_keys = sorted(trace_feats[0].keys()) if trace_feats else []
    trace_mat = np.array([[f.get(k, 0.0) for k in feat_keys] for f in trace_feats], dtype="float32")

    # Get base vectors (already LLM-augmented if --llm-mode embedding)
    if sklearn_input:
        base = to_dense_or_reduced(X, args.svd_dim)
        base = Normalizer(copy=False).fit_transform(base)
    else:
        base = X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype="float32")

    # Append trace features with lower weight so they're auxiliary, not dominant
    trace_weight = 0.5
    combined = np.concatenate([base, trace_mat * trace_weight], axis=1)
    combined = Normalizer(copy=False).fit_transform(combined)

    info(f"[trace] augmented with {len(feat_keys)} trace features (weight={trace_weight}), "
         f"shape=({combined.shape[0]}, {combined.shape[1]})")
    return combined, combined.shape, True


def cluster_with_llm_similarity_fusion(
    X: Any,
    feature_counters: Sequence[Counter],
    args: argparse.Namespace,
    effective_k: int,
    sklearn_input: bool,
) -> Tuple[List[int], Tuple[int, int], int]:
    if not SKLEARN_AVAILABLE or not sklearn_input:
        raise RuntimeError("LLM similarity fusion requires scikit-learn vectorization")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("LLM similarity fusion requires numpy") from exc

    docs = build_llm_case_documents(
        feature_counters,
        max_features=args.llm_doc_max_features,
        doc_style=args.llm_doc_style,
    )
    embeddings, model_name = fetch_llm_embeddings(docs, args)
    det = to_dense_or_reduced(X, args.svd_dim)
    det = Normalizer(copy=False).fit_transform(det)
    llm = np.asarray(embeddings, dtype="float32")
    if llm.ndim != 2 or llm.shape[0] != len(feature_counters):
        raise RuntimeError(f"unexpected embedding matrix shape: {llm.shape}")
    llm = Normalizer(copy=False).fit_transform(llm)

    det_sim = det @ det.T
    llm_sim = llm @ llm.T
    alpha = max(0.0, min(1.0, float(args.llm_alpha)))
    sim = alpha * det_sim + (1.0 - alpha) * llm_sim
    sim = np.clip(sim, -1.0, 1.0)
    distance = 1.0 - sim
    np.fill_diagonal(distance, 0.0)
    info(
        f"[llm] mode=embedding model={model_name} docs={len(docs)} "
        f"embedding_dim={llm.shape[1]} fusion=similarity alpha={alpha} "
        f"doc_style={args.llm_doc_style}"
    )
    selected_k = effective_k
    if args.k_selection == "dynamic":
        selected_k = select_dynamic_k_from_similarity(
            sim,
            requested_k=args.k,
            window=args.dynamic_k_window,
            merge_threshold=args.dynamic_merge_threshold,
            top_pairs=args.dynamic_top_pairs,
            label="llm_similarity",
            policy=args.dynamic_k_policy,
            gap_min=args.dynamic_gap_min,
            gap_ratio=args.dynamic_gap_ratio,
            start_factor=args.dynamic_start_factor,
            min_factor=args.dynamic_min_factor,
            local_quantile=args.dynamic_local_quantile,
            below_k_margin=args.dynamic_below_k_margin,
        )
    labels = cluster_precomputed_distance(distance, selected_k)
    return labels, distance.shape, selected_k


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


def initial_effective_k(args: argparse.Namespace, n_cases: int) -> int:
    if args.k_selection in {"fixed", "dynamic"}:
        return max(1, min(n_cases, int(args.k)))
    return max(1, min(n_cases, round(args.k * args.cluster_factor)))


def _dynamic_k_bounds(requested_k: int, n_cases: int, window: int) -> tuple[int, int]:
    requested_k = max(1, min(n_cases, int(requested_k)))
    window = max(0, int(window))
    lower = max(1, requested_k - window)
    upper = min(n_cases, requested_k + window)
    return lower, max(lower, upper)


def _dynamic_k_factor_bounds(
    requested_k: int,
    n_cases: int,
    start_factor: float,
    min_factor: float,
) -> tuple[int, int]:
    requested_k = max(1, min(n_cases, int(requested_k)))
    start_factor = max(1.0, float(start_factor))
    min_factor = max(0.0, min(float(min_factor), start_factor))
    upper = min(n_cases, max(requested_k, int(math.ceil(requested_k * start_factor))))
    lower = max(1, min(requested_k, int(math.floor(requested_k * min_factor))))
    return lower, max(lower, upper)


def _best_intercluster_score(similarity: Any, labels: Sequence[int], top_pairs: int) -> float:
    import numpy as np

    clusters: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[int(label)].append(idx)
    best = 0.0
    keys = sorted(clusters)
    top_pairs = max(1, int(top_pairs))
    for pos, a in enumerate(keys):
        ia = clusters[a]
        for b in keys[pos + 1:]:
            ib = clusters[b]
            vals = similarity[np.ix_(ia, ib)].reshape(-1)
            if vals.size == 0:
                continue
            if vals.size > top_pairs:
                vals = np.partition(vals, vals.size - top_pairs)[-top_pairs:]
            score = float(np.mean(vals))
            if score > best:
                best = score
    return best


def select_dynamic_k_from_similarity(
    similarity: Any,
    requested_k: int,
    window: int,
    merge_threshold: float,
    top_pairs: int,
    label: str = "",
    policy: str = "gap",
    gap_min: float = 0.05,
    gap_ratio: float = 0.92,
    start_factor: float = 1.2,
    min_factor: float = 0.8,
    local_quantile: float = 0.75,
    below_k_margin: float = 0.02,
) -> int:
    import numpy as np

    n = int(similarity.shape[0])
    if n <= 1:
        return n
    if not SKLEARN_AVAILABLE or AgglomerativeClustering is None:
        warn("dynamic k-selection requires scikit-learn; falling back to requested k")
        return max(1, min(n, int(requested_k)))
    policy = str(policy or "gap")
    if policy == "reference_band":
        lower, upper = _dynamic_k_factor_bounds(requested_k, n, start_factor, min_factor)
    else:
        lower, upper = _dynamic_k_bounds(requested_k, n, window)
    if lower == upper:
        return lower

    sim = np.asarray(similarity, dtype=np.float32)
    sim = np.clip(sim, 0.0, 1.0)
    distance = 1.0 - sim
    np.fill_diagonal(distance, 0.0)

    scores: List[tuple[int, float]] = []
    for candidate_k in range(upper, lower, -1):
        labels = cluster_precomputed_distance(distance, candidate_k)
        best_score = _best_intercluster_score(sim, labels, top_pairs=top_pairs)
        scores.append((candidate_k, best_score))

    selected = lower
    if policy == "threshold":
        for candidate_k, best_score in scores:
            if best_score < float(merge_threshold):
                selected = candidate_k
                break
    elif policy == "gap":
        # Scores are the best available merge from candidate_k clusters to
        # candidate_k - 1 clusters. Stop at the first local cliff near k:
        # if the next merge score drops sharply relative to the previous
        # merge, the lower-score merge is likely crossing a bug boundary.
        prev_score = None
        for candidate_k, best_score in scores:
            if prev_score is not None:
                abs_drop = float(prev_score) - float(best_score)
                ratio = float(best_score) / max(float(prev_score), 1e-12)
                if abs_drop >= float(gap_min) and ratio <= float(gap_ratio):
                    selected = candidate_k
                    break
            prev_score = best_score
        else:
            selected = lower
    elif policy == "reference_band":
        score_values = np.asarray([score for _, score in scores], dtype=np.float32)
        cutoff = float(np.quantile(score_values, max(0.0, min(1.0, float(local_quantile)))))
        requested_k = max(1, min(n, int(requested_k)))
        requested_score = next((score for k, score in scores if k == requested_k), None)
        if requested_score is None:
            requested_score = float(np.median(score_values))

        selected = upper
        prev_score = None
        for candidate_k, best_score in scores:
            best_score = float(best_score)
            if prev_score is not None:
                abs_drop = float(prev_score) - best_score
                ratio = best_score / max(float(prev_score), 1e-12)
                if abs_drop >= float(gap_min) and ratio <= float(gap_ratio):
                    selected = candidate_k
                    break

            if candidate_k <= requested_k:
                required = max(cutoff, float(requested_score) + float(below_k_margin))
                if best_score < required:
                    selected = candidate_k
                    break

            selected = max(lower, candidate_k - 1)
            prev_score = best_score
    else:
        raise ValueError(f"unknown dynamic k policy: {policy}")

    trace = [f"{k}:{score:.3f}" for k, score in scores]
    context = f" {label}" if label else ""
    info(
        f"[cluster] dynamic_k{context} requested_k={requested_k} range={lower}-{upper} "
        f"selected_k={selected} policy={policy} merge_threshold={float(merge_threshold):.3f} "
        f"gap_min={float(gap_min):.3f} gap_ratio={float(gap_ratio):.3f} "
        f"start_factor={float(start_factor):.3f} min_factor={float(min_factor):.3f} "
        f"local_quantile={float(local_quantile):.3f} below_k_margin={float(below_k_margin):.3f} "
        f"best_next_merge=" + ",".join(trace)
    )
    return selected


def select_dynamic_k_for_vectors(
    X: Any,
    args: argparse.Namespace,
    sklearn_input: bool = True,
    pre_reduced: bool = False,
) -> int:
    if not SKLEARN_AVAILABLE or not sklearn_input:
        warn("dynamic k-selection is unavailable without scikit-learn vector features; falling back to requested k")
        return max(1, min(X.shape[0] if sklearn_input else len(X), int(args.k)))
    import numpy as np

    reduced = X if pre_reduced else to_dense_or_reduced(X, args.svd_dim)
    mat = np.asarray(reduced.toarray() if hasattr(reduced, "toarray") else reduced, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.maximum(norms, 1e-12)
    sim = mat @ mat.T
    np.fill_diagonal(sim, 1.0)
    return select_dynamic_k_from_similarity(
        sim,
        requested_k=args.k,
        window=args.dynamic_k_window,
        merge_threshold=args.dynamic_merge_threshold,
        top_pairs=args.dynamic_top_pairs,
        label="vectors",
        policy=args.dynamic_k_policy,
        gap_min=args.dynamic_gap_min,
        gap_ratio=args.dynamic_gap_ratio,
        start_factor=args.dynamic_start_factor,
        min_factor=args.dynamic_min_factor,
        local_quantile=args.dynamic_local_quantile,
        below_k_margin=args.dynamic_below_k_margin,
    )


def run_pairwise_mlp_backend(args: argparse.Namespace, input_csv: Path, effective_k: int) -> tuple[List[int], int, tuple[int, int], int]:
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
    if checkpoint.get("feature_mode"):
        import pairwise_llm_features as plf
        model_pkg = plf.load_model_pkg(args.pairwise_model)
        model_pkg["device"] = device
        feature_mode = str(model_pkg.get("feature_mode", "summary21"))
        llm_args = plf._make_llm_args(
            llm_mode="embedding" if int(model_pkg.get("llm_reduce_dim", 0) or 0) > 0 else "none",
            llm_doc_style="features",
            llm_cache_dir=args.llm_cache_dir,
            svd_dim=int(model_pkg.get("svd_dim", args.svd_dim)),
            llm_dual=feature_mode in plf.DUAL_FEATURE_MODES,
        )
        features, bundle = plf.build_llm_case_features(
            input_csv,
            svd_dim=int(model_pkg.get("svd_dim", args.svd_dim)),
            llm_args=llm_args,
        )
        prob = plf.predict_probability_matrix_sklearn(
            model_pkg, features, batch_size=args.pairwise_batch_size
        )
        selected_k = effective_k
        if args.k_selection == "dynamic":
            selected_k = select_dynamic_k_from_similarity(
                prob, requested_k=args.k, window=args.dynamic_k_window,
                merge_threshold=args.dynamic_merge_threshold,
                top_pairs=args.dynamic_top_pairs, label="pairwise-rich",
                policy=args.dynamic_k_policy, gap_min=args.dynamic_gap_min,
                gap_ratio=args.dynamic_gap_ratio,
                start_factor=args.dynamic_start_factor,
                min_factor=args.dynamic_min_factor,
                local_quantile=args.dynamic_local_quantile,
                below_k_margin=args.dynamic_below_k_margin,
            )
        labels = plf.cluster_from_probability(prob, selected_k)
        return labels, bundle.template_count, prob.shape, selected_k

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
    selected_k = effective_k
    if args.k_selection == "dynamic":
        selected_k = select_dynamic_k_from_similarity(
            prob,
            requested_k=args.k,
            window=args.dynamic_k_window,
            merge_threshold=args.dynamic_merge_threshold,
            top_pairs=args.dynamic_top_pairs,
            label="pairwise",
            policy=args.dynamic_k_policy,
            gap_min=args.dynamic_gap_min,
            gap_ratio=args.dynamic_gap_ratio,
            start_factor=args.dynamic_start_factor,
            min_factor=args.dynamic_min_factor,
            local_quantile=args.dynamic_local_quantile,
            below_k_margin=args.dynamic_below_k_margin,
        )
    labels = cluster_precomputed_distance(distance, selected_k)
    return labels, bundle.template_count, prob.shape, selected_k


# ── Completion-based cluster merging ──────────────────────────────────────

def _load_completion_config() -> dict | None:
    """Read completion config from LLM_MODEL_CONFIG."""
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
    section = data.get("completion") or data.get("chat")
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
        "model": str(model), "base_url": str(base_url),
        "api_key": str(api_key or "dummy"),
        "timeout": float(cfg.get("timeout", 30.0)),
        "max_tokens": int(cfg.get("max_tokens", 320)),
    }


def _completion_hash(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}|v2|{prompt}".encode()).hexdigest()[:16]


def _completion_prompt(case_info: dict) -> str:
    """Build compact completion prompt from one case's structured info."""
    parts = []
    for key in ("primary_signature", "primary_type", "mismatch_type", "op_pair",
                "failed_reason", "fatal_file"):
        val = case_info.get(key, "")
        if val:
            parts.append(f"{key}: {val}")
    return (
        "Analyze this RISC-V CPU regression failure. "
        "Return JSON: {\"mechanism\":\"...\",\"trigger\":\"...\","
        "\"confidence\":\"low|medium|high\","
        "\"evidence_tags\":[],\"conflict_tags\":[]}\n\n"
        + "\n".join(parts[:8])
    )


def _parse_completion_json(text: str) -> dict:
    """Extract minimal JSON from completion response."""
    m = re.search(r"\{[\s\S]*\}", str(text))
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return {}


def _completion_cache_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"{cache_key}.json"


def _call_completions(
    prompts: list[tuple[int, str]],  # (cluster_id, prompt)
    config: dict,
    cache_dir: Path,
    runtime_budget_sec: float,
) -> dict[int, dict]:
    """Call completion LLM for selected cluster representatives.
    Returns {cluster_id: parsed_json}.
    """
    from openai import OpenAI
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"],
                    timeout=min(config["timeout"], runtime_budget_sec))
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, dict] = {}

    for cluster_id, prompt in prompts:
        cache_key = _completion_hash(config["model"], prompt)
        cache_path = _completion_cache_path(cache_dir, cache_key)
        if cache_path.is_file():
            try:
                results[cluster_id] = json.loads(cache_path.read_text())
                continue
            except (json.JSONDecodeError, OSError):
                pass
        try:
            resp = client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=config["max_tokens"],
            )
            text = resp.choices[0].message.content or ""
            parsed = _parse_completion_json(text)
            parsed["_raw"] = text[:200]
            parsed["_status"] = "ok"
        except Exception as exc:
            parsed = {"_status": f"error:{type(exc).__name__}", "_error": str(exc)[:200]}
        cache_path.write_text(json.dumps(parsed, ensure_ascii=False))
        results[cluster_id] = parsed

    return results


def refine_labels_with_completion(
    labels: list[int],
    case_infos: list[dict],
    n_cases: int,
    k: int,
    runtime_elapsed: float,
    runtime_limit: float = 30.0,
    completion_cache_dir: Path = Path("/tmp/regr_fail_completion_cache"),
) -> list[int]:
    """Post-process cluster labels using selective completion LLM calls.

    Strategy: pick one representative case per cluster, call completion,
    then merge clusters whose representatives share the same failure
    mechanism and trigger. Time-budget aware with graceful fallback.
    """
    import numpy as np

    # Time budget check
    remaining = runtime_limit - runtime_elapsed
    if remaining < 10:
        info("[completion] insufficient time budget, skipping refinement")
        return labels

    config = _load_completion_config()
    if config is None:
        info("[completion] no valid completion config, skipping")
        return labels

    # Pick one representative per cluster (case closest to centroid would be ideal,
    # but here we use the case with the most informative structured info)
    cluster_to_idx: dict[int, int] = {}
    for label in set(labels):
        members = [i for i, l in enumerate(labels) if l == label]
        if not members:
            continue
        # Pick member with richest info
        best = max(members, key=lambda i: sum(
            1 for k in ("primary_type", "mismatch_type", "op_pair", "failed_reason")
            if case_infos[i].get(k, "")
        ) if i < len(case_infos) else 0)
        cluster_to_idx[label] = best

    n_clusters = len(cluster_to_idx)
    # Limit reps: max 5 for >30 cases, all for ≤30 cases
    max_rep_calls = n_clusters if n_cases <= 30 else min(5, n_clusters)
    if max_rep_calls < n_clusters:
        # Pick clusters with fewest members first (small fragments need most help)
        cluster_sizes = {l: sum(1 for x in labels if x == l) for l in cluster_to_idx}
        sorted_clusters = sorted(cluster_to_idx.keys(), key=lambda l: cluster_sizes.get(l, 999))
        selected_clusters = sorted_clusters[:max_rep_calls]
    else:
        selected_clusters = list(cluster_to_idx.keys())

    info(f"[completion] calling {len(selected_clusters)}/{n_clusters} cluster reps (budget={remaining:.0f}s)")

    # Build prompts
    prompts = [(label, _completion_prompt(
        case_infos[cluster_to_idx[label]] if cluster_to_idx[label] < len(case_infos) else {}
    )) for label in selected_clusters]

    # Time budget per call: each call needs ~30s minimum
    per_call_budget = min(config["timeout"], max(20, (remaining - 5) / max(1, len(prompts))))
    config["timeout"] = per_call_budget

    # Call completions (sequential for simplicity)
    try:
        results = _call_completions(prompts, config, completion_cache_dir, remaining - 5)
    except Exception as exc:
        info(f"[completion] API calls failed ({exc}), returning base labels")
        return labels

    ok_count = sum(1 for v in results.values() if v.get("_status") == "ok")
    info(f"[completion] {ok_count}/{len(results)} cluster reps OK")

    if ok_count == 0:
        return labels

    # Build cluster signatures from completion results
    cluster_sig: dict[int, tuple[str, str]] = {}
    for label, data in results.items():
        mech = str(data.get("mechanism", "") or "").strip().lower()
        trig = str(data.get("trigger", "") or "").strip().lower()
        if mech or trig:
            cluster_sig[label] = (mech, trig)

    if len(cluster_sig) < 2:
        return labels

    # Merge clusters with same (mechanism, trigger)
    merge_map: dict[int, int] = {}
    seen_sigs: dict[tuple[str, str], int] = {}

    for label in sorted(cluster_sig.keys()):
        sig = cluster_sig[label]
        if sig in seen_sigs and sig != ("unknown", "unknown"):
            merge_map[label] = seen_sigs[sig]
        else:
            seen_sigs[sig] = label

    if merge_map:
        info(f"[completion] merging {len(merge_map)} clusters: {dict(merge_map)}")
        new_labels = []
        for l in labels:
            new_labels.append(merge_map.get(l, l))
        # Compact label IDs
        unique = sorted(set(new_labels))
        remap = {old: new for new, old in enumerate(unique)}
        return [remap[l] for l in new_labels]

    return labels


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
    parser.add_argument("--svd-dim", type=int, default=64, help="SVD dimension before agglomerative/HDBSCAN")
    parser.add_argument(
        "--feature-level",
        choices=("baseline", "structured"),
        default="baseline",
        help="feature extraction level; structured adds deterministic sim/regr failure semantics",
    )
    parser.add_argument(
        "--normalizer",
        choices=("v1", "semantic"),
        default="v1",
        help="log-line normalizer before Drain; semantic keeps stable failure slots and masks volatile details",
    )
    parser.add_argument(
        "--line-mode",
        choices=("default", "signal_window"),
        default="default",
        help="log line selection strategy before Drain",
    )
    parser.add_argument(
        "--template-weighting",
        choices=("none", "quality"),
        default="quality",
        help="quality-aware weighting/drop rules for Drain templates",
    )
    parser.add_argument("--token-weights", type=Path, help="optional token_weights.json learned from training data")
    parser.add_argument("--token-weight-mode", choices=("repeat", "none"), default="none")
    parser.add_argument("--cluster-factor", type=float, default=0.875, help="multiply requested k before clustering")
    parser.add_argument(
        "--k-selection",
        choices=("factor", "fixed", "dynamic"),
        default="factor",
        help="bucket-count policy: factor keeps legacy k*cluster_factor, fixed uses k, dynamic stops merges when the best cross-cluster score is low",
    )
    parser.add_argument("--dynamic-k-window", type=int, default=2, help="absolute +/- window around k for --k-selection dynamic")
    parser.add_argument("--dynamic-k-policy", choices=("gap", "threshold", "reference_band"), default="gap", help="dynamic k policy: gap uses within-dataset merge-score cliffs; threshold uses an absolute score cutoff; reference_band starts above k and stops by local score distribution")
    parser.add_argument("--dynamic-merge-threshold", type=float, default=0.95, help="absolute cutoff used by --dynamic-k-policy threshold")
    parser.add_argument("--dynamic-gap-min", type=float, default=0.05, help="minimum absolute drop between adjacent merge scores for --dynamic-k-policy gap/reference_band")
    parser.add_argument("--dynamic-gap-ratio", type=float, default=0.95, help="maximum current/previous score ratio for --dynamic-k-policy gap/reference_band")
    parser.add_argument("--dynamic-start-factor", type=float, default=1.20, help="start reference_band dynamic k from ceil(k*this factor)")
    parser.add_argument("--dynamic-min-factor", type=float, default=0.80, help="do not merge below floor(k*this factor) in reference_band dynamic k")
    parser.add_argument("--dynamic-local-quantile", type=float, default=0.75, help="within-dataset merge-score quantile required for reference_band continuation")
    parser.add_argument("--dynamic-below-k-margin", type=float, default=0.02, help="extra score margin required to continue below k in reference_band dynamic k")
    parser.add_argument("--dynamic-top-pairs", type=int, default=8, help="top pair similarities averaged for each candidate dynamic merge")
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
    parser.add_argument(
        "--llm-mode",
        choices=("none", "embedding", "auto"),
        default="none",
        help="LLM embedding mode: none=disabled, embedding=enabled with fallback, auto=enabled if LLM_MODEL_CONFIG is valid",
    )
    parser.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    parser.add_argument("--llm-batch-size", type=int, default=64)
    parser.add_argument("--llm-timeout-sec", type=float, default=20.0)
    parser.add_argument("--llm-weight", type=float, default=4.0)
    parser.add_argument("--llm-doc-max-features", type=int, default=80)
    parser.add_argument("--llm-fusion", choices=("concat", "similarity"), default="concat")
    parser.add_argument("--llm-alpha", type=float, default=0.75, help="deterministic similarity weight for --llm-fusion similarity")
    parser.add_argument("--llm-doc-style", choices=("features", "summary"), default="features")
    parser.add_argument("--strict-llm", action="store_true", help="fail instead of fallback when LLM embedding fails")
    parser.add_argument("--trace", action="store_true", help="augment case vectors with trace.log.gz features")
    parser.add_argument("--completion", choices=("none", "selective"), default="none",
                        help="Post-cluster refinement via completion LLM")
    parser.add_argument("--completion-cache-dir", type=Path,
                        default=Path("/tmp/regr_fail_completion_cache"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    start = time.perf_counter()
    args = parse_args(argv or sys.argv[1:])
    input_csv = args.input.resolve()
    rows, fieldnames = read_csv_rows(input_csv)
    sim_col = pick_column(fieldnames, "sim")
    regr_col = pick_column(fieldnames, "regr")
    output_cases = output_case_values(rows, fieldnames)

    if not rows:
        write_output(args.output.resolve(), [], [])
        return 0
    if sim_col is None and regr_col is None:
        raise SystemExit("input CSV must contain at least a sim.log or regr.log column")

    effective_k = initial_effective_k(args, len(rows))
    token_weights: Dict[str, float] = {}
    token_weights_label = "None"
    if args.token_weights and args.token_weight_mode == "none":
        warn("--token-weights was provided but --token-weight-mode=none, token weights will be ignored.")
    elif args.token_weights:
        token_weights = load_token_weights(args.token_weights)
        token_weights_label = str(args.token_weights)

    info(
        f"[config] parser={args.parser} cluster={args.cluster} "
        f"cluster_factor={args.cluster_factor} k_selection={args.k_selection} "
        f"feature_level={args.feature_level} "
        f"normalizer={args.normalizer} line_mode={args.line_mode} "
        f"template_weighting={args.template_weighting} "
        f"llm_mode={args.llm_mode} "
        f"token_weight_mode={args.token_weight_mode} "
        f"token_weights={token_weights_label}"
    )
    info("[config] primary_signature=enabled")

    if args.cluster == "pairwise_mlp":
        try:
            labels, template_count, prob_shape, selected_k = run_pairwise_mlp_backend(args, input_csv, effective_k)
            labels = remap_labels(labels)
            info(f"[data] cases={len(rows)} templates={template_count} vector_shape={prob_shape}")
            info(f"[cluster] requested_k={args.k} effective_k={selected_k} method=pairwise_mlp")
            write_output(args.output.resolve(), labels, output_cases)
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

    base_features, normalized_lines = collect_case_inputs(
        input_csv,
        rows,
        sim_col,
        regr_col,
        args.parser,
        feature_level=args.feature_level,
        normalizer=args.normalizer,
        line_mode=args.line_mode,
    )
    feature_counters, template_count = build_feature_counters(
        args,
        base_features,
        normalized_lines,
        token_weights=token_weights,
        token_weight_mode=args.token_weight_mode,
    )
    X, shape, sklearn_input = vectorize_features(feature_counters)
    pre_reduced = False
    use_llm = args.llm_mode == "embedding"
    if args.llm_mode == "auto":
        if load_llm_embedding_config() is not None:
            use_llm = True
            info("[llm] mode=auto detected valid LLM_MODEL_CONFIG; enabling embedding augmentation")
        else:
            info("[llm] mode=auto no valid LLM_MODEL_CONFIG detected; running deterministic")
    if use_llm:
        try:
            if args.llm_fusion == "similarity":
                labels, shape, selected_k = cluster_with_llm_similarity_fusion(X, feature_counters, args, effective_k, sklearn_input)
                labels = remap_labels(labels)
                info(f"[data] cases={len(rows)} templates={template_count} vector_shape={shape}")
                info(
                    f"[cluster] requested_k={args.k} effective_k={selected_k} "
                    f"method=llm_similarity+{args.cluster}"
                )
                write_output(args.output.resolve(), labels, output_cases)
                info(
                    f"[output] buckets={len(set(labels))} path={args.output.resolve()} "
                    f"runtime_sec={time.perf_counter() - start:.3f}"
                )
                return 0
            else:
                X, shape, sklearn_input = augment_with_llm_embeddings(X, feature_counters, args, sklearn_input)
                pre_reduced = True
        except Exception as exc:
            warn(f"LLM embedding augmentation failed ({exc}); falling back to deterministic baseline")
            if args.strict_llm:
                return 3

    # Trace feature augmentation
    if args.trace:
        try:
            X, shape, sklearn_input = augment_with_trace_features(
                X, input_csv, rows, fieldnames, args, sklearn_input)
        except Exception as exc:
            warn(f"trace augmentation failed ({exc}); continuing without trace features")

    if args.k_selection == "dynamic":
        effective_k = select_dynamic_k_for_vectors(
            X,
            args,
            sklearn_input=sklearn_input,
            pre_reduced=pre_reduced,
        )
    info(f"[data] cases={len(rows)} templates={template_count} vector_shape={shape}")
    info(f"[cluster] requested_k={args.k} effective_k={effective_k} method={args.cluster}")
    labels = cluster_vectors(
        X,
        effective_k,
        method=args.cluster,
        svd_dim=args.svd_dim,
        sklearn_input=sklearn_input,
        pre_reduced=pre_reduced,
    )
    labels = remap_labels(labels)

    # Completion-based cluster refinement
    if args.completion == "selective":
        try:
            # Collect structured info for each case from sim/regr logs
            case_infos: list[dict] = _collect_case_infos_for_completion(
                input_csv, rows, sim_col, regr_col)
            runtime_limit = _benchmark_runtime_limit(len(rows))
            refined = refine_labels_with_completion(
                labels, case_infos, len(rows), effective_k,
                runtime_elapsed=time.perf_counter() - start,
                runtime_limit=runtime_limit,
                completion_cache_dir=args.completion_cache_dir,
            )
            if len(set(refined)) != len(set(labels)):
                info(f"[completion] clusters changed: {len(set(labels))} → {len(set(refined))}")
            labels = refined
        except Exception as exc:
            info(f"[completion] refinement failed ({exc}), keeping base labels")

    write_output(args.output.resolve(), labels, output_cases)
    info(f"[output] buckets={len(set(labels))} path={args.output.resolve()} runtime_sec={time.perf_counter() - start:.3f}")
    return 0


def _benchmark_runtime_limit(n_cases: int) -> float:
    """Return runtime limit in seconds based on benchmark size (from Section 3.2)."""
    if n_cases <= 30:
        return 30.0
    if n_cases <= 300:
        return 100.0
    return 300.0


def _collect_case_infos_for_completion(
    input_csv: Path, rows: list[dict], sim_col: str | None, regr_col: str | None,
) -> list[dict]:
    """Collect minimal structured info per case for completion prompts.
    Re-reads sim/regr log snippets for each case (only error lines).
    """
    from pairwise_llm_features import _extract_rich_case_info
    infos: list[dict] = []
    for row in rows:
        info: dict = {}
        for prefix, col in (("sim", sim_col), ("regr", regr_col)):
            if not col:
                continue
            path = resolve_log_path(input_csv, row.get(col))
            text, _ = read_log_sample(path)
            lines = select_lines(text)
            # Keep only error/mismatch/fatal lines
            error_lines = [l for l in lines if re.search(
                r"fatal|error|failed|mismatch|timeout|interrupt|irq|debug|exception|illegal",
                l, re.IGNORECASE)]
            if not error_lines:
                error_lines = lines[-15:]  # fallback to last 15 lines
            if error_lines:
                info[f"{prefix}_errors"] = error_lines[:5]
        # Extract structured info
        sim_lines = info.get("sim_errors", [])
        regr_lines = info.get("regr_errors", [])
        primary_tokens = extract_primary_signature(
            {"path": "", "status": "ok"}, {"path": "", "status": "ok"},
            sim_lines, regr_lines,
        )
        rich = _extract_rich_case_info(sim_lines, regr_lines, primary_tokens)
        infos.append(rich)
    return infos


if __name__ == "__main__":
    raise SystemExit(main())
