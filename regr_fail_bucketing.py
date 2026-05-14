#!/usr/bin/env python3
"""Baseline regression-failure bucketing.

Pipeline:
  input.csv -> sim.log/regr.log -> Drain-like templates -> sparse TF-IDF
  -> cosine k-means -> output.csv

The implementation intentionally uses only Python's standard library so it can
run in a bare evaluation environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


HASH_DIM = 1 << 15
MAX_LOG_BYTES = 256 * 1024
MAX_SELECTED_LINES = 420
KMEANS_ITERS = 14
RANDOM_SEED = 20260212

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
REG_RE = re.compile(r"\bx(?:[12]?\d|3[01])\b", re.IGNORECASE)
LINE_NUM_RE = re.compile(r"^\s*(?:\[E\]\s*)?\d+:\s*")


def stable_hash(text: str, modulo: int = HASH_DIM) -> int:
    digest = hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def read_csv_rows(input_csv: Path) -> Tuple[List[dict], List[str]]:
    with input_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        return [], fieldnames
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


def template_line(line: str) -> str:
    s = line.strip().lower()
    s = LINE_NUM_RE.sub("", s)
    s = PATH_RE.sub("<path>", s)
    s = CASE_RE.sub("<case>", s)
    s = SEED_RE.sub("<seed>", s)
    s = TIME_RE.sub("@ <time>", s)
    s = HEX_RE.sub("<hex>", s)
    s = DEC_RE.sub("<num>", s)
    s = REG_RE.sub("<reg>", s)
    s = re.sub(r"\b\d+\b", "<n>", s)
    s = re.sub(r"['\"]", "", s)
    s = re.sub(r"[^a-z0-9_+./:<>\[\]-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def line_tokens(template: str) -> Iterable[str]:
    for token in re.findall(r"[a-z_][a-z0-9_./:+-]*|<[^>]+>|\[[a-z]\]", template):
        if len(token) > 1:
            yield token


def extract_status_features(prefix: str, text: str, feats: Counter) -> None:
    lower = text.lower()
    for key in ("uvm_fatal", "uvm_error", "uvm_warning", "mismatch", "failed", "passed", "timeout"):
        count = lower.count(key)
        if count:
            feats[f"{prefix}:count:{key}"] += min(count, 20)

    patterns = [
        ("reg_write_mismatch", r"register write data mismatch to\s+x\d+"),
        ("pc_mismatch", r"\bpc mismatch\b"),
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


def features_for_case(input_csv: Path, row: dict, idx: int, sim_col: str | None, regr_col: str | None) -> Counter:
    feats: Counter = Counter()
    used_cols = [c for c in (sim_col, regr_col) if c]
    cid = case_id_from_row(row, idx, used_cols)
    feats[f"case_shape:{re.sub(r'\\d', '0', cid)}"] += 1

    for prefix, col in (("sim", sim_col), ("regr", regr_col)):
        path = resolve_log_path(input_csv, row.get(col) if col else None)
        text, status = read_log_sample(path)
        feats[f"{prefix}:file_status:{status}"] += 3
        if path is not None:
            feats[f"{prefix}:basename:{path.name.lower()}"] += 1
        extract_status_features(prefix, text, feats)

        for line in select_lines(text):
            tmpl = template_line(line)
            if not tmpl:
                continue
            feats[f"{prefix}:tmpl:{tmpl}"] += 2
            toks = list(line_tokens(tmpl))
            for tok in toks:
                feats[f"{prefix}:tok:{tok}"] += 1
            for a, b in zip(toks, toks[1:]):
                feats[f"{prefix}:bi:{a}_{b}"] += 1
    return feats


def hashed_tfidf(cases: Sequence[Counter]) -> List[Dict[int, float]]:
    docs: List[Dict[int, float]] = []
    df: Dict[int, int] = defaultdict(int)

    for feats in cases:
        hashed: Dict[int, float] = defaultdict(float)
        for feat, cnt in feats.items():
            idx = stable_hash(feat)
            hashed[idx] += 1.0 + math.log1p(float(cnt))
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


def dot(a: Dict[int, float], b: Dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(i, 0.0) for i, v in a.items())


def normalize_dense_sparse(vec: Dict[int, float]) -> Dict[int, float]:
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm <= 0.0:
        return {}
    return {i: v / norm for i, v in vec.items() if v}


def choose_initial_centers(vectors: Sequence[Dict[int, float]], k: int) -> List[Dict[int, float]]:
    n = len(vectors)
    if k >= n:
        return [dict(v) for v in vectors]
    centers = [dict(vectors[0])]
    min_dist = [1.0 - max(0.0, dot(v, centers[0])) for v in vectors]
    while len(centers) < k:
        candidate = max(range(n), key=lambda i: (min_dist[i], -i))
        centers.append(dict(vectors[candidate]))
        for i, v in enumerate(vectors):
            d = 1.0 - max(0.0, dot(v, centers[-1]))
            if d < min_dist[i]:
                min_dist[i] = d
    return centers


def cluster_vectors(vectors: Sequence[Dict[int, float]], k: int) -> List[int]:
    n = len(vectors)
    if n == 0:
        return []
    k = max(1, min(k, n))
    if k == n:
        return list(range(n))

    centers = choose_initial_centers(vectors, k)
    labels = [-1] * n
    rng = random.Random(RANDOM_SEED)

    for _ in range(KMEANS_ITERS):
        changed = False
        for i, vec in enumerate(vectors):
            best = max(range(k), key=lambda c: (dot(vec, centers[c]), -c))
            if labels[i] != best:
                labels[i] = best
                changed = True

        sums: List[Dict[int, float]] = [defaultdict(float) for _ in range(k)]
        counts = [0] * k
        for label, vec in zip(labels, vectors):
            counts[label] += 1
            acc = sums[label]
            for idx, val in vec.items():
                acc[idx] += val

        for c in range(k):
            if counts[c] == 0:
                replacement = rng.randrange(n)
                centers[c] = dict(vectors[replacement])
            else:
                centers[c] = normalize_dense_sparse(sums[c])

        if not changed:
            break
    return labels


def write_output(path: Path, labels: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bucket"])
        for label in labels:
            writer.writerow([f"bucket_{label:03d}"])


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bucket Ibex regression failures.")
    parser.add_argument("--input", required=True, type=Path, help="Input CSV containing log paths.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV to write buckets.")
    parser.add_argument("--k", required=True, type=int, help="Number of requested buckets.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
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

    feature_counters = [
        features_for_case(input_csv, row, idx, sim_col, regr_col)
        for idx, row in enumerate(rows)
    ]
    vectors = hashed_tfidf(feature_counters)
    labels = cluster_vectors(vectors, args.k)
    write_output(args.output.resolve(), labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
