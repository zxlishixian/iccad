#!/usr/bin/env python3
"""Trace sequence tokenization for self-supervised Transformer pretraining.

Parses RISC-V trace.log / trace.log.gz files into structured token sequences
suitable for Masked Opcode Modeling (MOM) pretraining of a lightweight
TraceTransformerEncoder.
"""

from __future__ import annotations

import csv
import gzip
import re
from collections import Counter, deque
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# Opcode classification
# ---------------------------------------------------------------------------
LOAD_OPS = {"lb", "lh", "lw", "lbu", "lhu", "ld", "c.lw", "c.ld", "c.lwsp", "c.ldsp"}
STORE_OPS = {"sb", "sh", "sw", "sd", "c.sw", "c.sd", "c.swsp", "c.sdsp"}
BRANCH_OPS = {
    "beq", "bne", "blt", "bge", "bltu", "bgeu",
    "j", "jal", "jalr",
    "c.j", "c.jal", "c.jr", "c.jalr", "c.beqz", "c.bnez",
    "ret",
}
CSR_PREFIX = "csr"
SYSTEM_OPS = {"mret", "dret", "wfi", "ecall", "ebreak", "fence", "sfence.vma"}

# registers visible in decoded-instruction text
REG_RE = re.compile(
    r"\b(x(?:[0-2]?\d|3[01])|zero|ra|sp|gp|tp|t[0-6]|s(?:[0-9]|1[01])|a[0-7]|fp)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Special tokens (reserved ids 0-5)
# ---------------------------------------------------------------------------
PAD_TOKEN = "PAD"           # id 0
UNK_TOKEN = "UNK"           # id 1
MASK_TOKEN = "MASK"         # id 2
CLS_TOKEN = "CLS"           # id 3
TRACE_MISSING_TOKEN = "TRACE_MISSING"  # id 4
EMPTY_TRACE_TOKEN = "EMPTY_TRACE"      # id 5

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, CLS_TOKEN, TRACE_MISSING_TOKEN, EMPTY_TRACE_TOKEN]

TRACE_COLUMN_CANDIDATES = (
    "trace_log", "trace log", "trace", "trace_path", "tracefile", "trace_file",
    "rtl_trace", "trace core", "trace_core",
)


def _classify_opcode(opcode: str) -> str:
    """Map an opcode to exactly one functional-category token."""
    if opcode in LOAD_OPS:
        return "CLASS_LOAD"
    if opcode.startswith("l") and opcode not in {"lui", "la", "li"}:
        return "CLASS_LOAD"
    if opcode in STORE_OPS:
        return "CLASS_STORE"
    if opcode.startswith("s") and opcode not in {
        "sll", "slt", "sltu", "srl", "sra", "sub",
        "slli", "srli", "srai", "slti", "sltiu",
    }:
        return "CLASS_STORE"
    if opcode.startswith(CSR_PREFIX):
        return "CLASS_CSR"
    if opcode in SYSTEM_OPS:
        return "CLASS_SYSTEM"
    if opcode in BRANCH_OPS:
        return "CLASS_BRANCH"
    if opcode.startswith("c."):
        return "CLASS_COMPRESSED"
    return "CLASS_ARITH"


def _coarsen_pc(pc: str, prefix_digits: int = 4) -> str:
    """Coarsen a PC hex string to a prefix region token."""
    pc = pc.lower().removeprefix("0x")
    if len(pc) < 3:
        return f"PCREG_0x{pc}"
    return f"PCREG_0x{pc[:min(prefix_digits, len(pc))]}"


def _is_destination_first(opcode: str) -> bool:
    """Return True when the first register in the decoded string is RD."""
    if opcode in STORE_OPS:
        return False
    if opcode.startswith("s") and opcode not in {
        "sll", "slt", "sltu", "srl", "sra", "sub",
        "slli", "srli", "srai", "slti", "sltiu",
    }:
        return False
    if opcode in BRANCH_OPS:
        return False
    if opcode.startswith("b") and opcode not in {"break"}:
        return False
    return True


def _parse_trace_line(line: str) -> dict | None:
    """Parse a single trace line into structured fields.

    Returns dict with keys: opcode, pc, class, semantic_flags, regs, rd, rs
    or None if the line is a header / unparseable.
    """
    line = line.strip()
    if not line or line.lower().startswith("time\tcycle"):
        return None

    parts = [p.strip() for p in line.split("\t")]
    opcode = ""
    pc = ""
    decoded = ""

    if len(parts) >= 5:
        pc = parts[2].lower().removeprefix("0x")
        # Column 4 = opcode, column 5 (if present) = operands
        decoded = parts[4].strip()
        if len(parts) >= 6 and parts[5].strip():
            decoded = f"{decoded} {parts[5].strip()}"
        opcode = decoded.split()[0].lower().rstrip(":")
    else:
        # fallback: heuristics on free-form text
        hexes = re.findall(r"\b(?:0x)?([0-9a-fA-F]{6,16})\b", line)
        if len(hexes) >= 3:
            pc = hexes[2].lower()
        elif hexes:
            pc = hexes[0].lower()
        match = re.search(r"\b([a-z][a-z0-9_.]*)\b", line.lower())
        if match and match.group(1) not in {"time", "cycle", "pc", "insn", "decoded"}:
            opcode = match.group(1)

    if not opcode:
        return None

    opcode = opcode.strip().lower().removesuffix(":")
    cls = _classify_opcode(opcode)

    # semantic flags
    flags: list[str] = []
    if cls == "CLASS_LOAD":
        flags.append("HAS_LOAD")
    if cls == "CLASS_STORE":
        flags.append("HAS_STORE")
    if cls == "CLASS_BRANCH":
        flags.append("BRANCH")
    if cls == "CLASS_CSR":
        flags.append("CSR")
    if cls == "CLASS_SYSTEM":
        flags.append("SYSTEM")

    # registers from decoded instruction text
    regs = [r.lower() for r in REG_RE.findall(decoded)]
    rd: list[str] = []
    rs: list[str] = []
    if regs and _is_destination_first(opcode):
        rd = [regs[0]] if regs[0] != "x0" else []
        rs = [r for r in regs[1:] if r != "x0"]
    elif regs:
        # store / branch: no destination, all are sources
        rs = [r for r in regs if r != "x0"]

    return {
        "opcode": opcode,
        "pc": pc,
        "class": cls,
        "flags": flags,
        "regs": regs,
        "rd": rd,
        "rs": rs,
    }


def parse_trace_to_tokens(
    lines: Sequence[str],
    window_mode: str = "tail",
    window_size: int = 500,
    max_seq_len: int = 1024,
) -> list[str]:
    """Convert raw trace lines into a flat token sequence.

    Parameters
    ----------
    lines : Sequence[str]
        Raw trace file lines.
    window_mode : str
        "tail" for last N lines, "random" for random contiguous window.
    window_size : int
        Number of trace lines to read.
    max_seq_len : int
        Maximum token sequence length (truncation from the end).

    Returns
    -------
    list[str]
        Flat token list. Returns ["EMPTY_TRACE"] if no instructions found,
        ["TRACE_MISSING"] if lines is empty.
    """
    if not lines:
        return [TRACE_MISSING_TOKEN]

    if window_mode == "tail":
        selected = list(lines[-window_size:]) if len(lines) > window_size else list(lines)
    elif window_mode == "random":
        import random
        if len(lines) > window_size:
            start = random.randint(0, len(lines) - window_size)
            selected = list(lines[start:start + window_size])
        else:
            selected = list(lines)
    else:
        raise ValueError(f"unsupported window_mode: {window_mode}")

    tokens: list[str] = []
    for line in selected:
        parsed = _parse_trace_line(line)
        if parsed is None:
            continue
        # opcode token
        tokens.append(f"OP_{parsed['opcode']}")
        # class token
        tokens.append(parsed["class"])
        # PC region
        if parsed["pc"]:
            tokens.append(_coarsen_pc(parsed["pc"]))
        # destination register
        for r in parsed["rd"]:
            tokens.append(f"RD_{r}")
        # source registers
        for r in parsed["rs"]:
            tokens.append(f"RS_{r}")
        # semantic flags
        tokens.extend(parsed["flags"])

    if not tokens:
        return [EMPTY_TRACE_TOKEN]
    return tokens[-max_seq_len:]


def parse_trace_to_token_windows(
    lines: Sequence[str],
    window_size: int = 500,
    stride: int | None = None,
    max_seq_len: int = 1024,
) -> list[list[str]]:
    """Extract multiple token sequences via sliding windows over long traces.

    Parses each line once into tokens, then applies sliding windows over the
    pre-parsed token sequence.  Much faster than re-parsing lines per window.

    Returns one or more token sequences (each truncated to max_seq_len).
    """
    if not lines:
        return [[TRACE_MISSING_TOKEN]]

    if stride is None:
        stride = window_size

    # Parse each instruction line once into a list of tokens
    line_tokens: list[list[str]] = []
    for line in lines:
        if not line.strip() or line.lower().startswith("time\tcycle"):
            continue
        parsed = _parse_trace_line(line)
        if parsed is None:
            continue
        tokens: list[str] = []
        tokens.append(f"OP_{parsed['opcode']}")
        tokens.append(parsed["class"])
        if parsed["pc"]:
            tokens.append(_coarsen_pc(parsed["pc"]))
        for r in parsed["rd"]:
            tokens.append(f"RD_{r}")
        for r in parsed["rs"]:
            tokens.append(f"RS_{r}")
        tokens.extend(parsed["flags"])
        if tokens:
            line_tokens.append(tokens)

    if not line_tokens:
        return [[EMPTY_TRACE_TOKEN]]

    # Flatten to a single token sequence
    all_tokens: list[str] = []
    for lt in line_tokens:
        all_tokens.extend(lt)

    # Estimate tokens per line for window size conversion
    tokens_per_line = len(all_tokens) / max(1, len(line_tokens))
    # Convert line-based window_size to token-based
    token_window = int(window_size * tokens_per_line)
    token_stride = int(stride * tokens_per_line)
    token_window = max(64, min(token_window, max_seq_len))
    token_stride = max(32, token_stride)

    sequences: list[list[str]] = []
    for start in range(0, max(1, len(all_tokens)), token_stride):
        window = all_tokens[start:start + token_window]
        if len(window) < max(10, token_window // 4):
            continue
        sequences.append(window[-max_seq_len:])

    if not sequences:
        return [[EMPTY_TRACE_TOKEN]]
    return sequences


def build_vocabulary(
    token_sequences: Sequence[Sequence[str]],
    min_freq: int = 1,
    max_vocab_size: int = 16384,
) -> dict[str, int]:
    """Build a token→id vocabulary from a corpus of token sequences.

    Special tokens (PAD, UNK, MASK, CLS, etc.) are always assigned ids 0-5.
    Remaining tokens are sorted by descending frequency.
    """
    counter: Counter[str] = Counter()
    for seq in token_sequences:
        counter.update(seq)

    vocab: dict[str, int] = {}
    for i, tok in enumerate(SPECIAL_TOKENS):
        vocab[tok] = i

    # sort by frequency desc, then alphabetically for stability
    sorted_tokens = sorted(
        ((t, c) for t, c in counter.items() if t not in vocab),
        key=lambda x: (-x[1], x[0]),
    )
    available = max_vocab_size - len(vocab)
    for i, (tok, count) in enumerate(sorted_tokens[:available]):
        if count >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def read_trace_file(path: str | Path) -> list[str]:
    """Read trace file lines, handling .gz transparently."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in f]


def _normalize_key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def pick_trace_column(fields: Sequence[str]) -> str | None:
    """Find the trace column name from CSV fieldnames."""
    by_key = {_normalize_key(field): field for field in fields}
    for candidate in TRACE_COLUMN_CANDIDATES:
        key = _normalize_key(candidate)
        if key in by_key:
            return by_key[key]
    for field in fields:
        if "trace" in _normalize_key(field):
            return field
    return None


def _case_id_from_row(row: dict, idx: int, fields: Sequence[str]) -> str:
    wanted = {_normalize_key(name) for name in ("case_id", "case", "id")}
    for field in fields:
        if _normalize_key(field) in wanted and row.get(field):
            return str(row[field])
    return f"case_{idx + 1:06d}"


def _resolve_trace_path(input_csv: Path, value: str | None) -> tuple[Path | None, str]:
    if not value:
        return None, "missing_trace_value"
    path = Path(str(value))
    if not path.is_absolute():
        path = (input_csv.parent / path).resolve()
    if not path.exists():
        return path, "missing_file"
    if not path.is_file():
        return path, "not_file"
    return path, "ok"


def collect_trace_paths_from_input(
    input_csv: str | Path,
) -> list[tuple[str, Path | None, str]]:
    """Collect (case_id, trace_path|None, status) for every row in input.csv."""
    input_csv = Path(input_csv).resolve()
    with input_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    trace_col = pick_trace_column(fields)
    result: list[tuple[str, Path | None, str]] = []
    for idx, row in enumerate(rows):
        case_id = _case_id_from_row(row, idx, fields)
        if not trace_col:
            result.append((case_id, None, "missing_trace_column"))
            continue
        path, status = _resolve_trace_path(input_csv, row.get(trace_col))
        result.append((case_id, path, status))
    return result


def tokenize_traces_for_input(
    input_csv: str | Path,
    window_mode: str = "tail",
    window_size: int = 500,
    max_seq_len: int = 1024,
) -> list[tuple[str, list[str], str]]:
    """Tokenize all trace files referenced by one input CSV.

    Returns list of (case_id, token_list, status).
    """
    collected = collect_trace_paths_from_input(input_csv)
    result: list[tuple[str, list[str], str]] = []
    for case_id, path, status in collected:
        if status != "ok" or path is None:
            result.append((case_id, [TRACE_MISSING_TOKEN], status))
            continue
        try:
            lines = read_trace_file(path)
            tokens = parse_trace_to_tokens(lines, window_mode, window_size, max_seq_len)
            result.append((case_id, tokens, "ok"))
        except OSError:
            result.append((case_id, [TRACE_MISSING_TOKEN], "read_error"))
    return result


def collect_official_trace_paths(
    problem_dir: str | Path = "/home/lishixian/iccad/test_case/problem",
) -> list[Path]:
    """Collect all trace.log.gz paths from the official benchmark directories."""
    problem_dir = Path(problem_dir)
    paths: list[Path] = []
    for bench_dir in sorted(problem_dir.iterdir()):
        if not bench_dir.is_dir():
            continue
        for case_dir in sorted(bench_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            trace_file = case_dir / "trace.log.gz"
            if trace_file.exists():
                paths.append(trace_file)
    return paths


def _tail_lines(path: Path, tail_lines: int) -> list[str]:
    """Read last N lines from a (possibly .gz) trace file efficiently."""
    dq: deque[str] = deque(maxlen=max(1, int(tail_lines)))
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            dq.append(line.rstrip("\n"))
    return list(dq)


def tokenize_trace_tail(
    path: str | Path,
    tail_lines: int = 500,
    max_seq_len: int = 1024,
) -> list[str]:
    """Convenience: read tail of one trace file and tokenize it."""
    lines = _tail_lines(Path(path), tail_lines)
    return parse_trace_to_tokens(lines, window_mode="tail", window_size=tail_lines,
                                 max_seq_len=max_seq_len)
