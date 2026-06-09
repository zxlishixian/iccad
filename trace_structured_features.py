#!/usr/bin/env python3
"""Structured trace features for pairwise learning — deterministic behavior signals.

Extracts lightweight structured summaries from RISC-V trace.log/.log.gz files
and produces fixed-dimension pairwise feature vectors. No transformer, no LLM.

Modes:
  tail        — last N instructions before end-of-trace
  anchor      — window around sim/regr mismatch PC/opcode
  seq_stats   — global sequence statistics (entropy, diversity, loop)
  tail_anchor — combined tail + anchor
  all         — tail + anchor + seq_stats
"""

from __future__ import annotations

import csv
import gzip
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

# ── RISC-V instruction classification ──────────────────────────────────
LOAD_MNEMONICS = {"lb","lh","lw","lbu","lhu","ld","lwu","c.lw","c.ld","c.lwsp","c.ldsp","flw","fld"}
STORE_MNEMONICS = {"sb","sh","sw","sd","c.sw","c.sd","c.swsp","c.sdsp","fsw","fsd"}
BRANCH_MNEMONICS = {"beq","bne","blt","bge","bltu","bgeu","beqz","bnez","blez","bgez","bltz","bgtz","c.beqz","c.bnez"}
JUMP_MNEMONICS = {"jal","jalr","c.jal","c.jalr","c.j","c.jr","ret","c.ret"}
CSR_MNEMONICS = {"csrrw","csrrs","csrrc","csrrwi","csrrsi","csrrci","csrr","csrw","csrs","csrc","csrwi","csrsi","csrci"}
SYSTEM_MNEMONICS = {"ecall","ebreak","mret","sret","uret","dret","wfi","sfence.vma","fence","fence.i"}
COMPRESSED_PREFIX = "c."

# Line parsing patterns — supports two formats:
# 1. Tab-separated: Time Cycle PC Insn DecodedInstruction RegMemContents
# 2. Space-separated: pc <hex> <opcode> <rest>
TRACE_LINE_RE = re.compile(
    r"(?:pc\s*[:=]\s*)?(?P<pc>[0-9a-fA-F]{6,16})"
    r"\s+(?P<opcode>[a-zA-Z][a-zA-Z0-9_.]*)"
    r"(?:\s+(?P<rest>.*))?",
    re.IGNORECASE,
)
# Tab-separated format: Time\tCycle\tPC\tInsn\tDecoded instruction\tRegMem
TAB_TRACE_RE = re.compile(
    r"^\s*\d+\s+\d+\s+(?P<pc>[0-9a-fA-F]{6,16})\s+[0-9a-fA-F]+\s+(?P<decoded>.+?)(?:\s{2,}.*)?$",
    re.IGNORECASE,
)
# Extract opcode from decoded instruction like "lui x15,0x40001" → "lui"
DECODED_OPCODE_RE = re.compile(r"^\s*([a-zA-Z][a-zA-Z0-9_.]*)\b", re.IGNORECASE)
REG_RE = re.compile(r"\b(?:x(?:[0-2]?\d|3[01])|zero|ra|sp|gp|tp|t[0-6]|s(?:[0-9]|1[01])|fp|a[0-7])\b", re.IGNORECASE)
HEX_RE = re.compile(r"\b(?:0x)?([0-9a-fA-F]{4,16})\b")


@dataclass
class TraceStructuredSummary:
    """Per-case structured trace summary — fixed small memory footprint."""
    case_id: str = ""
    file_status: str = "missing"
    total_lines: int = 0

    # ── Tail features (last TAIL_WINDOW instructions) ──
    tail_opcodes: list[str] = field(default_factory=list)
    tail_opcode_hist: Counter = field(default_factory=Counter)
    tail_loads: int = 0
    tail_stores: int = 0
    tail_branches: int = 0
    tail_csr: int = 0
    tail_system: int = 0
    tail_compressed: int = 0
    tail_pcs: list[str] = field(default_factory=list)
    tail_pc_prefixes: Counter = field(default_factory=Counter)
    tail_regs: list[str] = field(default_factory=list)
    tail_has_loop: bool = False
    tail_loop_length: int = 0
    tail_exception_count: int = 0
    tail_unique_opcodes: int = 0

    # ── Anchor features (around mismatch PC/opcode) ──
    anchor_regr_opcodes: list[str] = field(default_factory=list)
    anchor_regr_pcs: list[str] = field(default_factory=list)
    anchor_sim_opcodes: list[str] = field(default_factory=list)
    anchor_sim_pcs: list[str] = field(default_factory=list)
    anchor_branch_count: int = 0
    anchor_csr_count: int = 0
    anchor_exception_markers: set = field(default_factory=set)

    # ── Sequence statistics ──
    opcode_entropy: float = 0.0
    pc_entropy: float = 0.0
    branch_repeat_ratio: float = 0.0
    loop_density: float = 0.0
    unique_opcode_ratio: float = 0.0
    load_store_ratio: float = 0.0
    compressed_ratio: float = 0.0
    csr_density: float = 0.0
    system_density: float = 0.0
    avg_instructions_per_pc: float = 0.0

    @property
    def missing(self) -> bool:
        return self.file_status != "ok"


# ── Constants ───────────────────────────────────────────────────────────
TAIL_WINDOW = 256
ANCHOR_WINDOW = 64


def _ensure_path(input_csv: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return (input_csv.parent / path).resolve() if not path.is_absolute() else path


def _read_trace_lines(path: Path) -> list[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return [line.rstrip("\n") for line in fh if line.strip()]
    else:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return [line.rstrip("\n") for line in fh if line.strip()]


def _parse_trace_line(line: str) -> dict | None:
    """Parse one trace line into {pc, opcode, rest} or None.

    Supports both tab-separated RISC-V trace format and space-separated format.
    """
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("//"):
        return None

    # Try tab-separated format first (Time Cycle PC Insn DecodedInstruction ...)
    m_tab = TAB_TRACE_RE.match(line)
    if m_tab:
        decoded = m_tab.group("decoded").strip()
        op_match = DECODED_OPCODE_RE.match(decoded)
        opcode = op_match.group(1).lower() if op_match else "unknown"
        return {
            "pc": m_tab.group("pc"),
            "opcode": opcode,
            "rest": decoded,
        }

    # Try space-separated format
    m = TRACE_LINE_RE.match(line)
    if not m:
        return None
    return {
        "pc": m.group("pc"),
        "opcode": m.group("opcode").lower(),
        "rest": (m.group("rest") or "").strip(),
    }


def _classify_opcode(op: str) -> str:
    ol = op.lower()
    if ol in LOAD_MNEMONICS: return "load"
    if ol in STORE_MNEMONICS: return "store"
    if ol in BRANCH_MNEMONICS: return "branch"
    if ol in JUMP_MNEMONICS: return "jump"
    if ol in CSR_MNEMONICS: return "csr"
    if ol in SYSTEM_MNEMONICS: return "system"
    if ol.startswith(COMPRESSED_PREFIX): return "compressed"
    return "arith"


def _coarsen_pc(pc: str, bits: int = 12) -> str:
    """Coarsen PC to region prefix (keep top bits)."""
    try:
        val = int(pc, 16)
        shift = max(0, bits)
        region = val >> shift
        return f"PC_{region:05x}"
    except (ValueError, TypeError):
        return f"PC_{pc[:6]}"


def _detect_loop(opcodes: list[str], min_len: int = 3, max_len: int = 64) -> tuple[bool, int]:
    """Detect if there's a repeated opcode subsequence (tight loop)."""
    if len(opcodes) < min_len * 2:
        return False, 0
    for length in range(min_len, min(max_len, len(opcodes) // 3) + 1):
        pattern = tuple(opcodes[-length:])
        count = 0
        i = len(opcodes) - length
        while i >= length:
            if tuple(opcodes[i - length:i]) == pattern:
                count += 1
                i -= length
            else:
                break
        if count >= 2:
            return True, length
    return False, 0


def _entropy(counter: Counter, total: int | None = None) -> float:
    if total is None:
        total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)


def extract_trace_summary(
    trace_path: Path,
    mode: str = "tail_anchor",
    regr_info: dict | None = None,
    sim_info: dict | None = None,
) -> TraceStructuredSummary:
    """Extract structured trace summary from a trace file.

    Args:
        trace_path: Path to trace.log or trace.log.gz
        mode: tail | anchor | seq_stats | tail_anchor | all
        regr_info: dict with keys like 'mismatch_pc', 'mismatch_opcode', etc.
        sim_info: dict with keys like 'mismatch_pc', 'mismatch_opcode', etc.
    """
    summary = TraceStructuredSummary(case_id="", file_status="missing")

    if trace_path is None or not Path(trace_path).exists():
        return summary

    try:
        lines = _read_trace_lines(Path(trace_path))
    except (OSError, gzip.BadGzipFile):
        summary.file_status = "read_error"
        return summary

    if not lines:
        summary.file_status = "empty"
        return summary

    summary.file_status = "ok"
    summary.total_lines = len(lines)

    parsed = []
    for line in lines:
        p = _parse_trace_line(line)
        if p:
            parsed.append(p)

    if not parsed:
        summary.file_status = "no_valid_lines"
        return summary

    opcodes_all = [p["opcode"] for p in parsed]
    pcs_all = [p["pc"] for p in parsed]

    # ── Tail features ──
    if mode in ("tail", "tail_anchor", "all"):
        tail = parsed[-TAIL_WINDOW:]
        summary.tail_opcodes = [p["opcode"] for p in tail]
        summary.tail_opcode_hist = Counter(summary.tail_opcodes)
        summary.tail_pcs = [p["pc"] for p in tail]
        summary.tail_pc_prefixes = Counter(_coarsen_pc(p["pc"]) for p in tail)
        summary.tail_unique_opcodes = len(summary.tail_opcode_hist)

        for p in tail:
            cls = _classify_opcode(p["opcode"])
            if cls == "load": summary.tail_loads += 1
            elif cls == "store": summary.tail_stores += 1
            elif cls == "branch": summary.tail_branches += 1
            elif cls == "csr": summary.tail_csr += 1
            elif cls == "system": summary.tail_system += 1
            if p["opcode"].startswith(COMPRESSED_PREFIX): summary.tail_compressed += 1
            rest = p.get("rest", "")
            if rest:
                regs = REG_RE.findall(rest)
                summary.tail_regs.extend(regs)

        has_loop, loop_len = _detect_loop(summary.tail_opcodes)
        summary.tail_has_loop = has_loop
        summary.tail_loop_length = loop_len

        # Exception markers in tail
        exception_text = " ".join(p.get("rest", "") for p in tail).lower()
        for marker in ("exception", "trap", "interrupt", "illegal", "fault", "timeout", "fatal"):
            if marker in exception_text:
                summary.tail_exception_count += 1

    # ── Anchor features (around mismatch PC/opcode) ──
    if mode in ("anchor", "tail_anchor", "all"):
        mismatch_pc = None
        mismatch_opcode = None
        if regr_info:
            mismatch_pc = regr_info.get("mismatch_pc") or regr_info.get("pc_region")
            mismatch_opcode = regr_info.get("mismatch_opcode") or regr_info.get("ibex_opcode") or regr_info.get("spike_opcode")
        if not mismatch_pc and sim_info:
            mismatch_pc = sim_info.get("mismatch_pc") or sim_info.get("pc_region")
            mismatch_opcode = sim_info.get("mismatch_opcode") or sim_info.get("ibex_opcode") or sim_info.get("spike_opcode")

        if mismatch_pc is not None or mismatch_opcode is not None:
            # Find anchor point in trace
            anchor_idx = -1
            for i, p in enumerate(parsed):
                if mismatch_pc is not None and p["pc"] == str(mismatch_pc):
                    anchor_idx = i
                    break
                if mismatch_opcode is not None and p["opcode"].lower() == str(mismatch_opcode).lower():
                    anchor_idx = i
                    break

            if anchor_idx < 0:
                # Fallback: search near the end
                anchor_idx = max(0, len(parsed) - ANCHOR_WINDOW)

            start = max(0, anchor_idx - ANCHOR_WINDOW // 2)
            end = min(len(parsed), anchor_idx + ANCHOR_WINDOW // 2)
            anchor = parsed[start:end]

            summary.anchor_regr_opcodes = [p["opcode"] for p in anchor]
            summary.anchor_regr_pcs = [p["pc"] for p in anchor]
            summary.anchor_branch_count = sum(1 for p in anchor if _classify_opcode(p["opcode"]) in ("branch", "jump"))
            summary.anchor_csr_count = sum(1 for p in anchor if _classify_opcode(p["opcode"]) == "csr")
            for p in anchor:
                rest = p.get("rest", "").lower()
                for marker in ("exception", "trap", "interrupt", "illegal", "fault"):
                    if marker in rest:
                        summary.anchor_exception_markers.add(marker)

    # ── Sequence statistics ──
    if mode in ("seq_stats", "tail_anchor", "all"):
        opcode_counter = Counter(opcodes_all)
        pc_counter = Counter(pcs_all)
        total = len(opcodes_all)

        summary.opcode_entropy = _entropy(opcode_counter, total)
        summary.pc_entropy = _entropy(pc_counter, total)
        summary.unique_opcode_ratio = len(opcode_counter) / max(1, total)
        summary.avg_instructions_per_pc = total / max(1, len(pc_counter))

        # Load/store ratio
        loads = sum(opcode_counter.get(op, 0) for op in LOAD_MNEMONICS)
        stores = sum(opcode_counter.get(op, 0) for op in STORE_MNEMONICS)
        summary.load_store_ratio = loads / max(1, stores)

        # Compressed ratio
        compressed = sum(1 for op in opcodes_all if op.startswith(COMPRESSED_PREFIX))
        summary.compressed_ratio = compressed / max(1, total)

        # CSR/system density
        csr_count = sum(opcode_counter.get(op, 0) for op in CSR_MNEMONICS)
        system_count = sum(opcode_counter.get(op, 0) for op in SYSTEM_MNEMONICS)
        summary.csr_density = csr_count / max(1, total)
        summary.system_density = system_count / max(1, total)

        # Branch repetition (how often same branch repeats)
        branches = [op for op in opcodes_all if _classify_opcode(op) in ("branch", "jump")]
        if len(branches) > 1:
            repeats = sum(1 for i in range(1, len(branches)) if branches[i] == branches[i - 1])
            summary.branch_repeat_ratio = repeats / max(1, len(branches) - 1)

        # Loop density (fraction of instructions in repeated patterns)
        loop_ops = 0
        for i, op in enumerate(opcodes_all[:-5]):
            if opcodes_all[i:i + 5] == opcodes_all[i + 5:i + 10]:
                loop_ops += 5
        summary.loop_density = loop_ops / max(1, total)

    return summary


# ── Pairwise feature builders ───────────────────────────────────────────

def _safe_mean(values) -> float:
    return float(np.mean(values)) if values else 0.0


def _hist_cosine(a: Counter, b: Counter) -> float:
    all_keys = set(a.keys()) | set(b.keys())
    va = np.array([a.get(k, 0) for k in all_keys], dtype=np.float64)
    vb = np.array([b.get(k, 0) for k in all_keys], dtype=np.float64)
    dot = float(np.dot(va, vb))
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    return dot / max(na * nb, 1e-12)


def _hist_jaccard(a: Counter, b: Counter) -> float:
    keys_a = set(a.keys())
    keys_b = set(b.keys())
    u = len(keys_a | keys_b)
    return len(keys_a & keys_b) / u if u else 1.0


def trace_tail_pair_features(a: TraceStructuredSummary, b: TraceStructuredSummary) -> np.ndarray:
    """Pairwise features from tail trace summaries (~20 dims)."""
    if a.missing or b.missing:
        return np.zeros(20, dtype=np.float32)

    # Opcode similarity
    opcode_cos = _hist_cosine(a.tail_opcode_hist, b.tail_opcode_hist)
    opcode_jac = _hist_jaccard(a.tail_opcode_hist, b.tail_opcode_hist)

    # PC region similarity
    pc_jac = _hist_jaccard(a.tail_pc_prefixes, b.tail_pc_prefixes)

    # Behavior ratios
    denom_a = max(1, len(a.tail_opcodes))
    denom_b = max(1, len(b.tail_opcodes))
    load_ratio_diff = abs(a.tail_loads / denom_a - b.tail_loads / denom_b)
    store_ratio_diff = abs(a.tail_stores / denom_a - b.tail_stores / denom_b)
    branch_ratio_diff = abs(a.tail_branches / denom_a - b.tail_branches / denom_b)
    csr_ratio_diff = abs(a.tail_csr / denom_a - b.tail_csr / denom_b)
    system_ratio_diff = abs(a.tail_system / denom_a - b.tail_system / denom_b)

    # Loop
    both_have_loop = float(a.tail_has_loop and b.tail_has_loop)
    loop_len_ratio = min(a.tail_loop_length, b.tail_loop_length) / max(1, max(a.tail_loop_length, b.tail_loop_length)) if a.tail_loop_length and b.tail_loop_length else 0.0

    # Unique opcodes
    unique_ratio_diff = abs(a.tail_unique_opcodes / denom_a - b.tail_unique_opcodes / denom_b)

    # Exception
    both_have_exception = float(a.tail_exception_count > 0 and b.tail_exception_count > 0)
    exception_diff = abs(a.tail_exception_count - b.tail_exception_count) / max(1, max(a.tail_exception_count, b.tail_exception_count)) if (a.tail_exception_count or b.tail_exception_count) else 0.0

    # Compressed ratio
    comp_a = a.tail_compressed / denom_a
    comp_b = b.tail_compressed / denom_b
    comp_ratio_diff = abs(comp_a - comp_b)

    return np.asarray([
        opcode_cos,          # 0: tail opcode histogram cosine
        opcode_jac,          # 1: tail opcode jaccard
        pc_jac,              # 2: tail PC region jaccard
        load_ratio_diff,     # 3: load ratio difference
        store_ratio_diff,    # 4: store ratio difference
        branch_ratio_diff,   # 5: branch ratio difference
        csr_ratio_diff,      # 6: CSR ratio difference
        system_ratio_diff,   # 7: system ratio difference
        both_have_loop,      # 8: both have tight loop
        loop_len_ratio,      # 9: loop length similarity
        unique_ratio_diff,   # 10: unique opcode ratio diff
        both_have_exception, # 11: both have exception markers
        exception_diff,      # 12: exception count diff (normalized)
        comp_ratio_diff,     # 13: compressed ratio diff
        float(a.missing),    # 14: a missing flag
        float(b.missing),    # 15: b missing flag
        float(a.missing or b.missing),  # 16: any missing
        float(a.missing != b.missing),  # 17: xor missing
        0.0,                 # 18: reserved
        0.0,                 # 19: reserved
    ], dtype=np.float32)


def trace_anchor_pair_features(a: TraceStructuredSummary, b: TraceStructuredSummary) -> np.ndarray:
    """Pairwise features from anchor window summaries (~16 dims)."""
    if a.missing or b.missing:
        return np.zeros(16, dtype=np.float32)

    # Anchor opcode similarity
    anchor_a_hist = Counter(a.anchor_regr_opcodes)
    anchor_b_hist = Counter(b.anchor_regr_opcodes)
    anchor_opcode_cos = _hist_cosine(anchor_a_hist, anchor_b_hist)
    anchor_opcode_jac = _hist_jaccard(anchor_a_hist, anchor_b_hist)

    # Anchor PC similarity
    anchor_pc_a = Counter(_coarsen_pc(pc) for pc in a.anchor_regr_pcs)
    anchor_pc_b = Counter(_coarsen_pc(pc) for pc in b.anchor_regr_pcs)
    anchor_pc_jac = _hist_jaccard(anchor_pc_a, anchor_pc_b)

    # Branch/CSR similarity in anchor
    denom_a = max(1, len(a.anchor_regr_opcodes))
    denom_b = max(1, len(b.anchor_regr_opcodes))
    branch_diff = abs(a.anchor_branch_count / denom_a - b.anchor_branch_count / denom_b)
    csr_diff = abs(a.anchor_csr_count / denom_a - b.anchor_csr_count / denom_b)

    # Exception overlap
    exception_jac = float(len(a.anchor_exception_markers & b.anchor_exception_markers)) / max(1, len(a.anchor_exception_markers | b.anchor_exception_markers))
    any_exception_a = float(len(a.anchor_exception_markers) > 0)
    any_exception_b = float(len(b.anchor_exception_markers) > 0)

    return np.asarray([
        anchor_opcode_cos,       # 0: anchor opcode cosine
        anchor_opcode_jac,       # 1: anchor opcode jaccard
        anchor_pc_jac,           # 2: anchor PC jaccard
        branch_diff,             # 3: branch count diff
        csr_diff,                # 4: CSR count diff
        exception_jac,           # 5: exception marker jaccard
        any_exception_a,         # 6: a has exception
        any_exception_b,         # 7: b has exception
        float(a.missing),        # 8
        float(b.missing),        # 9
        float(a.missing or b.missing),  # 10
        float(a.missing != b.missing),  # 11
        0.0, 0.0, 0.0, 0.0,     # 12-15: reserved
    ], dtype=np.float32)


def trace_seq_stats_pair_features(a: TraceStructuredSummary, b: TraceStructuredSummary) -> np.ndarray:
    """Pairwise features from sequence statistics (~16 dims)."""
    if a.missing or b.missing:
        return np.zeros(16, dtype=np.float32)

    return np.asarray([
        abs(a.opcode_entropy - b.opcode_entropy),       # 0
        abs(a.pc_entropy - b.pc_entropy),                # 1
        abs(a.branch_repeat_ratio - b.branch_repeat_ratio),  # 2
        abs(a.loop_density - b.loop_density),            # 3
        abs(a.unique_opcode_ratio - b.unique_opcode_ratio),  # 4
        abs(math.log1p(a.load_store_ratio) - math.log1p(b.load_store_ratio)),  # 5
        abs(a.compressed_ratio - b.compressed_ratio),    # 6
        abs(a.csr_density - b.csr_density),              # 7
        abs(a.system_density - b.system_density),        # 8
        abs(a.avg_instructions_per_pc - b.avg_instructions_per_pc),  # 9
        float(a.missing),        # 10
        float(b.missing),        # 11
        float(a.missing or b.missing),  # 12
        float(a.missing != b.missing),  # 13
        0.0, 0.0,                # 14-15: reserved
    ], dtype=np.float32)


def build_trace_structured_pair_features(
    a_summary: TraceStructuredSummary | None,
    b_summary: TraceStructuredSummary | None,
    trace_mode: str = "tail_anchor",
) -> np.ndarray:
    """Build combined trace structured pairwise features from summaries."""
    if a_summary is None:
        a_summary = TraceStructuredSummary()
    if b_summary is None:
        b_summary = TraceStructuredSummary()

    blocks = []
    if trace_mode in ("tail", "tail_anchor", "all"):
        blocks.append(trace_tail_pair_features(a_summary, b_summary))
    if trace_mode in ("anchor", "tail_anchor", "all"):
        blocks.append(trace_anchor_pair_features(a_summary, b_summary))
    if trace_mode in ("seq_stats", "all"):
        blocks.append(trace_seq_stats_pair_features(a_summary, b_summary))

    if not blocks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(blocks).astype(np.float32, copy=False)


def trace_structured_pair_dim(trace_mode: str) -> int:
    """Return the fixed dimension of trace structured pair features."""
    dims = {
        "tail": 20,
        "anchor": 16,
        "seq_stats": 16,
        "tail_anchor": 36,  # 20 + 16
        "all": 52,          # 20 + 16 + 16
    }
    return dims.get(trace_mode, 0)


# ── Bulk extraction ──────────────────────────────────────────────────────

def pick_trace_column(fields: Sequence[str]) -> str | None:
    candidates = ("trace_log", "trace log", "trace", "trace_path", "tracefile")
    by_key = {"".join(ch for ch in f.lower() if ch.isalnum()): f for f in fields}
    for cand in candidates:
        key = "".join(ch for ch in cand.lower() if ch.isalnum())
        if key in by_key:
            return by_key[key]
    for f in fields:
        if "trace" in "".join(ch for ch in f.lower() if ch.isalnum()):
            return f
    return None


def extract_trace_summaries_for_dataset(
    input_csv: Path,
    trace_mode: str = "tail_anchor",
    rich_infos: Sequence[dict] | None = None,
) -> dict[str, TraceStructuredSummary]:
    """Extract trace summaries for all cases in a dataset.

    Returns dict mapping case_id → TraceStructuredSummary.
    Missing traces get a 'missing' summary with all zeros.
    """
    import regr_fail_bucketing as rfb
    rows, fields = rfb.read_csv_rows(input_csv)
    trace_col = pick_trace_column(fields)

    result: dict[str, TraceStructuredSummary] = {}
    for idx, row in enumerate(rows):
        # Case ID (same logic as rfb.case_id_from_row)
        sim_col = rfb.pick_column(fields, "sim")
        regr_col = rfb.pick_column(fields, "regr")
        used_cols = [c for c in (sim_col, regr_col) if c]
        case_id = rfb.case_id_from_row(row, idx, used_cols)

        regr_info = rich_infos[idx] if rich_infos and idx < len(rich_infos) else None

        if not trace_col:
            result[case_id] = TraceStructuredSummary(case_id=case_id, file_status="no_trace_column")
            continue

        path = _ensure_path(input_csv, row.get(trace_col))
        if path is None or not path.exists():
            result[case_id] = TraceStructuredSummary(case_id=case_id, file_status="missing_file")
            continue

        summary = extract_trace_summary(path, mode=trace_mode, regr_info=regr_info)
        summary.case_id = case_id
        result[case_id] = summary

    return result
