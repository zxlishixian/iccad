#!/usr/bin/env python3
"""Failure-signature extraction helpers (torch-free, shared by training + inference)."""
from __future__ import annotations

import re
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

import official_style_features as osf

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





def extract_sim_failure_message(sim_text: str) -> str:
    """Extract the specific UVM failure message from a sim.log (semantic, not a rule).

    Returns the UVM_FATAL/UVM_ERROR lines (stripped of file:line/time noise) which
    carry the *why* of the failure (e.g. 'Did not receive core_status HANDLING_IRQ'
    vs 'mstatus.mpie was not set').  Falls back to the last non-boilerplate line.
    """
    lines = [ln.strip() for ln in sim_text.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        if "UVM_FATAL" in ln or "UVM_ERROR" in ln:
            # strip 'file.sv(N) @ time:' prefix noise, keep the message core
            core = re.sub(r"^UVM_(FATAL|ERROR)\s+\S+\.sv\(\d+\)\s*@\s*\d+\s*:\s*\S+\s*\[\S+\]\s*", "", ln)
            out.append(core)
            if len(out) >= 2:
                break
    if out:
        return " | ".join(out)
    # fallback: any line with a test/error keyword
    for ln in lines:
        if re.search(r"\[FAILED\]|FAILED|error|Error|timeout|Timeout", ln):
            return ln
    return lines[-1] if lines else ""


def extract_sim_failure_messages(dataset: Path, max_chars: int = 400) -> list[str]:
    """Return per-case sim.log failure message strings (input.csv order)."""
    import gzip
    cases = osf.read_cases(dataset / "input.csv")
    out = []
    for case_id in cases:
        r = _find_regr(dataset, case_id)
        # sim.log is next to regr.log
        sim = r.parent / "sim.log" if r else None
        txt = ""
        if sim and sim.exists():
            txt = sim.read_text(errors="ignore")
        elif sim and (sim.parent / "sim.log.gz").exists():
            txt = gzip.decompress((sim.parent / "sim.log.gz").read_bytes()).decode(errors="ignore")
        msg = extract_sim_failure_message(txt)
        out.append(msg[:max_chars])
    return out
