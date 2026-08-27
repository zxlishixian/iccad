#!/usr/bin/env python3
"""Hierarchical trace features for the experimental Theta TriLog model.

Every available trace is scanned from beginning to end.  The encoder keeps a
compact global execution sketch, relative-time segment summaries, and
sim/regr-conditioned anchor windows.  It never reads gold or meta files.
"""

from __future__ import annotations

import gzip
import hashlib
import math
import pickle
import re
from collections import Counter, deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import official_style_features as osf
import regr_fail_bucketing as rfb
import trace_anchor as ta
import trace_sequence as ts


CACHE_VERSION = 2
TRACE_CLASSES = (
    "CLASS_LOAD",
    "CLASS_STORE",
    "CLASS_BRANCH",
    "CLASS_CSR",
    "CLASS_SYSTEM",
    "CLASS_COMPRESSED",
    "CLASS_ARITH",
)
ANCHOR_SOURCES = ("regr_index", "regr_pc", "sim_time", "opcode", "fallback_tail")
LOCATED_BY = ("pc", "index", "time", "opcode", "tail", "missing", "read_error")
PA_RE = re.compile(r"\bPA\s*:\s*(?:0x)?([0-9a-fA-F]{4,16})\b", re.IGNORECASE)


@dataclass
class HierarchicalTraceFeature:
    case_id: str
    trace_path: str
    file_status: str
    instruction_count: int
    opcode_counts: dict[str, int]
    transition_counts: dict[str, int]
    register_counts: dict[str, int]
    pc_region_counts: dict[str, int]
    class_counts: dict[str, int]
    global_struct: np.ndarray
    anchor_struct: np.ndarray
    global_document: str
    anchor_document: str
    segment_matrix: np.ndarray
    anchor_opcodes: dict[int, tuple[str, ...]]
    anchor_pc_regions: dict[int, tuple[str, ...]]
    anchor_registers: dict[int, tuple[str, ...]]
    anchor_source: str
    located_by: str
    anchor_tags: tuple[str, ...]
    target_pc_present: bool
    target_register_present: bool

    @property
    def has_trace(self) -> bool:
        return self.file_status == "ok" and self.instruction_count > 0


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("rt", encoding="utf-8", errors="ignore")


def _pc_region(pc: str, digits: int = 4) -> str:
    value = str(pc).lower().removeprefix("0x")
    return value[: min(digits, len(value))] if value else ""


def _parse_instruction(line: str) -> dict[str, Any] | None:
    parsed = ts._parse_trace_line(line)
    if parsed is None:
        return None
    time_value, cycle_value, pc, _ = ta._parse_trace_meta(line)
    memory_regions = tuple(_pc_region(value) for value in PA_RE.findall(line))
    return {
        **parsed,
        "time": time_value,
        "cycle": cycle_value,
        "pc_region": _pc_region(pc or parsed.get("pc", "")),
        "memory_regions": memory_regions,
    }


def _stable_index(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dim


def _hashed_counter(counter: Counter[str], dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    total = float(sum(counter.values()))
    if total <= 0.0:
        return out
    for token, count in counter.items():
        out[_stable_index(str(token), dim)] += float(count) / total
    return out


def _hashed_distribution_delta(local: Counter[str], global_counts: dict[str, int], dim: int) -> np.ndarray:
    output = np.zeros(dim, dtype=np.float32)
    local_total = max(1.0, float(sum(local.values())))
    global_total = max(1.0, float(sum(global_counts.values())))
    for token in set(local) | set(global_counts):
        delta = float(local.get(token, 0)) / local_total - float(global_counts.get(token, 0)) / global_total
        output[_stable_index(str(token), dim)] += delta
    return output


def _counter_cosine(a: dict[str, int], b: dict[str, int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(float(a.get(key, 0)) * float(b.get(key, 0)) for key in keys)
    na = math.sqrt(sum(float(value) ** 2 for value in a.values()))
    nb = math.sqrt(sum(float(value) ** 2 for value in b.values()))
    return dot / max(na * nb, 1e-12)


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def _ngrams(values: Sequence[str], size: int) -> set[tuple[str, ...]]:
    return {tuple(values[idx : idx + size]) for idx in range(max(0, len(values) - size + 1))}


@lru_cache(maxsize=None)
def _cached_ngrams(values: tuple[str, ...], size: int) -> frozenset[tuple[str, ...]]:
    return frozenset(tuple(values[idx : idx + size]) for idx in range(max(0, len(values) - size + 1)))


def _lcs_ratio(a: Sequence[str], b: Sequence[str], max_len: int = 256) -> float:
    aa = list(a[-max_len:])
    bb = list(b[-max_len:])
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    previous = [0] * (len(bb) + 1)
    for left in aa:
        current = [0]
        for idx, right in enumerate(bb, 1):
            current.append(previous[idx - 1] + 1 if left == right else max(previous[idx], current[-1]))
        previous = current
    return previous[-1] / max(len(aa), len(bb))


def _ordered_overlap(a: Sequence[str], b: Sequence[str], max_samples: int = 64) -> float:
    """Cheap order-sensitive overlap after relative-position resampling."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    samples = min(max_samples, max(len(a), len(b)))
    if samples <= 1:
        return float(a[-1] == b[-1])
    matches = 0
    for idx in range(samples):
        left = int(round(idx * (len(a) - 1) / (samples - 1)))
        right = int(round(idx * (len(b) - 1) / (samples - 1)))
        matches += int(a[left] == b[right])
    return matches / samples


def _chunk_vector(
    class_counts: Counter[str],
    pc_counts: Counter[str],
    instruction_count: int,
    pc_change_count: int,
    destination_count: int,
    memory_count: int,
) -> np.ndarray:
    total = max(1, int(instruction_count))
    values = [float(class_counts.get(name, 0)) / total for name in TRACE_CLASSES]
    values.extend([
        len(pc_counts) / total,
        pc_change_count / max(1, total - 1),
        destination_count / total,
        memory_count / total,
    ])
    return np.asarray(values, dtype=np.float32)


def _pool_relative_segments(chunks: Sequence[np.ndarray], segment_count: int) -> np.ndarray:
    width = len(TRACE_CLASSES) + 4
    out = np.zeros((segment_count, width), dtype=np.float32)
    counts = np.zeros(segment_count, dtype=np.float32)
    if not chunks:
        return out
    for idx, chunk in enumerate(chunks):
        segment = min(segment_count - 1, int(idx * segment_count / len(chunks)))
        out[segment] += np.asarray(chunk, dtype=np.float32)
        counts[segment] += 1.0
    for idx in range(segment_count):
        if counts[idx] > 0:
            out[idx] /= counts[idx]
    return out


def _counter_document(prefix: str, counter: Counter[str], limit: int) -> list[str]:
    if not counter:
        return []
    maximum = max(counter.values())
    tokens: list[str] = []
    for value, count in counter.most_common(limit):
        clean = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(value))
        bucket = min(4, int(round(4.0 * count / max(1, maximum))))
        tokens.extend((f"{prefix}_{clean}", f"{prefix}_{clean}_Q{bucket}"))
    return tokens


def _empty_feature(case_id: str, trace_path: str, status: str, segment_count: int) -> HierarchicalTraceFeature:
    return HierarchicalTraceFeature(
        case_id=case_id,
        trace_path=trace_path,
        file_status=status,
        instruction_count=0,
        opcode_counts={},
        transition_counts={},
        register_counts={},
        pc_region_counts={},
        class_counts={},
        global_struct=np.zeros(23 + 64 + 64 + 32 + 32 + segment_count * (len(TRACE_CLASSES) + 4), dtype=np.float32),
        anchor_struct=np.zeros(len(ANCHOR_SOURCES) + len(LOCATED_BY) + 3 * (len(TRACE_CLASSES) + 5), dtype=np.float32),
        global_document="TRACE_MISSING",
        anchor_document="TRACE_MISSING",
        segment_matrix=np.zeros((segment_count, len(TRACE_CLASSES) + 4), dtype=np.float32),
        anchor_opcodes={32: (), 64: (), 128: ()},
        anchor_pc_regions={32: (), 64: (), 128: ()},
        anchor_registers={32: (), 64: (), 128: ()},
        anchor_source="fallback_tail",
        located_by=status,
        anchor_tags=(),
        target_pc_present=False,
        target_register_present=False,
    )


def build_trace_residual_struct(feature: HierarchicalTraceFeature) -> np.ndarray:
    """Failure-window distributions normalized by the case's full trace."""
    global_total = max(1, feature.instruction_count)
    global_class = np.asarray(
        [float(feature.class_counts.get(name, 0)) / global_total for name in TRACE_CLASSES],
        dtype=np.float32,
    )
    blocks: list[np.ndarray] = []
    radii = sorted(feature.anchor_opcodes)
    for radius in radii:
        opcodes = feature.anchor_opcodes.get(radius, ())
        pcs = feature.anchor_pc_regions.get(radius, ())
        regs = feature.anchor_registers.get(radius, ())
        local_opcodes = Counter(opcodes)
        local_pcs = Counter(pcs)
        local_regs = Counter(regs)
        local_classes = Counter(ts._classify_opcode(opcode) for opcode in opcodes)
        local_total = max(1, len(opcodes))
        class_delta = np.asarray(
            [float(local_classes.get(name, 0)) / local_total for name in TRACE_CLASSES],
            dtype=np.float32,
        ) - global_class
        blocks.extend([
            class_delta,
            _hashed_distribution_delta(local_opcodes, feature.opcode_counts, 32),
            _hashed_distribution_delta(local_pcs, feature.pc_region_counts, 16),
            _hashed_distribution_delta(local_regs, feature.register_counts, 16),
            np.asarray([
                len(set(opcodes)) / local_total - len(feature.opcode_counts) / global_total,
                len(set(pcs)) / local_total - len(feature.pc_region_counts) / global_total,
                float(feature.target_pc_present),
                float(feature.target_register_present),
            ], dtype=np.float32),
        ])
    if not blocks:
        blocks.append(np.zeros(3 * (len(TRACE_CLASSES) + 32 + 16 + 16 + 4), dtype=np.float32))
    scale_features: list[float] = []
    for left, right in zip(radii, radii[1:]):
        scale_features.extend([
            _jaccard(feature.anchor_opcodes.get(left, ()), feature.anchor_opcodes.get(right, ())),
            _jaccard(feature.anchor_pc_regions.get(left, ()), feature.anchor_pc_regions.get(right, ())),
            _jaccard(feature.anchor_registers.get(left, ()), feature.anchor_registers.get(right, ())),
        ])
    while len(scale_features) < 6:
        scale_features.append(0.0)
    blocks.append(np.asarray(scale_features[:6], dtype=np.float32))
    return np.concatenate(blocks).astype(np.float32, copy=False)


def build_trace_residual_document(feature: HierarchicalTraceFeature) -> str:
    tokens = ["TRACE_RESIDUAL", f"RES_LOC_{feature.located_by}"]
    global_total = max(1.0, float(feature.instruction_count))
    for radius in sorted(feature.anchor_opcodes):
        local = Counter(feature.anchor_opcodes.get(radius, ()))
        local_total = max(1.0, float(sum(local.values())))
        for opcode, count in local.most_common(48):
            local_rate = float(count) / local_total
            global_rate = float(feature.opcode_counts.get(opcode, 0)) / global_total
            enrichment = math.log((local_rate + 1e-4) / (global_rate + 1e-4))
            direction = "UP" if enrichment >= 0 else "DOWN"
            bucket = min(6, int(abs(enrichment)))
            tokens.append(f"R{radius}_OP_{opcode}_{direction}_Q{bucket}")
        sequence = feature.anchor_opcodes.get(radius, ())
        local_transitions = Counter(f"{a}>{b}" for a, b in zip(sequence, sequence[1:]))
        for transition, count in local_transitions.most_common(32):
            local_rate = float(count) / max(1.0, float(sum(local_transitions.values())))
            global_rate = float(feature.transition_counts.get(transition, 0)) / max(
                1.0, float(sum(feature.transition_counts.values()))
            )
            direction = "UP" if local_rate >= global_rate else "DOWN"
            tokens.append(f"R{radius}_TR_{transition}_{direction}")
    return " ".join(tokens)


def _cache_key(paths: Sequence[Path | None], config: str) -> str:
    parts = [f"v={CACHE_VERSION}", config]
    for path in paths:
        if path is None:
            parts.append("missing")
            continue
        try:
            stat = path.stat()
            parts.append(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            parts.append(f"{path}:missing")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _select_center(candidates: dict[str, tuple[float, int] | int | None], instruction_count: int) -> tuple[int, str]:
    pc_value = candidates.get("pc")
    if isinstance(pc_value, int):
        return pc_value, "pc"
    for name in ("index", "time"):
        value = candidates.get(name)
        if isinstance(value, tuple):
            return int(value[1]), name
    opcode_value = candidates.get("opcode")
    if isinstance(opcode_value, int):
        return opcode_value, "opcode"
    return max(0, instruction_count - 1), "tail"


def parse_hierarchical_trace(
    case_id: str,
    trace_path: Path,
    anchor: ta.TraceAnchor,
    segment_count: int = 16,
    chunk_size: int = 512,
    anchor_sizes: Sequence[int] = (32, 64, 128),
    max_instructions: int = 50000,
) -> HierarchicalTraceFeature:
    if not trace_path.exists():
        return _empty_feature(case_id, str(trace_path), "missing", segment_count)

    opcode_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    register_counts: Counter[str] = Counter()
    pc_region_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    memory_counts: Counter[str] = Counter()
    chunks: list[np.ndarray] = []
    chunk_classes: Counter[str] = Counter()
    chunk_pcs: Counter[str] = Counter()
    chunk_n = chunk_pc_changes = chunk_destinations = chunk_memory = 0
    previous_opcode = previous_pc = ""
    pc_change_count = destination_count = memory_count = 0
    first_time = last_time = first_cycle = last_cycle = None
    target_pcs = {value for value in (anchor.dut_pc, anchor.iss_pc) if value}
    target_opcodes = {value for value in (anchor.ibex_opcode, anchor.spike_opcode) if value}
    candidates: dict[str, tuple[float, int] | int | None] = {
        "pc": None,
        "index": None,
        "time": None,
        "opcode": None,
    }
    instruction_count = 0

    # 截断优化：只保留尾段 max_instructions 条指令原始文本（轻量判断），
    # 跳过前部的完整正则解析（fatal 类型 bug 的 trace 会冲到几十万行）。
    tail_lines = deque(maxlen=max_instructions)
    try:
        with _open_text(trace_path) as handle:
            for line in handle:
                if "\t" in line:
                    tail_lines.append(line)
    except OSError:
        return _empty_feature(case_id, str(trace_path), "read_error", segment_count)

    for line in tail_lines:
        item = _parse_instruction(line)
        if item is None:
            continue
        idx = instruction_count
        instruction_count += 1
        opcode = str(item["opcode"])
        cls = str(item["class"])
        pc = str(item.get("pc", "")).lower().removeprefix("0x")
        pc_region = str(item.get("pc_region", ""))
        regs = [str(value) for value in item.get("regs", ())]
        opcode_counts[opcode] += 1
        class_counts[cls] += 1
        if pc_region:
            pc_region_counts[pc_region] += 1
        register_counts.update(regs)
        memory_counts.update(item.get("memory_regions", ()))
        if previous_opcode:
            transition_counts[f"{previous_opcode}>{opcode}"] += 1
        if previous_pc and pc and previous_pc != pc:
            pc_change_count += 1
            chunk_pc_changes += 1
        previous_opcode, previous_pc = opcode, pc
        destination_count += int(bool(item.get("rd")))
        memory_count += len(item.get("memory_regions", ()))

        chunk_n += 1
        chunk_classes[cls] += 1
        if pc_region:
            chunk_pcs[pc_region] += 1
        chunk_destinations += int(bool(item.get("rd")))
        chunk_memory += len(item.get("memory_regions", ()))
        if chunk_n >= chunk_size:
            chunks.append(_chunk_vector(chunk_classes, chunk_pcs, chunk_n, chunk_pc_changes, chunk_destinations, chunk_memory))
            chunk_classes = Counter()
            chunk_pcs = Counter()
            chunk_n = chunk_pc_changes = chunk_destinations = chunk_memory = 0

        time_value = item.get("time")
        cycle_value = item.get("cycle")
        if time_value is not None:
            first_time = time_value if first_time is None else first_time
            last_time = time_value
        if cycle_value is not None:
            first_cycle = cycle_value if first_cycle is None else first_cycle
            last_cycle = cycle_value
        if candidates["pc"] is None and pc and pc in target_pcs:
            candidates["pc"] = idx
        if anchor.instr_index is not None and cycle_value is not None:
            delta = abs(float(cycle_value) - float(anchor.instr_index))
            if candidates["index"] is None or delta < float(candidates["index"][0]):
                candidates["index"] = (delta, idx)
        if anchor.sim_time is not None and time_value is not None:
            delta = abs(float(time_value) - float(anchor.sim_time))
            if candidates["time"] is None or delta < float(candidates["time"][0]):
                candidates["time"] = (delta, idx)
        if candidates["opcode"] is None and opcode in target_opcodes:
            candidates["opcode"] = idx

    if chunk_n:
        chunks.append(_chunk_vector(chunk_classes, chunk_pcs, chunk_n, chunk_pc_changes, chunk_destinations, chunk_memory))
    if instruction_count == 0:
        return _empty_feature(case_id, str(trace_path), "empty", segment_count)

    center, located_by = _select_center(candidates, instruction_count)
    max_radius = max(int(value) for value in anchor_sizes)
    window_items: list[tuple[int, dict[str, Any]]] = []
    try:
        with _open_text(trace_path) as handle:
            parsed_index = 0
            for line in handle:
                item = _parse_instruction(line)
                if item is None:
                    continue
                if center - max_radius <= parsed_index <= center + max_radius:
                    window_items.append((parsed_index, item))
                if parsed_index > center + max_radius:
                    break
                parsed_index += 1
    except OSError:
        return _empty_feature(case_id, str(trace_path), "read_error", segment_count)

    segment_matrix = _pool_relative_segments(chunks, segment_count)
    total = max(1, instruction_count)
    class_ratios = [class_counts.get(name, 0) / total for name in TRACE_CLASSES]
    span_time = 0 if first_time is None or last_time is None else max(0, last_time - first_time)
    span_cycle = 0 if first_cycle is None or last_cycle is None else max(0, last_cycle - first_cycle)
    global_scalars = np.asarray([
        1.0,
        math.log1p(instruction_count) / 16.0,
        len(opcode_counts) / total,
        len(transition_counts) / max(1, total - 1),
        len(pc_region_counts) / total,
        len(register_counts) / 32.0,
        len(memory_counts) / total,
        pc_change_count / max(1, total - 1),
        destination_count / total,
        memory_count / total,
        math.log1p(span_time) / 24.0,
        math.log1p(span_cycle) / 18.0,
        *class_ratios,
        float(any(op in opcode_counts for op in ("ebreak", "dret", "mret"))),
        float(bool(class_counts.get("CLASS_CSR"))),
        float(bool(class_counts.get("CLASS_SYSTEM"))),
        float(located_by != "tail"),
    ], dtype=np.float32)
    global_struct = np.concatenate([
        global_scalars,
        _hashed_counter(opcode_counts, 64),
        _hashed_counter(transition_counts, 64),
        _hashed_counter(register_counts, 32),
        _hashed_counter(pc_region_counts, 32),
        segment_matrix.reshape(-1),
    ]).astype(np.float32, copy=False)

    anchor_opcodes: dict[int, tuple[str, ...]] = {}
    anchor_pcs: dict[int, tuple[str, ...]] = {}
    anchor_regs: dict[int, tuple[str, ...]] = {}
    anchor_blocks: list[np.ndarray] = []
    target_pc_present = False
    target_register_present = False
    anchor_tokens = [f"ASRC_{anchor.source}", f"LOC_{located_by}"]
    anchor_tokens.extend(f"ATAG_{tag}" for tag in anchor.anchor_tags)
    for radius in anchor_sizes:
        selected = [item for idx, item in window_items if center - int(radius) <= idx <= center + int(radius)]
        opcodes = tuple(str(item["opcode"]) for item in selected)
        pcs = tuple(str(item.get("pc_region", "")) for item in selected if item.get("pc_region"))
        regs = tuple(str(reg) for item in selected for reg in item.get("regs", ()))
        anchor_opcodes[int(radius)] = opcodes
        anchor_pcs[int(radius)] = pcs
        anchor_regs[int(radius)] = regs
        local_classes = Counter(str(item["class"]) for item in selected)
        local_total = max(1, len(selected))
        pc_hit = any(str(item.get("pc", "")).lower().removeprefix("0x") in target_pcs for item in selected)
        reg_hit = bool(anchor.mismatch_register and anchor.mismatch_register in regs)
        target_pc_present = target_pc_present or pc_hit
        target_register_present = target_register_present or reg_hit
        local = [local_classes.get(name, 0) / local_total for name in TRACE_CLASSES]
        local.extend([
            len(set(opcodes)) / local_total,
            len(set(pcs)) / local_total,
            len(set(regs)) / max(1, len(regs)),
            float(pc_hit),
            float(reg_hit),
        ])
        anchor_blocks.append(np.asarray(local, dtype=np.float32))
        prefix = f"A{int(radius)}"
        anchor_tokens.extend(f"{prefix}_OP_{value}" for value in opcodes)
        anchor_tokens.extend(f"{prefix}_PC_{value}" for value in pcs[:64])
        anchor_tokens.extend(f"{prefix}_REG_{value}" for value in regs[:64])

    source_onehot = np.asarray([float(anchor.source == value) for value in ANCHOR_SOURCES], dtype=np.float32)
    located_onehot = np.asarray([float(located_by == value) for value in LOCATED_BY], dtype=np.float32)
    anchor_struct = np.concatenate([source_onehot, located_onehot, *anchor_blocks]).astype(np.float32, copy=False)

    global_tokens = ["TRACE_OK"]
    global_tokens.extend(_counter_document("GOP", opcode_counts, 64))
    global_tokens.extend(_counter_document("GTR", transition_counts, 64))
    global_tokens.extend(_counter_document("GPC", pc_region_counts, 32))
    global_tokens.extend(_counter_document("GREG", register_counts, 32))
    for idx, row in enumerate(segment_matrix):
        for class_idx, class_name in enumerate(TRACE_CLASSES):
            bucket = min(4, int(round(float(row[class_idx]) * 4.0)))
            if bucket:
                global_tokens.append(f"S{idx:02d}_{class_name}_Q{bucket}")

    return HierarchicalTraceFeature(
        case_id=case_id,
        trace_path=str(trace_path),
        file_status="ok",
        instruction_count=instruction_count,
        opcode_counts=dict(opcode_counts),
        transition_counts=dict(transition_counts),
        register_counts=dict(register_counts),
        pc_region_counts=dict(pc_region_counts),
        class_counts=dict(class_counts),
        global_struct=global_struct,
        anchor_struct=anchor_struct,
        global_document=" ".join(global_tokens),
        anchor_document=" ".join(anchor_tokens) if anchor_tokens else "ANCHOR_EMPTY",
        segment_matrix=segment_matrix,
        anchor_opcodes=anchor_opcodes,
        anchor_pc_regions=anchor_pcs,
        anchor_registers=anchor_regs,
        anchor_source=anchor.source,
        located_by=located_by,
        anchor_tags=tuple(anchor.anchor_tags),
        target_pc_present=target_pc_present,
        target_register_present=target_register_present,
    )


def _process_one_case(task: tuple) -> tuple[str, HierarchicalTraceFeature, bool]:
    """Parse (or load from cache) a single case's hierarchical trace feature.

    Module-level so it can be pickled for multiprocessing.  Returns
    (case_id, feature, cache_hit).
    """
    (input_csv, case_id, sim_path, regr_path, trace_path, config, cache_dir,
     segment_count, chunk_size, anchor_sizes, max_instructions, force_rebuild) = task
    key = _cache_key((sim_path, regr_path, trace_path), config)
    cache_path = cache_dir / f"{input_csv.parent.name}_{case_id}_{key[:20]}.pkl"
    feature: HierarchicalTraceFeature | None = None
    cache_hit = False
    if cache_path.exists() and not force_rebuild:
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("version") == CACHE_VERSION:
                feature = payload["feature"]
                cache_hit = True
        except Exception:
            feature = None
    if feature is None:
        sim_text, _ = rfb.read_log_sample(sim_path)
        regr_text, _ = rfb.read_log_sample(regr_path)
        anchor = ta.extract_trace_anchor(case_id, sim_text.splitlines(), regr_text.splitlines())
        if trace_path is None:
            feature = _empty_feature(case_id, "", "missing", segment_count)
        else:
            feature = parse_hierarchical_trace(
                case_id,
                trace_path,
                anchor,
                segment_count=segment_count,
                chunk_size=chunk_size,
                anchor_sizes=anchor_sizes,
                max_instructions=max_instructions,
            )
        with cache_path.open("wb") as handle:
            pickle.dump({"version": CACHE_VERSION, "feature": feature}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return case_id, feature, cache_hit


def build_hierarchical_trace_features(
    input_csv: str | Path,
    cache_dir: str | Path = "/tmp/theta_trilog_trace_cache",
    segment_count: int = 16,
    chunk_size: int = 512,
    anchor_sizes: Sequence[int] = (32, 64, 128),
    force_rebuild: bool = False,
    max_instructions: int = 5000,
) -> tuple[list[HierarchicalTraceFeature], list[dict[str, Any]]]:
    input_csv = Path(input_csv).resolve()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows, fields = rfb.read_csv_rows(input_csv)
    sim_col = rfb.pick_column(fields, "sim")
    regr_col = rfb.pick_column(fields, "regr")
    trace_col = rfb.pick_column(fields, "trace")
    output: list[HierarchicalTraceFeature] = []
    debug: list[dict[str, Any]] = []
    config = f"segments={segment_count};chunk={chunk_size};anchors={','.join(map(str, anchor_sizes))};max_inst={max_instructions}"

    tasks: list[tuple] = []
    for idx, row in enumerate(rows):
        case_id = osf.infer_case_id(input_csv, row, fields, idx)
        sim_path = rfb.resolve_log_path(input_csv, row.get(sim_col) if sim_col else None)
        regr_path = rfb.resolve_log_path(input_csv, row.get(regr_col) if regr_col else None)
        trace_path = rfb.resolve_log_path(input_csv, row.get(trace_col) if trace_col else None)
        tasks.append((input_csv, case_id, sim_path, regr_path, trace_path, config,
                      cache_dir, segment_count, chunk_size, anchor_sizes, max_instructions, force_rebuild))

    # 多进程并行解析（trace 解析是 CPU 密集，32 核并行能 ~30x 加速）。
    # 每个 case 写独立的 cache 文件，多进程间不冲突。
    import os as _os
    n_proc = min(32, _os.cpu_count() or 1)
    if n_proc > 1 and len(tasks) > 8:
        from multiprocessing import Pool
        with Pool(processes=n_proc) as pool:
            results = pool.map(_process_one_case, tasks)
    else:
        results = [_process_one_case(t) for t in tasks]

    for case_id, feature, cache_hit in results:
        output.append(feature)
        debug.append({
            "input_csv": str(input_csv),
            "case_id": case_id,
            "trace_path": feature.trace_path,
            "trace_status": feature.file_status,
            "instruction_count": feature.instruction_count,
            "anchor_source": feature.anchor_source,
            "located_by": feature.located_by,
            "target_pc_present": int(feature.target_pc_present),
            "target_register_present": int(feature.target_register_present),
            "cache_hit": int(cache_hit),
        })
    return output, debug


def _pad_columns(matrix: np.ndarray, dim: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[1] >= dim:
        return matrix[:, :dim]
    return np.pad(matrix, ((0, 0), (0, dim - matrix.shape[1]))).astype(np.float32)


def _fit_dense_view(matrix: np.ndarray, train_indices: Sequence[int], dim: int, seed: int) -> tuple[dict[str, Any], np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    matrix = np.asarray(matrix, dtype=np.float32)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    scaler = StandardScaler()
    scaler.fit(matrix[train_indices])
    scaled = scaler.transform(matrix)
    n_components = min(int(dim), len(train_indices) - 1, matrix.shape[1])
    if n_components >= 2:
        reducer = PCA(n_components=n_components, random_state=seed)
        reducer.fit(scaled[train_indices])
        transformed = reducer.transform(scaled)
    else:
        reducer = None
        transformed = scaled[:, : max(1, n_components)]
    return {"scaler": scaler, "reducer": reducer, "dim": int(dim), "n_components": int(n_components)}, _pad_columns(transformed, int(dim))


def _fit_text_view(documents: Sequence[str], train_indices: Sequence[int], dim: int, seed: int) -> tuple[dict[str, Any], np.ndarray]:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import Normalizer

    train_indices = np.asarray(train_indices, dtype=np.int64)
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\S+\b",
        lowercase=False,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=8192,
    )
    vectorizer.fit([documents[idx] for idx in train_indices])
    sparse = vectorizer.transform(documents)
    n_components = min(int(dim), len(train_indices) - 1, sparse.shape[1] - 1)
    if n_components >= 2:
        reducer = TruncatedSVD(n_components=n_components, random_state=seed)
        reducer.fit(sparse[train_indices])
        transformed = reducer.transform(sparse)
    else:
        reducer = None
        transformed = sparse[:, : max(1, n_components)].toarray()
    transformed = Normalizer(copy=False).fit_transform(transformed)
    return {"vectorizer": vectorizer, "reducer": reducer, "dim": int(dim), "n_components": int(n_components)}, _pad_columns(transformed, int(dim))


def fit_transform_trace_views(
    features: Sequence[HierarchicalTraceFeature],
    train_indices: Sequence[int],
    seed: int = 0,
    global_struct_dim: int = 48,
    global_text_dim: int = 48,
    anchor_struct_dim: int = 32,
    anchor_text_dim: int = 32,
    residual_struct_dim: int = 48,
    residual_text_dim: int = 48,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    raw = build_trace_raw_views(features)
    global_struct, anchor_struct = raw["global_struct"], raw["anchor_struct"]
    residual_struct = raw["residual_struct"]
    global_docs, anchor_docs = raw["global_text"], raw["anchor_text"]
    residual_docs = raw["residual_text"]
    bundle: dict[str, Any] = {}
    matrices: dict[str, np.ndarray] = {}
    bundle["global_struct"], matrices["global_struct"] = _fit_dense_view(global_struct, train_indices, global_struct_dim, seed + 11)
    bundle["global_text"], matrices["global_text"] = _fit_text_view(global_docs, train_indices, global_text_dim, seed + 23)
    bundle["anchor_struct"], matrices["anchor_struct"] = _fit_dense_view(anchor_struct, train_indices, anchor_struct_dim, seed + 37)
    bundle["anchor_text"], matrices["anchor_text"] = _fit_text_view(anchor_docs, train_indices, anchor_text_dim, seed + 53)
    bundle["residual_struct"], matrices["residual_struct"] = _fit_dense_view(residual_struct, train_indices, residual_struct_dim, seed + 67)
    bundle["residual_text"], matrices["residual_text"] = _fit_text_view(residual_docs, train_indices, residual_text_dim, seed + 79)
    return bundle, matrices


def build_trace_raw_views(features: Sequence[HierarchicalTraceFeature]) -> dict[str, Any]:
    return {
        "global_struct": np.vstack([feature.global_struct for feature in features]).astype(np.float32),
        "anchor_struct": np.vstack([feature.anchor_struct for feature in features]).astype(np.float32),
        "residual_struct": np.vstack([build_trace_residual_struct(feature) for feature in features]).astype(np.float32),
        "global_text": [feature.global_document for feature in features],
        "anchor_text": [feature.anchor_document for feature in features],
        "residual_text": [build_trace_residual_document(feature) for feature in features],
    }


def _apply_dense_view(matrix: np.ndarray, bundle: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    scaled = bundle["scaler"].transform(matrix)
    reducer = bundle.get("reducer")
    if reducer is not None:
        transformed = reducer.transform(scaled)
    else:
        transformed = scaled[:, : max(1, int(bundle.get("n_components", 1)))]
    return _pad_columns(transformed, int(bundle["dim"]))


def _apply_text_view(documents: Sequence[str], bundle: dict[str, Any]) -> np.ndarray:
    from sklearn.preprocessing import Normalizer

    sparse = bundle["vectorizer"].transform(documents)
    reducer = bundle.get("reducer")
    if reducer is not None:
        transformed = reducer.transform(sparse)
    else:
        transformed = sparse[:, : max(1, int(bundle.get("n_components", 1)))].toarray()
    transformed = Normalizer().transform(transformed)
    return _pad_columns(transformed, int(bundle["dim"]))


def apply_trace_reducers(bundle: dict[str, Any], features: Sequence[HierarchicalTraceFeature]) -> dict[str, np.ndarray]:
    raw = build_trace_raw_views(features)
    return {
        "global_struct": _apply_dense_view(raw["global_struct"], bundle["global_struct"]),
        "global_text": _apply_text_view(raw["global_text"], bundle["global_text"]),
        "anchor_struct": _apply_dense_view(raw["anchor_struct"], bundle["anchor_struct"]),
        "anchor_text": _apply_text_view(raw["anchor_text"], bundle["anchor_text"]),
        "residual_struct": _apply_dense_view(raw["residual_struct"], bundle["residual_struct"]),
        "residual_text": _apply_text_view(raw["residual_text"], bundle["residual_text"]),
    }


def build_trace_sequence(
    features: Sequence[HierarchicalTraceFeature],
    seq_len: int = 128,
) -> np.ndarray:
    """Ordered SPECIFIC-opcode sequence centered on the divergence anchor.

    Returns an int32 array of shape (n_cases, seq_len): each position is the
    SPECIFIC opcode index (0..71 via failure_signature.OPCODE_VOCAB) of the retired
    instruction nearest the divergence, ordered by trace position.  72 = OOV,
    73 = PAD.  Using the specific opcode (not just the family) is essential to
    separate same-family control-flow bugs (BNE vs BLT vs BGE vs C.BEQZ).
    """
    import failure_signature as fsig
    vocab = len(fsig.OPCODE_VOCAB)  # 72
    OOV = vocab      # 72
    PAD = vocab + 1  # 73
    out = np.full((len(features), seq_len), PAD, dtype=np.int64)
    for i, f in enumerate(features):
        opcodes = f.anchor_opcodes.get(128) or f.anchor_opcodes.get(64) or f.anchor_opcodes.get(32) or ()
        if not opcodes:
            continue
        toks = [fsig.OPCODE_IDX.get(op, OOV) for op in opcodes]
        if len(toks) > seq_len:
            half = len(toks) // 2
            start = max(0, half - seq_len // 2)
            toks = toks[start:start + seq_len]
        offset = (seq_len - len(toks)) // 2
        out[i, offset:offset + len(toks)] = toks
    return out
    raw = build_trace_raw_views(features)
    return {
        "global_struct": _apply_dense_view(raw["global_struct"], bundle["global_struct"]),
        "global_text": _apply_text_view(raw["global_text"], bundle["global_text"]),
        "anchor_struct": _apply_dense_view(raw["anchor_struct"], bundle["anchor_struct"]),
        "anchor_text": _apply_text_view(raw["anchor_text"], bundle["anchor_text"]),
        "residual_struct": _apply_dense_view(raw["residual_struct"], bundle["residual_struct"]),
        "residual_text": _apply_text_view(raw["residual_text"], bundle["residual_text"]),
    }


def _relation(matrix: np.ndarray, left: int, right: int) -> np.ndarray:
    a, b = matrix[left], matrix[right]
    cosine = float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))
    return np.concatenate([
        np.abs(a - b),
        a * b,
        np.asarray([cosine, float(np.linalg.norm(a - b))], dtype=np.float32),
    ]).astype(np.float32, copy=False)


def _relation_matrix(matrix: np.ndarray, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    """Vectorized abs/product/cosine/distance relation block."""
    width = int(matrix.shape[1])
    if not pairs:
        return np.zeros((0, 2 * width + 2), dtype=np.float32)
    left = np.fromiter((pair[0] for pair in pairs), dtype=np.int64, count=len(pairs))
    right = np.fromiter((pair[1] for pair in pairs), dtype=np.int64, count=len(pairs))
    a = np.asarray(matrix[left], dtype=np.float32)
    b = np.asarray(matrix[right], dtype=np.float32)
    difference = a - b
    dot = np.einsum("ij,ij->i", a, b)
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = dot / np.maximum(norm, 1e-12)
    euclidean = np.linalg.norm(difference, axis=1)
    return np.hstack([
        np.abs(difference),
        a * b,
        cosine[:, None],
        euclidean[:, None],
    ]).astype(np.float32, copy=False)


def build_trace_pair_scalars(a: HierarchicalTraceFeature, b: HierarchicalTraceFeature, include_anchor: bool = True) -> np.ndarray:
    segment_a = a.segment_matrix.reshape(-1)
    segment_b = b.segment_matrix.reshape(-1)
    segment_cosine = float(np.dot(segment_a, segment_b) / max(float(np.linalg.norm(segment_a) * np.linalg.norm(segment_b)), 1e-12))
    values = [
        _counter_cosine(a.opcode_counts, b.opcode_counts),
        _jaccard(a.opcode_counts, b.opcode_counts),
        _counter_cosine(a.transition_counts, b.transition_counts),
        _jaccard(a.transition_counts, b.transition_counts),
        _counter_cosine(a.register_counts, b.register_counts),
        _jaccard(a.register_counts, b.register_counts),
        _counter_cosine(a.pc_region_counts, b.pc_region_counts),
        _jaccard(a.pc_region_counts, b.pc_region_counts),
        segment_cosine,
        float(np.mean(np.abs(segment_a - segment_b))),
        abs(math.log1p(a.instruction_count) - math.log1p(b.instruction_count)) / 16.0,
        float(a.has_trace and b.has_trace),
        float(a.has_trace != b.has_trace),
    ]
    if include_anchor:
        radius = max(set(a.anchor_opcodes) | set(b.anchor_opcodes) | {128})
        ops_a = a.anchor_opcodes.get(radius, ())
        ops_b = b.anchor_opcodes.get(radius, ())
        values.extend([
            _jaccard(ops_a, ops_b),
            _jaccard(_cached_ngrams(tuple(ops_a), 2), _cached_ngrams(tuple(ops_b), 2)),
            _jaccard(_cached_ngrams(tuple(ops_a), 3), _cached_ngrams(tuple(ops_b), 3)),
            _ordered_overlap(ops_a, ops_b),
            _jaccard(a.anchor_pc_regions.get(radius, ()), b.anchor_pc_regions.get(radius, ())),
            _jaccard(a.anchor_registers.get(radius, ()), b.anchor_registers.get(radius, ())),
            float(bool(a.anchor_source and a.anchor_source == b.anchor_source)),
            float(bool(a.located_by and a.located_by == b.located_by)),
            _jaccard(a.anchor_tags, b.anchor_tags),
            float(a.target_pc_present == b.target_pc_present),
            float(a.target_register_present == b.target_register_present),
        ])
    return np.asarray(values, dtype=np.float32)


def build_trace_pair_feature_matrix(
    features: Sequence[HierarchicalTraceFeature],
    view_matrices: dict[str, np.ndarray],
    pairs: Sequence[tuple[int, int]],
    mode: str = "full",
) -> np.ndarray:
    if mode not in {"global", "anchor", "residual", "full", "full_residual"}:
        raise ValueError(f"unsupported trace feature mode: {mode}")
    names: list[str] = []
    if mode in {"global", "full"}:
        names.extend(("global_struct", "global_text"))
    if mode in {"anchor", "full"}:
        names.extend(("anchor_struct", "anchor_text"))
    if mode in {"residual", "full_residual"}:
        names.extend(("residual_struct", "residual_text"))
    if mode == "full_residual":
        names = ["global_struct", "global_text", "anchor_struct", "anchor_text", "residual_struct", "residual_text"]
    relation_blocks = [_relation_matrix(view_matrices[name], pairs) for name in names]
    scalar_width = len(build_trace_pair_scalars(features[0], features[0], include_anchor=mode != "global")) if features else 0
    scalars = np.empty((len(pairs), scalar_width), dtype=np.float32)
    for row, (left, right) in enumerate(pairs):
        scalars[row] = build_trace_pair_scalars(features[left], features[right], include_anchor=mode != "global")
    return np.hstack([*relation_blocks, scalars]).astype(np.float32, copy=False)


def build_trace_pair_feature_components(
    features: Sequence[HierarchicalTraceFeature],
    view_matrices: dict[str, np.ndarray],
    pairs: Sequence[tuple[int, int]],
) -> dict[str, np.ndarray]:
    """Build global/anchor/full blocks while sharing all expensive work."""
    global_relations = np.hstack([
        _relation_matrix(view_matrices["global_struct"], pairs),
        _relation_matrix(view_matrices["global_text"], pairs),
    ]).astype(np.float32, copy=False)
    anchor_relations = np.hstack([
        _relation_matrix(view_matrices["anchor_struct"], pairs),
        _relation_matrix(view_matrices["anchor_text"], pairs),
    ]).astype(np.float32, copy=False)
    residual_relations = np.hstack([
        _relation_matrix(view_matrices["residual_struct"], pairs),
        _relation_matrix(view_matrices["residual_text"], pairs),
    ]).astype(np.float32, copy=False)
    scalar_width = len(build_trace_pair_scalars(features[0], features[0], include_anchor=True)) if features else 24
    full_scalars = np.empty((len(pairs), scalar_width), dtype=np.float32)
    for row, (left, right) in enumerate(pairs):
        full_scalars[row] = build_trace_pair_scalars(features[left], features[right], include_anchor=True)
    global_scalars = full_scalars[:, :13]
    return {
        "global": np.hstack([global_relations, global_scalars]).astype(np.float32, copy=False),
        "anchor": np.hstack([anchor_relations, full_scalars]).astype(np.float32, copy=False),
        "residual": np.hstack([residual_relations, full_scalars]).astype(np.float32, copy=False),
        "full": np.hstack([global_relations, anchor_relations, full_scalars]).astype(np.float32, copy=False),
        "full_residual": np.hstack([global_relations, anchor_relations, residual_relations, full_scalars]).astype(np.float32, copy=False),
    }
