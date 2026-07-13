#!/usr/bin/env python3
"""Fail-closed train/evaluation overlap checks for experimental artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import regr_fail_bucketing as rfb


class TrainingEvaluationOverlapError(RuntimeError):
    """Raised when evaluation data overlaps an artifact's training data."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_log_hashes(dataset: Path) -> list[str]:
    """Hash the sim/regr text visible to the model for every input case."""
    input_path = dataset.resolve() / "input.csv"
    rows, fields = rfb.read_csv_rows(input_path)
    sim_col = rfb.pick_column(fields, "sim")
    regr_col = rfb.pick_column(fields, "regr")
    hashes: list[str] = []
    for row in rows:
        sim_path = rfb.resolve_log_path(input_path, row.get(sim_col) if sim_col else None)
        regr_path = rfb.resolve_log_path(input_path, row.get(regr_col) if regr_col else None)
        sim_text, _ = rfb.read_log_sample(sim_path)
        regr_text, _ = rfb.read_log_sample(regr_path)
        visible = sim_text + "\n<REGR_LOG>\n" + regr_text
        hashes.append(hashlib.sha256(visible.encode("utf-8", errors="replace")).hexdigest())
    return sorted(set(hashes))


def dataset_identity(dataset: Path, include_case_logs: bool = False) -> dict[str, object]:
    dataset = dataset.resolve()
    input_path = dataset / "input.csv"
    if not input_path.is_file():
        raise FileNotFoundError(f"missing dataset input: {input_path}")
    identity: dict[str, object] = {
        "name": dataset.name,
        "input_sha256": file_sha256(input_path),
    }
    if include_case_logs:
        identity["case_log_sha256"] = case_log_hashes(dataset)
    return identity


def training_identities(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    raw = manifest.get("training_datasets")
    identities: list[dict[str, object]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name", "")).strip()
            digest = str(item.get("input_sha256", "")).strip().lower()
            case_hashes = item.get("case_log_sha256", [])
            if not isinstance(case_hashes, list):
                case_hashes = []
            normalized_hashes = sorted({
                str(value).strip().lower() for value in case_hashes if str(value).strip()
            })
            if name or digest or normalized_hashes:
                identities.append({
                    "name": name,
                    "input_sha256": digest,
                    "case_log_sha256": normalized_hashes,
                })
    if identities:
        return identities
    names = manifest.get("training_dataset_names")
    if not isinstance(names, list) or not names:
        raise ValueError(
            "artifact manifest has no training dataset identities; refusing held-out scoring"
        )
    return [
        {"name": str(name).strip(), "input_sha256": "", "case_log_sha256": []}
        for name in names
    ]


def find_training_overlap(
    manifest: Mapping[str, object], eval_datasets: Sequence[Path]
) -> list[dict[str, object]]:
    trained = training_identities(manifest)
    overlaps: list[dict[str, object]] = []
    for dataset in eval_datasets:
        current = dataset_identity(dataset, include_case_logs=True)
        current_name = str(current["name"])
        current_hash = str(current["input_sha256"]).lower()
        current_cases = set(current.get("case_log_sha256", []))
        for train in trained:
            train_cases = set(train.get("case_log_sha256", []))
            shared_cases = current_cases & train_cases
            same_name = bool(train["name"]) and train["name"] == current_name
            # Equal CSV text is not enough when relative paths resolve to
            # different log contents. Use the CSV hash only for legacy
            # identities that lack per-case content hashes.
            same_hash = (
                (not current_cases or not train_cases)
                and bool(train["input_sha256"])
                and train["input_sha256"] == current_hash
            )
            if same_name or shared_cases or same_hash:
                matched_by = (
                    "name" if same_name
                    else "case_log_sha256" if shared_cases
                    else "input_sha256"
                )
                overlaps.append({
                    "evaluation_dataset": current_name,
                    "training_dataset": str(train["name"]),
                    "matched_by": matched_by,
                    "shared_case_hashes": len(shared_cases),
                })
    return overlaps


def assert_held_out(manifest: Mapping[str, object], eval_datasets: Sequence[Path]) -> None:
    overlaps = find_training_overlap(manifest, eval_datasets)
    if overlaps:
        details = ", ".join(
            f"{item['evaluation_dataset']} vs {item['training_dataset']} "
            f"({item['matched_by']}, shared_cases={item['shared_case_hashes']})"
            for item in overlaps
        )
        raise TrainingEvaluationOverlapError(
            "refusing non-independent scoring; evaluation data overlaps artifact "
            f"training data: {details}"
        )
