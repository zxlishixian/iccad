#!/usr/bin/env python3
"""Cheap ablation: does the RICH divergence signature (DUT-vs-ref opcode pair,
full operand set, destination value) discriminate bugs better than the current
4-tuple (dtype, opcode, register, pc_bucket)?

Reads each case's regr.log, extracts two signatures, clusters with k-means,
and reports Pairwise Balanced Accuracy for each.  No siamese training — this
isolates the raw-feature discrimination.

Run:  python scratch_rich_signature_ablation.py <dataset_dir> [k]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import failure_signature as fs
from official_style_features import read_cases


# ---- rich divergence parsing ----
_PC_LINE = re.compile(
    r"pc\[([^\]]+)\]\s+(\S+)\s+(.*?):\s*(\S+):\s*(\S+)"
)


def parse_divergence(regr_text: str) -> dict:
    """Extract DUT and reference divergence lines from a regr.log.

    Returns a dict with dut/ref opcode, operands, dest, value, pc.
    Empty dict if the two-side mismatch format is absent.
    """
    out: dict = {}
    for line in regr_text.splitlines():
        s = line.strip()
        if s.startswith("ibex") or s.startswith("dut"):
            m = _PC_LINE.search(s)
            if m:
                out["dut"] = {
                    "pc": m.group(1), "opcode": m.group(2).lower(),
                    "ops": [o.strip().split(",")[0] for o in m.group(3).split(",")],
                    "dest": m.group(4), "val": m.group(5),
                }
        elif s.startswith("spike") or s.startswith("ref"):
            m = _PC_LINE.search(s)
            if m:
                out["ref"] = {
                    "pc": m.group(1), "opcode": m.group(2).lower(),
                    "ops": [o.strip().split(",")[0] for o in m.group(3).split(",")],
                    "dest": m.group(4), "val": m.group(5),
                }
    return out


def rich_feature_matrix(regr_texts: list[str]) -> np.ndarray:
    """Build a one-hot matrix from the DUT-vs-ref divergence semantics."""
    n_op = len(fs.OPCODE_VOCAB)
    n_reg = len(fs.REGISTER_VOCAB)
    reg_idx = fs.REGISTER_IDX
    abi = fs.REGISTER_ABI
    op_idx = fs.OPCODE_IDX

    feats = np.zeros((len(regr_texts), n_op * 2 + n_reg * 2 + 4), dtype=np.float32)
    for i, txt in enumerate(regr_texts):
        d = parse_divergence(txt)
        base = 0
        dut = d.get("dut")
        ref = d.get("ref")
        if dut:
            if dut["opcode"] in op_idx:
                feats[i, base + op_idx[dut["opcode"]]] = 1.0
        base += n_op
        if ref:
            if ref["opcode"] in op_idx:
                feats[i, base + op_idx[ref["opcode"]]] = 1.0
        base += n_op
        if dut:
            r = abi.get(dut["dest"], dut["dest"] if dut["dest"].startswith("x") else "None")
            if r in reg_idx:
                feats[i, base + reg_idx[r]] = 1.0
        base += n_reg
        if ref:
            r = abi.get(ref["dest"], ref["dest"] if ref["dest"].startswith("x") else "None")
            if r in reg_idx:
                feats[i, base + reg_idx[r]] = 1.0
        base += n_reg
        # 4 scalar signals
        if dut and ref:
            feats[i, base + 0] = 1.0 if dut["opcode"] == ref["opcode"] else 0.0   # opcode_match
            feats[i, base + 1] = 1.0 if dut["val"] == ref["val"] else 0.0          # value_match
            feats[i, base + 2] = 1.0 if dut["dest"] == ref["dest"] else 0.0        # dest_match
            feats[i, base + 3] = 1.0 if dut["pc"] == ref["pc"] else 0.0            # pc_match
    return feats


def balanced_accuracy(labels: np.ndarray, pred: np.ndarray) -> float:
    labels = np.asarray(labels)
    pred = np.asarray(pred)
    n = len(labels)
    if n < 2:
        return 0.0
    same = labels[:, None] == labels[None, :]
    same_pred = pred[:, None] == pred[None, :]
    eye = ~np.eye(n, dtype=bool)
    tp = (same & same_pred & eye).sum()
    tn = ((~same) & (~same_pred) & eye).sum()
    p = same[eye].sum()
    neg = (~same)[eye].sum()
    tpr = tp / max(p, 1)
    tnr = tn / max(neg, 1)
    return 0.5 * (tpr + tnr)


def main() -> int:
    dataset = Path(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else None
    from sklearn.cluster import KMeans

    cases = read_cases(dataset / "input.csv")
    # golden labels (Case column is the bare numeric id for official-format)
    import csv
    with open(dataset / "golden.csv") as fh:
        rows = list(csv.DictReader(fh))
    by_case = {str(r["Case"]): r["Bug"] for r in rows}
    y = np.array([by_case.get(str(c), "?") for c in cases])
    bug_ids = {b: i for i, b in enumerate(sorted(set(y)))}
    yint = np.array([bug_ids[b] for b in y])
    if k is None:
        k = len(bug_ids)

    # read regr.log texts (normalize bare case id -> case_N dir via _find_regr)
    regr_texts = []
    for c in cases:
        p = fs._find_regr(dataset, str(c))
        regr_texts.append(p.read_text(errors="replace") if p else "")

    # Sig A: current 4-tuple rich signature (training path)
    sigs = [fs.parse_rich_signature(t) for t in regr_texts]
    featA = fs.rich_signature_features(sigs)
    # Sig B: new rich divergence semantics
    featB = rich_feature_matrix(regr_texts)

    def cluster_ba(feat):
        X = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        kk = max(1, min(int(k), X.shape[0]))
        pred = KMeans(n_clusters=kk, random_state=0, n_init=10).fit_predict(X)
        return balanced_accuracy(yint, pred)

    print(f"dataset={dataset.name}  n={len(cases)}  k={k}  bugs={len(bug_ids)}")
    print(f"  Sig A  (dtype+opcode+reg+pc)        BA = {cluster_ba(featA):.4f}")
    print(f"  Sig B  (dut/ref opcode+dest+match)  BA = {cluster_ba(featB):.4f}")
    print(f"  Sig A+B (concat)                    BA = {cluster_ba(np.hstack([featA, featB])):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
