#!/usr/bin/env python3
"""Assemble a self-contained final_submission/ directory.

Copies final_inference.py and every repo-local module it transitively imports
into final_submission/, then copies the trained manifest.json and model/
reducers from ``--model-dir``. The resulting directory runs standalone: the
``regr_fail_bucketing`` shell wrapper executes ``final_inference.py`` from the
same directory, so Python resolves all local imports from the bundled copies.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import stat
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SUBMISSION_DIR = PROJECT_ROOT / "final_submission"
ENTRY = "final_inference"


def local_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                out.add(node.module.split(".")[0])
    return out


def local_modules() -> set[str]:
    return {p.stem for p in PROJECT_ROOT.glob("*.py")}


def transitive_closure(start: str) -> list[str]:
    all_local = local_modules()
    queue, seen = [start], set()
    while queue:
        mod = queue.pop()
        if mod in seen:
            continue
        seen.add(mod)
        path = PROJECT_ROOT / f"{mod}.py"
        if not path.exists():
            continue
        for imported in local_imports(path):
            if imported in all_local and imported not in seen:
                queue.append(imported)
    return sorted(seen)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True, help="trained run_final_submission_train output")
    p.add_argument("--submission-dir", type=Path, default=SUBMISSION_DIR)
    args = p.parse_args(argv)

    model_dir = args.model_dir
    manifest = model_dir / "manifest.json"
    models = model_dir / "models"
    if not manifest.exists() or not models.exists():
        raise SystemExit(f"--model-dir {model_dir} is missing manifest.json or models/")

    out = args.submission_dir
    out.mkdir(parents=True, exist_ok=True)

    closure = transitive_closure(ENTRY)
    copied = []
    for mod in closure:
        src = PROJECT_ROOT / f"{mod}.py"
        dst = out / f"{mod}.py"
        shutil.copyfile(src, dst)
        copied.append(mod)
    print(f"[package] copied {len(copied)} modules: {', '.join(copied)}")

    shutil.copyfile(manifest, out / "manifest.json")
    dst_models = out / "models"
    if dst_models.exists():
        shutil.rmtree(dst_models)
    shutil.copytree(models, dst_models)
    n_pt = len(list(dst_models.glob("model_*_seed*.pt")))
    n_pkl = len(list(dst_models.glob("preprocess_*_seed*.pkl")))
    print(f"[package] copied manifest.json and models/ ({n_pt} .pt, {n_pkl} .pkl)")

    wrapper = out / "regr_fail_bucketing"
    if wrapper.exists():
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"[package] assembled {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
