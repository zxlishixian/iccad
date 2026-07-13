#!/usr/bin/env python3
"""Compatibility entry for the experimental parallel anytime supervisor."""

from __future__ import annotations

from pathlib import Path

import official_style_features as osf
import parallel_anytime_inference as pai


def _read_inferred_case_ids(input_csv: Path) -> list[str]:
    # Official inputs use an explicit Case column. Legacy fake datasets do not;
    # use the same robust path-based inference as the evaluated backends so
    # strict candidate validation remains meaningful on both formats.
    return [str(case) for case in osf.read_cases(input_csv)]


def main() -> int:
    pai.read_case_ids = _read_inferred_case_ids
    return pai.main()


if __name__ == "__main__":
    raise SystemExit(main())
