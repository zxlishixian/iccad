#!/usr/bin/env python3
"""Generate runtime-only profiles with normalization-resistant unique signals."""

from __future__ import annotations

import generate_runtime_only_benchmarks as base


_ORIGINAL_COMBINED_LINES = base._combined_lines


def _alpha_tag(value: int) -> str:
    chars = []
    value = max(1, value)
    while value:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(ord("a") + remainder))
    return "".join(reversed(chars)).rjust(4, "a")


def _combined_lines(left, right, kind: str, case_index: int) -> list[str]:
    lines = _ORIGINAL_COMBINED_LINES(left, right, kind, case_index)
    # Digits are normalized by the production parser. A pure alphabetic tag
    # survives normalization and UVM_FATAL source/reason fields survive primary-signature extraction,
    # preventing artificial cross-case embedding deduplication.
    tag = _alpha_tag(case_index)
    return [
        (
            f"UVM_FATAL /runtime_only_unique_{tag}.sv(1) @ 1: "
            f"runtime_only_unique_failure_{tag} normalization resistant timing marker"
        ),
        *lines,
    ]


def main() -> int:
    base._combined_lines = _combined_lines
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
