# Debug Balance Supplement

- Generated: 2026-08-27
- Cases: 12
- Bugs: 2 (`bug_273`, `bug_276`)
- Tests: `riscv_debug_csr_entry_test`, `riscv_debug_ebreak_test`, `riscv_debug_ebreakmu_test`
- Layout: official (`input.csv`, `golden.csv`, `case_N/`)

## Quality gates

- `bug_273=6`, `bug_276=6`; both bugs occur in all three tests.
- Every test covers both bugs; no test-to-bug one-to-one shortcut.
- All cases contain a standard UVM failure marker and no synthetic mismatch marker.
- Failure-only validator: 0 errors, 0 warnings.
- Signature audit: 0 cross-bug collisions.
- No injected bug labels or host-specific absolute paths were found.
- Only runner jobs with status `ok` were collected.

## Root causes

See `root_causes.csv`. This is an independent supplement and is not merged into the
931-case/64-bug release.
