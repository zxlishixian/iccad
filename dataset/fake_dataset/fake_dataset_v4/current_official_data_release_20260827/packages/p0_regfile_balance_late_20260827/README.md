# Register-File Balance Supplement

- Generated: 2026-08-27
- Cases: 9
- Bugs: 2 (`bug_223`, `bug_224`)
- Tests: `riscv_machine_mode_rand_test`, `riscv_rand_instr_test`, `riscv_rv32im_instr_test`
- Layout: official (`input.csv`, `golden.csv`, `case_N/`)

## Quality gates

- `bug_223=6`, `bug_224=3`; both bugs occur in all three tests.
- Every test covers both bugs; no test-to-bug one-to-one shortcut.
- All cases contain exactly one standard `Mismatch[1]` failure marker.
- First mismatch index range: 68-176; no index at or below 32.
- Dataset validator: 0 errors, 0 warnings.
- Signature audit: 0 cross-bug collisions.
- No injected bug labels or host-specific absolute paths were found.
- Only runner jobs with status `ok` were collected.

## Root causes

See `root_causes.csv`. This is an independent supplement and is not merged into the
931-case/64-bug release.
