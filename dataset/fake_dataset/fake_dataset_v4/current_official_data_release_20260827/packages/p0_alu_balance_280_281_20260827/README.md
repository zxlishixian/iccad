# ALU Balance Supplement 280/281

- Generated: 2026-08-27
- Cases: 11
- Bugs: 2 (`bug_280`, `bug_281`)
- Tests: `riscv_machine_mode_rand_test`, `riscv_rand_instr_test`, `riscv_rv32im_instr_test`
- Layout: official (`input.csv`, `golden.csv`, `case_N/`)

## Quality gates

- `bug_280=6`, `bug_281=5`; both bugs occur in all three tests.
- Every test covers both bugs; no test-to-bug one-to-one shortcut.
- Ten cases contain one standard `Mismatch[1]`; one is a standard UVM failure.
- First mismatch index range for mismatch cases: 104-2727; no index at or below 32.
- Dataset validator: 0 errors, 0 warnings.
- Signature audit: 0 cross-bug collisions.
- No injected bug labels or host-specific absolute paths were found.
- Only runner jobs with status `ok` were collected.

## Root causes

See `root_causes.csv`. This is an independent supplement and is not merged into the
931-case/64-bug release.
