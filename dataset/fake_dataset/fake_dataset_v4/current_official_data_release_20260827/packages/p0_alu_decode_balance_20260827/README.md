# P0 ALU/Decode Balance Supplement

- Generated: 2026-08-27
- Cases: 6
- Bugs: 2 (`bug_053`, `bug_128`)
- Tests: `riscv_rand_instr_test`, `riscv_rv32im_instr_test`
- Layout: official (`input.csv`, `golden.csv`, `case_N/`)

## Quality gates

- Both bugs occur in both tests; no test-to-bug one-to-one mapping.
- All cases contain exactly one `Mismatch[1]` failure marker.
- First mismatch index range: 67-469; no cycle-0 or index-at/below-32 mismatch.
- Dataset validator: 0 errors, 0 warnings.
- Signature audit: 0 cross-bug collisions.
- No injected bug labels or host-specific absolute paths were found.
- Timeout and failed runner jobs were excluded.

## Root causes

See `root_causes.csv`. This supplement is intentionally independent of the 931-case/64-bug release.
