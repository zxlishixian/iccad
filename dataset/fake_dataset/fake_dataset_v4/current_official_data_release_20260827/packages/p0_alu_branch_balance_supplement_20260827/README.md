# ALU/Branch Balance Supplement (2026-08-27)

This is an independent official-layout supplement for the existing coverage64
candidate. It is not merged into any previous archive.

## Contents

- 6 cases total.
- `bug_045`: 2 cases, conditional arithmetic-right-shift behavior is disabled
  for shift amount 7.
- `bug_047`: 4 cases, equality comparison is inverted for a selected operand
  low-bit pattern.
- Both bugs cover `riscv_machine_mode_rand_test` and
  `riscv_rand_instr_test`.
- Both tests cover both bugs, avoiding a test-name-only shortcut.
- Three additional probe cases from arithmetic-basic/jump-stress tests were
  intentionally excluded because each of those tests covered only one bug.

## Quality checks

- Generated with the standard VCS/UVM/Spike flow on the official Ibex base
  commit.
- Official-layout validator: 6 cases, 0 errors, 0 warnings.
- Full audit: 6 single `Mismatch[1]` cases; first mismatch index 173--3073;
  no index at or before 32.
- No missing failure markers, malformed cases, bug-label/path leakage, or
  exact duplicate log bundles.
- Cross-bug signature audit: 6 failure cases, 0 signature collisions; both
  bugs and both tests have two-way coverage.

Archive: `p0_alu_branch_balance_supplement_20260827.tar.gz`
