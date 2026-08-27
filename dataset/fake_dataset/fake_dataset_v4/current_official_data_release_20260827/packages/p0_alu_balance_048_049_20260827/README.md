# ALU Balance Supplement 048/049 (2026-08-27)

This is an independent official-layout supplement for the existing coverage64
candidate. It is not merged into any previous archive.

## Contents

- 8 cases total.
- `bug_048`: 3 cases, a selected ADD/SUB operand pattern flips one result bit.
- `bug_049`: 5 cases, arithmetic-right-shift behavior is disabled for shift
  amount 31.
- Both bugs cover `riscv_machine_mode_rand_test` and
  `riscv_rand_instr_test`.
- Both tests cover both bugs, avoiding a test-name-only shortcut.

## Quality checks

- Generated with the standard VCS/UVM/Spike flow on the official Ibex base
  commit.
- Official-layout validator: 8 cases, 0 errors, 0 warnings.
- Full audit: 8 single `Mismatch[1]` cases; first mismatch index 208--1786;
  no index at or before 32.
- No missing failure markers, malformed cases, bug-label/path leakage, or
  exact duplicate log bundles.
- Cross-bug signature audit: 8 failure cases, 0 signature collisions; both
  bugs and both tests have two-way coverage.

Archive: `p0_alu_balance_048_049_20260827.tar.gz`
