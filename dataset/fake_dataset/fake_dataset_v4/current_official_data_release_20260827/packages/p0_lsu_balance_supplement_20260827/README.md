# LSU Balance Supplement (2026-08-27)

This is an independent official-layout supplement for the existing coverage64
candidate. It is not merged into any previous archive.

## Contents

- 9 cases total.
- `bug_051`: 6 cases, late byte-load sign-extension error.
- `bug_052`: 3 cases, late byte-load sign-extension error for byte lanes.
- Both bugs cover `riscv_mmu_stress_test` and
  `riscv_unaligned_load_store_test`.
- Both tests cover both bugs, avoiding a test-name-only shortcut.

## Quality checks

- Generated with the standard VCS/UVM/Spike flow on the official Ibex base
  commit.
- Official layout: `input.csv`, `gold.csv`, `golden.csv`, `meta.csv`, and
  `case_N/{regr.log,sim.log.gz,trace.log.gz}`.
- Official-layout validator: 9 cases, 0 errors, 0 warnings.
- Full audit: 9 single `Mismatch[1]` cases; first mismatch index 66--610;
  no index at or before 32.
- No missing failure markers, malformed cases, bug-label/path leakage, or
  exact duplicate log bundles.
- Cross-bug signature audit: 9 failure cases, 0 signature collisions; both bugs
  and both tests have two-way coverage.

Archive: `p0_lsu_balance_supplement_20260827_v2.tar.gz`
