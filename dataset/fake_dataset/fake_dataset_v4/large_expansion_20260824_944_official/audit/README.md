# Large Expansion Dataset (944 Cases)

This archive is a new standalone expansion dataset. It is not merged with the earlier 944-case baseline or any previous package.

## Provenance

- RTL base: lowRISC Ibex commit `8ce399dbe678f0a66856ac302ec7609ba366d8fd`
- RTL simulation: Synopsys VCS `V-2023.12-SP2`
- ISS: Spike
- Flow: official UVM/riscv-dv flow through `run_matrix.py`, `collect_dataset.py`, and `validate_dataset.py`
- Raw candidates: 1041
- Failure-only collected candidates: 961
- Final selected cases: 944

## Layout

The dataset root contains the official files `input.csv`, `golden.csv`, `gold.csv`, `meta.csv`, and `case_N/{regr.log,sim.log.gz,trace.log.gz}`.

## Quality Results

- Official validator: 944 cases, 0 errors, 0 warnings
- Bugs: 30
- Tests: 10
- Failure-marked cases: 944 according to the official validator warning gate
- Mismatch cases: 699
- First mismatch range: index 39 to 5287
- First mismatch at or below index 32: 0
- Detailed `Mismatch[1]` cases: 699; violations: 0
- Cross-bug identical three-log groups: 0
- Label/path leakage: 0 according to the official validator

See `quality_audit.json`, `case_audit.json`, `root_cause_map.csv`, and `validation.json` for machine-readable evidence.

## Known Coverage Limitation

Three low-volume bugs are represented by one test only, and `riscv_illegal_instr_test` is represented by one bug only. These are retained as clean, observable cases rather than replaced with synthetic or modified logs.
