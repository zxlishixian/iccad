# Coverage64 Official Candidate Release v5

This is the current high-k candidate dataset. It is independent of the smaller
supplement archives in this directory.

- Cases: 931
- Distinct bugs: 64
- Layout: official `input.csv`, `golden.csv`, `case_N/{regr.log,sim.log.gz,trace.log.gz}`
- Ibex base commit: `8ce399dbe678f0a66856ac302ec7609ba366d8fd`
- Oracle: standard VCS/UVM testbench with Spike; RTL-only injections

## Quality Audit

- Official validator: 931 cases, 0 errors, 0 warnings.
- Failure marker: present in 931/931 cases.
- Mismatch cases: 769; standard UVM failure cases without mismatch blocks: 162.
- First mismatch index: 37--6820; 0 cases at or before index 32.
- Every bug appears in at least two tests; every selected test covers at least two bugs.
- Missing logs: 0; bad metadata: 0; bug-label or host-path leakage: 0.
- Exact cross-bug three-log duplicate groups: 0.
- Maximum log length: 4,179,017 lines, below the 10M-line limit.

Root causes and machine-readable audit are provided as
`coverage64_release_20260827_v5_root_cause_map.csv` and
`coverage64_release_20260827_v5_audit.json`. The archive checksum is in
`coverage64_release_20260827_v5.tar.gz.sha256`.

## Scope Boundary

The 64 retained bugs cover ALU, Branch/Jump, LSU, MDU, RV32C/decode, decode,
register-file/writeback, CSR, interrupt/exception, Debug, and pipeline/control
roots. PMP, AHB-Lite, and RV32B are not fabricated: no stable,
collision-free standard-oracle candidates were available in the current
Ibex/UVM configuration.
