# Batch 4 Root-Cause Catalog

## Provenance

- RTL baseline: upstream lowRISC Ibex commit
  `8ce399dbe678f0a66856ac302ec7609ba366d8fd`.
- Injection scope: RTL files only. No UVM testbench, assertion, ISS, or log
  formatting change is used to create a failure.
- Oracle: VCS 2023.12 running the existing Ibex UVM flow, with Spike comparison.
- Comparison policy: `mismatch_print_limit=1`; waveform dumping is disabled.
- Formal plan: 11 roots, 50 planned runs per root, 550 unique seeds in total.

## Batch scope

This is a control-flow-focused batch. Its retained roots cover conditional
branches, JAL/JALR, compressed branch expansion, and link/target calculation.
It is intended to provide a clean, disambiguated batch for these root-cause
families; it does not by itself satisfy the roadmap's eventual broad coverage
of MDU, LSU, CSR/PMP, WFI/interrupt, bus, and reset behavior. Those families
remain explicit targets for subsequent independent batches and the clean
high-`k` aggregate.

## Retained roots

| Root | RTL location | Injected root cause | Formal tests | Expected observable symptom |
|---|---|---|---|---|
| `bug_128` | `rtl/ibex_decoder.sv` | Decode BNE (`funct3=001`) with `ALU_EQ` instead of `ALU_NE`. | machine-mode random; random jump | Taken/not-taken branch divergence followed by an Ibex/Spike mismatch. |
| `bug_129` | `rtl/ibex_decoder.sv` | Decode BLT with unsigned `ALU_LTU` instead of signed `ALU_LT`. | jump stress; machine-mode random | Signed operands with different signed/unsigned ordering produce a control-flow mismatch. |
| `bug_130` | `rtl/ibex_decoder.sv` | Decode BGE with unsigned `ALU_GEU` instead of signed `ALU_GE`. | jump stress; machine-mode random; random jump | Signed boundary operands produce a control-flow mismatch. |
| `bug_132` | `rtl/ibex_compressed_decoder.sv` | Invert the expanded `funct3` selector for C.BEQZ/C.BNEZ. | jump stress; random instruction; random jump | Compressed conditional branch takes the opposite condition and diverges from Spike. |
| `bug_133` | `rtl/ibex_decoder.sv` | Zero-extend, rather than sign-extend, a negative JAL immediate. | debug branch/jump; jump stress; random jump | Backward JAL targets the wrong region, causing mismatch or a standard UVM memory failure. |
| `bug_134` | `rtl/ibex_id_stage.sv` | Replace a new JALR instruction's sign-extended I immediate with its zero-extended 12-bit field. | debug branch/jump; random jump | Negative JALR offset targets an invalid/different address. |
| `bug_136` | `rtl/ibex_decoder.sv` | Select the wrong increment immediate when writing the compressed JALR link. | debug branch/jump; random jump | C.JALR writes an incorrect return address and later diverges. |
| `bug_140` | `rtl/ibex_decoder.sv` | Use the S immediate selector instead of I immediate for the first-cycle JALR target. | debug branch/jump; random jump | JALR target calculation uses unrelated instruction bits. |
| `bug_141` | `rtl/ibex_decoder.sv` | Use register A instead of the current PC while calculating the JAL link value. | debug branch/jump; jump stress; random jump | JAL writes the wrong link register value. |
| `bug_142` | `rtl/ibex_decoder.sv` | Use register A instead of the current PC while calculating the JALR link value. | debug branch/jump; random jump | JALR writes the wrong return address, often surfacing through standard control-flow or memory checks. |
| `bug_143` | `rtl/ibex_compressed_decoder.sv` | Invert bit 7 of the compressed branch source-register field during expansion. | random instruction; random jump | C.BEQZ/C.BNEZ reads the wrong compact register and diverges. |

Every retained root has at least two effective tests. Every formal test covers at
least two roots; no test name uniquely determines a root.

## Probe disambiguation

The initial and supplemental probes generated 57 runs. For the retained roots:

- all observed dual-trace mismatches had a first Ibex index above 32;
- no case had incomplete logs or malformed metadata;
- no cross-root three-log byte-identical case was found;
- two coarse symptom collisions were found and removed at selection time:
  `bug_128/riscv_jump_stress_test` and
  `bug_142/riscv_jump_stress_test` are not present in the formal matrix.

This is selection-level disambiguation. The generated simulator and comparison
logs are not edited to manufacture distinct symptoms.

## Rejected candidates

| Candidate | Reason not used in the formal batch |
|---|---|
| `bug_131` | Probe produced no observable real failure. |
| `bug_135` | One initial observation was not stable; six additional runs produced no real failure. |
| `bug_137` | Probe produced no observable real failure. |
| `bug_138` | Probe produced no observable real failure. |
| `bug_139` | Three supplemental runs produced no real failure. |
| `bug_144` | Three supplemental runs produced no real failure. |

## Finalization gates

The final catalog must be updated after the run with collected case counts. A
package is eligible for release only if all 550 raw runs have valid metadata
and complete logs. A metadata-only selection stage then excludes non-failures,
first mismatch indices at or below 32, every case participating in a cross-root
symptom-signature collision, and every case participating in a cross-root exact
three-log collision. Raw logs are never edited or deleted. The selected staging
set must re-audit with zero shallow mismatches and zero cross-root signature or
exact-log collisions before collection. Finally, the official layout validator
must report zero errors/warnings and the collected label/path leakage checks
must pass. Final retained counts, exclusion reasons, failure-type counts,
first-mismatch depth, and selected-set gate results are generated from the
machine-readable audits into `FORMAL_RESULTS.md`; this avoids relying on
pre-run estimates in the static catalog.
