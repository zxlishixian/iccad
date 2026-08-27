# Current Official-Flow Data Release

Generated: 2026-08-27

This is an inventory bundle of the data already generated under the official-flow
constraints. It contains the authoritative 931-case/64-bug candidate and the
independent supplements produced during coverage and balance work.

## How to use

- Use `packages/coverage64_release_20260827_v5.tar.gz` as the primary dataset.
- Treat every other archive as an independent supplement.
- Do not concatenate archives directly: several supplements intentionally overlap
  bug IDs, tests, or cases with the primary dataset.
- Validate and deduplicate after selecting a training split.
- Each archive keeps its own official layout, root-cause information, and audit
  materials.

## Current state

- Primary candidate: 931 cases, 64 bugs.
- All selected supplements passed the official-layout validator and the applicable
  failure-marker, leakage, duplicate-triplet, and signature checks documented in
  `DATA_GENERATION_ROADMAP.md`.
- The package inventory contains 1292 case copies across the primary dataset and
  independent supplements; this is not a claim of 1292 unique cases.
- PMP, RV32B, and AHB-Lite remain unobserved under the official configuration and
  standard oracle flow; no synthetic cases were fabricated for them.

See `manifest.csv` for the exact archive inventory and `../DATA_GENERATION_ROADMAP.md`
for the generation and rejection ledger.
