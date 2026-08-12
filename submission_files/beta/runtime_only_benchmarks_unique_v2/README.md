# Runtime-Only Benchmarks

These datasets are derived from public official logs exclusively for runtime, watchdog, memory, and output-format testing. They contain no labels and MUST NOT be used for training, model selection, calibration, or score reporting.

`max_lines` is the official target. `materialized_lines` records the actual local stress size; use `--max-materialized-lines 0` only when intentionally generating the full, potentially very large context workload.
