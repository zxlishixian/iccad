# Runtime-Only Final-Limit Stress Validation

## Scope

The generated `runtime_only_benchmarks_unique_v2/` datasets are derived from
the public official set1/set2 logs. They exist only to exercise runtime,
memory, watchdog, unseen-document embedding, and output-format behavior.

- They contain no `gold.csv`, `golden.csv`, or `meta.csv`.
- They must not be used for training, calibration, model selection, or score
  reporting.
- Pure alphabetic failure source/reason tags keep documents distinct after
  the production normalizer, Drain parser, and embedding canonicalizer.
- The local materialization cap is about 100k lines per benchmark; the
  manifest preserves the corresponding official Max Lines target.

## Cold-Cache Results

The parallel anytime policy publishes singleton output immediately, runs the
deterministic baseline, and only upgrades to the expert when the expert
finishes and passes strict `Case,bucket` validation.

| Profile | Cases | k | Final limit | Wall time | Selected | Valid |
|---|---:|---:|---:|---:|---|---|
| B1 | 10 | 2 | 30 s | 18.83 s | five-view expert | yes |
| B2 | 30 | 4 | 30 s | 23.48 s | baseline | yes |
| B3 | 100 | 8 | 100 s | 71.20 s | five-view expert | yes |
| B4 | 300 | 16 | 100 s | 71.39 s | baseline after expert timeout | yes |
| B5 | 1000 | 32 | 100 s | 8.70 s | baseline | yes |
| B6 | 3000 | 64 | 100 s | 9.81 s | baseline | yes |

For B3, 492 of 500 five-view documents were unique. The baseline was valid at
5.56 seconds and the expert replaced it at 69.66 seconds.

A separate B4 two-seed selective five-view run used 596 unique dual documents
and did not complete within 95 seconds. Therefore a multi-view expert is not
safe for the 300-case Final profile under cold unseen-document conditions.

After disabling the expert for `n > 100`, B4 completed with a valid baseline
in 12.35 seconds, leaving 87.6 seconds of the Final limit.

## Recommended Experimental Routing

- `n <= 100`: baseline and expert may run concurrently within the profile
  deadline; publish expert only after strict validation.
- `n > 100`: use the deterministic baseline path; do not wait for the current
  embedding expert.
- Always write singleton output before backend startup.
- Preserve the latest validated output on timeout, crash, invalid dimensions,
  malformed output, or missing backend executable.

This is an experimental runtime policy. The existing Alpha and Beta submission
directories were not modified by this experiment.
