# Batch 4 Formal Results

## Scope

- Raw runs: 550
- Selected genuine, disambiguated failures: 364
- Excluded runs: 186
- Collected official-layout cases: 364
- Raw simulator/comparator logs were not edited; selection was metadata-only.

## Selected Cases By Root

| Root | Cases |
| --- | --- |
| bug_128 | 47 |
| bug_129 | 46 |
| bug_130 | 41 |
| bug_132 | 46 |
| bug_133 | 41 |
| bug_134 | 14 |
| bug_136 | 25 |
| bug_140 | 25 |
| bug_141 | 20 |
| bug_142 | 18 |
| bug_143 | 41 |

## Exclusion Reasons

| Reason | Runs |
| --- | --- |
| cross_bug_signature_collision | 145 |
| no_real_failure | 40 |
| sanitized_log_label_leak | 2 |

## Failure Types

| Primary type | Cases |
| --- | --- |
| REGR_MISMATCH | 255 |
| UVM_ERROR | 108 |
| UVM_FATAL | 1 |

## First Mismatch Depth

- Minimum Ibex instruction index: 56
- Maximum Ibex instruction index: 9700
- Indexed mismatch cases: 255

## Selected-Set Quality Gates

| Gate | Violations |
| --- | --- |
| bad_meta | 0 |
| missing_logs | 0 |
| startup_or_shallow_mismatches | 0 |
| cross_bug_signature_collisions_with_pc_region | 0 |
| cross_bug_exact_three_log_collisions | 0 |

All listed gates must be zero for this report to be packaged.
