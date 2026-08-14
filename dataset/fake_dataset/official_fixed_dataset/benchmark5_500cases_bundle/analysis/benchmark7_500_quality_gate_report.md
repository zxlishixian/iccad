# Quality Gate Report: benchmark7_500

## Verdict

```text
pass: true
```

## Selected Distribution

```text
num_selected: 500
dual_compare_rate: 0.600
op_pair_nonempty_rate: 0.600
same_test_pair_gap: 0.256
```

## Feature Summary

```text
num_cases: 500
num_bugs: 10
dual_compare_count: 300
op_pair_nonempty_count: 300
pc_region_count: 2
sim_template_unique_count: 468
regr_template_unique_count: 307
```

## Pair Separability

```text
positive_pairs: 12250
negative_pairs: 112500
same_test_gap: 0.256
same_mismatch_type_gap: 0.192
same_op_pair_gap: 0.277
token_jaccard_gap: 0.372
deterministic_cosine_gap: 0.180
```

## Warnings

- `test_single_or_low_bug:riscv_debug_branch_jump_test:bug_056`
- `test_single_or_low_bug:riscv_debug_csr_entry_test:bug_037`
- `test_single_or_low_bug:riscv_debug_ebreak_test:bug_037`
- `test_single_bug:riscv_debug_branch_jump_test:bug_056`
- `test_single_bug:riscv_debug_csr_entry_test:bug_037`
- `test_single_bug:riscv_debug_ebreak_test:bug_037`
