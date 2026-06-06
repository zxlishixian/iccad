# Alpha Submission Checklist

- Submission deadline: June 12, 2026, 17:00 GMT+8.
- Upload method for Problem B: the organizer-provided Google Drive folder.
- Target folder name: `alpha_test_submission`.
- Do not add another compression layer in the target folder.
- Required executable name: `regr_fail_bucketing`.
- Required interface:
  `regr_fail_bucketing --input <input.csv> --output <output.csv> --k <k>`.
- Output columns: `Case,bucket`.
- `LLM_MODEL_CONFIG` must contain YAML text, not a path.
- Alpha runtime allowance is three times the normal limit.
- Public benchmark 1/2 Alpha runtime limit: 90 seconds each.
- Verify the executable bit after upload.
