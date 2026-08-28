# Globally Deduplicated Official-Flow Dataset

Derived from the 53 independent packages in the current release. The first
occurrence of each exact three-log signature is kept, with the primary
coverage64 package taking precedence.

- Cases after deduplication: 1376
- Exact duplicate case copies removed: 76
- Distinct bug labels: 118
- Signature: SHA256 of decompressed regr.log, sim.log.gz, and trace.log.gz
- No cross-bug exact duplicate signatures were found.
- Original independent packages are not modified.

See audit.json for package-level overlap details.
