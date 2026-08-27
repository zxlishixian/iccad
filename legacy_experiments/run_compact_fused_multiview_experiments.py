#!/usr/bin/env python3
"""Experimental compact four-view strict-LODO runner.

Unlike the naive fused-global ablation, this view keeps only the leading
high-value lines from features and summary.  The average stage2 document is
about 1.9K characters instead of 5.1K, so reducing the number of embedding
views also reduces total token work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pairwise_features as pf
import regr_fail_bucketing as rfb
import run_graph_multiview_experiments as gm


_ORIGINAL_BUILD_DOCUMENTS = gm.build_all_view_documents
_ORIGINAL_BUILD_CASE_FEATURES = gm.plf.build_llm_case_features_for_inputs


def _trim_feature(document: str, max_lines: int = 18) -> list[str]:
    lines = document.splitlines()
    if lines and lines[0].startswith("case_index:"):
        lines[0] = "case_index:"
    return lines[:max_lines]


def _trim_summary(document: str, max_lines: int = 10) -> list[str]:
    lines = document.splitlines()
    if lines and lines[0].startswith("Case "):
        lines[0] = "Regression failure summary."
    return lines[:max_lines]


def build_compact_documents(dataset: Path) -> dict[str, list[str]]:
    docs = _ORIGINAL_BUILD_DOCUMENTS(dataset)
    input_csv = dataset / "input.csv"
    _case_ids, base_features, normalized_lines, _infos = pf._collect_case_inputs(
        input_csv, "drain"
    )
    parser_args = pf._make_parser_args("drain")
    counters, _template_count = rfb.build_feature_counters(
        parser_args,
        base_features,
        normalized_lines,
        token_weights=None,
        token_weight_mode="none",
    )
    feature_docs = rfb.build_llm_case_documents(
        counters, max_features=80, doc_style="features"
    )
    summary_docs = rfb.build_llm_case_documents(
        counters, max_features=80, doc_style="summary"
    )
    docs["compact_global"] = [
        "\n".join((
            "COMPACT_GLOBAL_FAILURE_EVIDENCE:",
            "FEATURE_VIEW:",
            *_trim_feature(feature_doc),
            "SUMMARY_VIEW:",
            *_trim_summary(summary_doc),
        ))
        for feature_doc, summary_doc in zip(feature_docs, summary_docs)
    ]
    return docs


def main(argv: Sequence[str] | None = None) -> int:
    gm.VIEW_CONFIGS["compact_fused_event_object_context"] = [
        "compact_global", "event", "object", "context"
    ]
    gm.build_all_view_documents = build_compact_documents

    def build_structured_case_features(
        inputs, parser, svd_dim, llm_args=None, **kwargs
    ):
        return _ORIGINAL_BUILD_CASE_FEATURES(
            inputs,
            parser=parser,
            svd_dim=svd_dim,
            llm_args=None,
            **kwargs,
        )

    # The compact view configuration does not consume the legacy dual vectors.
    gm.plf.build_llm_case_features_for_inputs = build_structured_case_features
    args = list(argv) if argv is not None else None
    return gm.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
