#!/usr/bin/env python3
"""Experimental four-view LODO runner with one fused global embedding.

The existing five-view path embeds features and summary separately.  This
wrapper combines those two documents before embedding, then reuses the event,
object, and context views unchanged.  It monkey-patches only the experimental
runner process; submission and formal prediction files are untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pairwise_features as pf
import regr_fail_bucketing as rfb
import run_graph_multiview_experiments as gm


_ORIGINAL_BUILD_DOCUMENTS = gm.build_all_view_documents


def _canonical_feature_document(document: str) -> str:
    lines = document.splitlines()
    if lines and lines[0].startswith("case_index:"):
        lines[0] = "case_index:"
    return "\n".join(lines)


def _canonical_summary_document(document: str) -> str:
    lines = document.splitlines()
    if lines and lines[0].startswith("Case "):
        lines[0] = "Regression failure summary."
    return "\n".join(lines)


def build_documents_with_fused_global(dataset: Path) -> dict[str, list[str]]:
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
    docs["fused_global"] = [
        "\n".join((
            "GLOBAL_FAILURE_EVIDENCE:",
            "FEATURE_VIEW:",
            _canonical_feature_document(feature_doc),
            "SUMMARY_VIEW:",
            _canonical_summary_document(summary_doc),
        ))
        for feature_doc, summary_doc in zip(feature_docs, summary_docs)
    ]
    return docs


def main(argv: Sequence[str] | None = None) -> int:
    gm.VIEW_CONFIGS["fused_event_object_context"] = [
        "fused_global", "event", "object", "context"
    ]
    gm.build_all_view_documents = build_documents_with_fused_global
    args = list(argv) if argv is not None else None
    return gm.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
