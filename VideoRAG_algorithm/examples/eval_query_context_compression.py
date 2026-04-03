#!/usr/bin/env python3
"""
Offline evaluation: baseline (prefix truncation) vs query-aware chunk context.

Activate your conda environment first (dependencies: tiktoken, numpy, etc.)::

    conda activate videorag

From the **repository root** (directory that contains ``VideoRAG_algorithm/``)::

    PYTHONPATH=VideoRAG_algorithm python VideoRAG_algorithm/examples/eval_query_context_compression.py
    PYTHONPATH=VideoRAG_algorithm python VideoRAG_algorithm/examples/eval_query_context_compression.py --json

Or from inside ``VideoRAG_algorithm/``::

    PYTHONPATH=. python examples/eval_query_context_compression.py

Uses the same metrics as a live ``VideoRAG.query`` when
``QueryParam(use_query_aware_chunk_compression=True, compression_metrics={})``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as script from repo root or VideoRAG_algorithm/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from videorag.context_compression import (
    compute_context_compression_metrics,
    select_chunks_baseline,
    select_chunks_query_aware,
)


def _demo_chunks():
    """Synthetic chunks: vector order differs from query relevance."""
    return [
        {
            "tokens": 0,
            "content": "The weather in Paris was mild. Tourists walked along the Seine.",
            "video_segment_id": "v_0",
            "chunk_order_index": 0,
        },
        {
            "tokens": 0,
            "content": (
                "Iron Man and Spider-Man first met during the Civil War storyline. "
                "Tony Stark recruited Peter Parker and provided the Iron Spider suit."
            ),
            "video_segment_id": "v_1",
            "chunk_order_index": 1,
        },
        {
            "tokens": 0,
            "content": "A recipe for bread uses flour, water, and yeast. Knead for ten minutes.",
            "video_segment_id": "v_2",
            "chunk_order_index": 2,
        },
        {
            "tokens": 0,
            "content": (
                "Spider-Man later fought alongside Iron Man against Thanos. "
                "Their relationship evolved from mentor to allies facing cosmic threats."
            ),
            "video_segment_id": "v_3",
            "chunk_order_index": 3,
        },
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query", default="How did Iron Man help Spider-Man?")
    p.add_argument("--budget", type=int, default=120, help="Max context tokens (tiktoken)")
    p.add_argument("--tiktoken-model", default="gpt-4o")
    p.add_argument("--json", action="store_true", help="Print metrics as JSON only")
    args = p.parse_args()

    chunks = _demo_chunks()
    _, baseline_ctx = select_chunks_baseline(chunks, args.budget)
    _, compressed_ctx = select_chunks_query_aware(
        chunks, args.query, args.budget, args.tiktoken_model
    )
    full_pool_ctx = "-----New Chunk-----\n".join(c["content"] for c in chunks)

    metrics = compute_context_compression_metrics(
        args.query,
        baseline_ctx,
        compressed_ctx,
        full_pool_ctx,
        args.tiktoken_model,
    )

    if args.json:
        print(json.dumps(metrics, indent=2))
        return

    print("Query:", args.query)
    print("Token budget:", args.budget)
    print()
    print("=== Baseline (prefix chunks in retrieval/list order) ===")
    print(baseline_ctx[:800] + ("..." if len(baseline_ctx) > 800 else ""))
    print()
    print("=== Query-aware compressed ===")
    print(compressed_ctx[:800] + ("..." if len(compressed_ctx) > 800 else ""))
    print()
    print("=== Metrics (interpretation) ===")
    print(json.dumps(metrics, indent=2))
    print()
    print(
        "relative_query_recall_compressed_vs_baseline ≈ 1.0 means compressed text "
        "covers query terms as well as baseline; >1 means better coverage."
    )
    print(
        "baseline_content_token_recall: fraction of baseline word-types still present "
        "after compression (high = kept baseline wording)."
    )
    print(
        "accuracy_proxy: average of relative query recall (capped) and content recall — "
        "useful for quick A/B; for end-task accuracy, compare LLM answers on a labeled set."
    )


if __name__ == "__main__":
    main()
