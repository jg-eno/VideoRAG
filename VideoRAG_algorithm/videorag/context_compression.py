"""
Training-free, query-time compression for retrieved text chunks.

Uses only tokenization + lexical overlap with the user query (no learned models).
"""

from __future__ import annotations

import re
from typing import Any, List, Sequence, Tuple

# Sentence boundaries: . ! ? and newlines
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-']{1,}")


def query_term_set(query: str, min_len: int = 2) -> set[str]:
    """Lowercased word-like tokens from the query (filters very short tokens)."""
    terms = {m.group(0).lower() for m in _WORD_RE.finditer(query)}
    return {t for t in terms if len(t) >= min_len}


def chunk_lexical_score(content: str, query_terms: set[str]) -> float:
    """
    Cheap overlap score: sum of query-term hit counts in chunk, length-normalized.
    """
    if not query_terms or not content:
        return 0.0
    words = [m.group(0).lower() for m in _WORD_RE.finditer(content)]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in query_terms)
    return hits / (len(words) ** 0.5)


def _split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    if not parts:
        return [text.strip()] if text.strip() else []
    return parts


def compress_text_to_token_budget(
    text: str,
    query_terms: set[str],
    max_tokens: int,
    enc: Any,
    min_sentence_tokens: int = 4,
) -> str:
    """
    Greedy sentence selection by lexical query score until max_tokens.
    ``enc`` is a tiktoken ``Encoding`` (encode/decode).
    """
    if max_tokens <= 0:
        return ""
    sentences = _split_sentences(text)
    if not sentences:
        return ""

    def encode(s: str) -> list:
        return enc.encode(s)

    scored: List[Tuple[float, int, str]] = []
    for i, s in enumerate(sentences):
        sc = chunk_lexical_score(s, query_terms) + 1e-6 * (1.0 / (i + 1))
        scored.append((sc, i, s))
    scored.sort(key=lambda x: -x[0])

    picked: List[Tuple[int, str]] = []
    used = 0
    for _, orig_idx, sent in scored:
        tlen = len(encode(sent))
        if tlen < min_sentence_tokens and len(scored) > 1:
            continue
        if used + tlen <= max_tokens:
            picked.append((orig_idx, sent))
            used += tlen
        elif used == 0 and tlen > max_tokens:
            tokens = encode(sent)[:max_tokens]
            picked.append((orig_idx, enc.decode(tokens)))
            used = max_tokens
            break

    if not picked:
        tokens = encode(text)[:max_tokens]
        return enc.decode(tokens)

    picked.sort(key=lambda x: x[0])
    return " ".join(s for _, s in picked)


def _get_encoding(model_name: str) -> Any:
    import tiktoken

    return tiktoken.encoding_for_model(model_name)


def select_chunks_baseline(
    chunks: Sequence[dict],
    max_token_size: int,
) -> Tuple[List[dict], str]:
    """Prefix list truncation by token budget (matches legacy behavior)."""
    from ._utils import truncate_list_by_token_size

    valid = [c for c in chunks if c is not None]
    selected = truncate_list_by_token_size(
        valid,
        key=lambda x: x["content"],
        max_token_size=max_token_size,
    )
    text = "-----New Chunk-----\n".join(c["content"] for c in selected)
    return selected, text


def select_chunks_query_aware(
    chunks: Sequence[dict],
    query: str,
    max_token_size: int,
    tiktoken_model_name: str,
) -> Tuple[List[dict], str]:
    """
    Same token budget as baseline, but:
    1. Sort chunks by lexical relevance to query (tie-break: original retrieval order).
    2. Greedily pack chunks; last partial slot uses sentence-level query-aware trimming.
    """
    enc = _get_encoding(tiktoken_model_name)
    query_terms = query_term_set(query)
    valid = [c for c in chunks if c is not None]
    if not valid:
        return [], ""

    indexed = list(enumerate(valid))
    scored = [
        (
            -chunk_lexical_score(c["content"], query_terms),
            i,
            c,
        )
        for i, c in indexed
    ]
    scored.sort(key=lambda x: (x[0], x[1]))

    selected: List[dict] = []
    remaining = max_token_size
    for _, _, chunk in scored:
        content = chunk["content"]
        tlen = len(enc.encode(content))
        if tlen <= remaining:
            selected.append(chunk)
            remaining -= tlen
        elif remaining > 0:
            trimmed = compress_text_to_token_budget(
                content, query_terms, remaining, enc
            )
            if trimmed.strip():
                new_chunk = {**chunk, "content": trimmed}
                selected.append(new_chunk)
                remaining -= len(enc.encode(trimmed))

    text = "-----New Chunk-----\n".join(c["content"] for c in selected)
    return selected, text


def _token_bag(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def compute_context_compression_metrics(
    query: str,
    baseline_context: str,
    compressed_context: str,
    full_retrieved_context: str,
    tiktoken_model_name: str,
) -> dict[str, Any]:
    """
    Compare baseline (prefix truncation) vs query-aware compression.

    Metrics:
    - token_counts: baseline / compressed / full retrieved union
    - query_term_recall_*: fraction of query terms (len>=2) present in each context
    - relative_query_recall: compressed / baseline query recall (target ~1.0; <1 means loss)
    - baseline_content_recall: |intersection(baseline tokens, compressed)| / |baseline tokens|
      (how much of the baseline wording survives compression)
    - oracle_query_gain: query recall compressed vs full (how close to upper bound from pool)
    """
    enc = _get_encoding(tiktoken_model_name)

    def tok_len(s: str) -> int:
        return len(enc.encode(s))

    q_terms = query_term_set(query)
    if not q_terms:
        q_terms = set(query.lower().split())

    def q_recall(ctx: str) -> float:
        if not q_terms:
            return 1.0
        bag = _token_bag(ctx)
        hit = len(q_terms & bag)
        return hit / len(q_terms)

    tb = tok_len(baseline_context)
    tc = tok_len(compressed_context)
    tf = tok_len(full_retrieved_context)

    qb = q_recall(baseline_context)
    qc = q_recall(compressed_context)
    qf = q_recall(full_retrieved_context)

    bag_b = _token_bag(baseline_context)
    bag_c = _token_bag(compressed_context)
    if bag_b:
        content_recall = len(bag_b & bag_c) / len(bag_b)
    else:
        content_recall = 1.0 if not bag_c else 0.0

    return {
        "token_count_baseline": tb,
        "token_count_compressed": tc,
        "token_count_full_pool": tf,
        "token_reduction_ratio": (1.0 - tc / tb) if tb else 0.0,
        "query_term_recall_baseline": qb,
        "query_term_recall_compressed": qc,
        "query_term_recall_full_pool": qf,
        "relative_query_recall_compressed_vs_baseline": (qc / qb) if qb > 1e-9 else float(
            qc > 0
        ),
        "baseline_content_token_recall": content_recall,
        "accuracy_proxy": 0.5 * min((qc / qb) if qb > 1e-9 else float(qc > 0), 2.0)
        + 0.5 * content_recall,
    }
