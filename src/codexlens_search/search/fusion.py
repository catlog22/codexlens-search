from __future__ import annotations

import re
from enum import Enum

DEFAULT_WEIGHTS: dict[str, float] = {
    "exact": 0.17,
    "fuzzy": 0.085,
    "vector": 0.34,
    "graph": 0.085,
    "symbol": 0.17,
    "entity": 0.15,
}

_CODE_CAMEL_RE = re.compile(r"[a-z][A-Z]")
_CODE_SNAKE_RE = re.compile(r"\b[a-z_]+_[a-z_]+\b")
_CODE_SYMBOLS_RE = re.compile(r"[.\[\](){}]|->|::")
_CODE_KEYWORDS_RE = re.compile(r"\b(import|def|class|return|from|async|await|lambda|yield)\b")
_QUESTION_WORDS_RE = re.compile(r"\b(how|what|why|when|where|which|who|does|do|is|are|can|should)\b", re.IGNORECASE)


class QueryIntent(Enum):
    CODE_SYMBOL = "code_symbol"
    NATURAL_LANGUAGE = "natural"
    MIXED = "mixed"


def detect_query_intent(query: str) -> QueryIntent:
    """Detect whether query is a code symbol, natural language, or mixed."""
    words = query.strip().split()
    word_count = len(words)

    # Long queries (bug reports, issue descriptions) are natural language
    # even if they contain code snippets or identifiers.
    if word_count > 20:
        return QueryIntent.NATURAL_LANGUAGE
    if word_count > 10:
        return QueryIntent.MIXED

    code_signals = 0
    natural_signals = 0

    if _CODE_CAMEL_RE.search(query):
        code_signals += 2
    if _CODE_SNAKE_RE.search(query):
        code_signals += 2
    if _CODE_SYMBOLS_RE.search(query):
        code_signals += 2
    if _CODE_KEYWORDS_RE.search(query):
        code_signals += 2
    if "`" in query:
        code_signals += 1
    if word_count < 4:
        code_signals += 1

    if _QUESTION_WORDS_RE.search(query):
        natural_signals += 2
    if word_count > 5:
        natural_signals += 2
    if code_signals == 0 and word_count >= 3:
        natural_signals += 1

    if code_signals >= 2 and natural_signals == 0:
        return QueryIntent.CODE_SYMBOL
    if natural_signals >= 2 and code_signals == 0:
        return QueryIntent.NATURAL_LANGUAGE
    if natural_signals > code_signals:
        return QueryIntent.NATURAL_LANGUAGE
    if code_signals > natural_signals:
        return QueryIntent.CODE_SYMBOL
    return QueryIntent.MIXED


def get_adaptive_weights(intent: QueryIntent, base: dict | None = None) -> dict[str, float]:
    """Return weights adapted to query intent."""
    weights = dict(base or DEFAULT_WEIGHTS)
    if intent == QueryIntent.CODE_SYMBOL:
        # Balanced: keep vector+graph competitive with symbol+entity.
        weights["exact"] = 0.20
        weights["fuzzy"] = 0.05
        weights["vector"] = 0.20
        weights["graph"] = 0.20
        weights["symbol"] = 0.20
        weights["entity"] = 0.15
    elif intent == QueryIntent.NATURAL_LANGUAGE:
        weights["exact"] = 0.08
        weights["fuzzy"] = 0.07
        weights["vector"] = 0.55
        weights["graph"] = 0.15
        weights["symbol"] = 0.0
        weights["entity"] = 0.15
    else:
        # MIXED
        weights["exact"] = 0.15
        weights["fuzzy"] = 0.07
        weights["vector"] = 0.38
        weights["graph"] = 0.20
        weights["symbol"] = 0.05
        weights["entity"] = 0.15
    return weights


def reciprocal_rank_fusion(
    results: dict[str, list[tuple[int, float]]],
    weights: dict[str, float] | None = None,
    k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse ranked result lists using Reciprocal Rank Fusion.

    results: {source_name: [(doc_id, score), ...]} each list sorted desc by score.
    weights: weight per source (defaults to equal weight across all sources).
    k: RRF constant (default 60).
    Returns sorted list of (doc_id, fused_score) descending.
    """
    if not results:
        return []

    sources = list(results.keys())
    if weights is None:
        equal_w = 1.0 / len(sources)
        weights = {s: equal_w for s in sources}

    scores: dict[int, float] = {}
    for source, ranked_list in results.items():
        w = weights.get(source, 0.0)
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w * (1.0 / (k + rank))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
