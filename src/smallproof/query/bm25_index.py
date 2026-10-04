"""BM25 keyword retrieval with rank_bm25.

rank_bm25 has no save format, so the index is rebuilt from the saved chunks
when a pipeline loads (fast: the Phase 0 bench measures it).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from smallproof.core.interfaces import Retriever
from smallproof.core.registry import register
from smallproof.core.types import Chunk

# Small English stopword list: question words would otherwise dominate short queries.
STOPWORDS = set(
    "a an and are as at be been by did do does for from had has have how in is it its of on or that the "
    "their this to was were what when which who why will with".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, thousands separators removed (1,577 -> 1577), stopwords dropped."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())
    return [token for token in re.findall(r"\w+", text) if token not in STOPWORDS]


@register("retriever", "bm25")
class BM25Retriever(Retriever):
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        self.index = BM25Okapi([tokenize(chunk.text) for chunk in self.chunks])

    def retrieve(self, query: str, filters: dict[str, Any] | None, k: int) -> list[Chunk]:
        """Top k chunks with a positive score. filters keep only chunks whose metadata matches (used for the
        extra filtered search; the unfiltered search always runs too, so the overall effect stays soft)."""
        import numpy as np

        scores = self.index.get_scores(tokenize(query))
        if filters:
            allowed = np.array([all(chunk.metadata.get(key) == value for key, value in filters.items())
                                for chunk in self.chunks])
            scores = np.where(allowed, scores, 0.0)
        top = np.argsort(-scores)[:k]
        return [self.chunks[i].with_score(float(scores[i])) for i in top if scores[i] > 0]
