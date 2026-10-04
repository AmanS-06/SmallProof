"""Cross-encoder reranker (ms-marco MiniLM L6), CPU.

Scores are turned into probabilities (sigmoid) so they are on a fixed 0 to 1
scale that the evidence gate can threshold.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from smallproof.core.interfaces import Reranker
from smallproof.core.registry import register
from smallproof.core.types import Chunk


@register("reranker", "cross_encoder")
class CrossEncoderReranker(Reranker):
    def __init__(self, model_path: str, batch_size: int = 32) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_path, device="cpu")
        self.batch_size = batch_size
        # Find out once whether the model already outputs probabilities or raw logits.
        probe = self._raw([("what was net income", "Net income was $5 billion."), ("what was net income", "The sky is blue.")])
        self.outputs_logits = any(score < 0 or score > 1 for score in probe)

    def _raw(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(s) for s in self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)]

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        raw = self._raw([(query, text) for text in texts]) if texts else []
        return [1 / (1 + math.exp(-s)) for s in raw] if self.outputs_logits else raw

    def rerank(self, query: str, chunks: Sequence[Chunk], k: int) -> list[Chunk]:
        scores = self.score(query, [chunk.text for chunk in chunks])
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])[:k]
        return [chunks[i].with_score(scores[i]) for i in order]
