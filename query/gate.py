"""Evidence gate: is there enough evidence to answer?

The evidence score comes from one of three signals (compared on the dev split):
- rerank: the best reranker probability among the final chunks
- nli:    NLI entailment of "the passages answer the question"
- both:   the mean of the two
score >= sufficient_threshold -> sufficient, >= partial_threshold -> partial,
otherwise insufficient. Thresholds are calibrated on the dev split only.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.interfaces import Gate
from core.registry import register
from core.types import Chunk, Verdict

MODES = ("rerank", "nli", "both")


@register("gate", "nli_rerank")
class EvidenceGate(Gate):
    def __init__(self, sufficient_threshold: float, partial_threshold: float, mode: str = "rerank",
                 classifier=None, max_premise_chars: int = 6000) -> None:
        if mode not in MODES:
            raise ValueError(f"Unknown gate mode '{mode}'. Choose one of {MODES}")
        if mode != "rerank" and classifier is None:
            raise ValueError("The nli and both modes need a classifier")
        self.sufficient, self.partial = sufficient_threshold, partial_threshold
        self.mode, self.classifier, self.max_premise_chars = mode, classifier, max_premise_chars

    def evidence_score(self, query: str, chunks: Sequence[Chunk]) -> float:
        """chunks must carry reranker probabilities in .score."""
        rerank = max((chunk.score or 0.0 for chunk in chunks), default=0.0)
        if self.mode == "rerank":
            return rerank
        premise = "\n\n".join(chunk.text for chunk in chunks)[: self.max_premise_chars]
        nli = self.classifier.entailment(premise, f"These passages answer the question: {query}")
        return nli if self.mode == "nli" else (rerank + nli) / 2

    def verdict_for(self, score: float) -> Verdict:
        if score >= self.sufficient:
            return Verdict.SUFFICIENT
        return Verdict.PARTIAL if score >= self.partial else Verdict.INSUFFICIENT

    def check(self, query: str, chunks: Sequence[Chunk]) -> tuple[Verdict, float]:
        score = self.evidence_score(query, chunks) if chunks else 0.0
        return self.verdict_for(score), round(score, 4)
