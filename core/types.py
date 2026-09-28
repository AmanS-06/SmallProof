"""Shared data types used by every stage of the pipeline.

Chunk:      one piece of a document, with its page range for citations.
Citation:   one page reference in an answer, linked to its document.
Answer:     what the pipeline returns for a question.
Verdict:    the evidence gate's decision.
LabelScore: one classifier result, for example ("table", 0.91).
Entity:     one extracted span, for example ("3M", "company", 0.88).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, NamedTuple


class Verdict(StrEnum):
    """Evidence gate decision. A StrEnum saves to JSON as plain text."""

    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class LabelScore(NamedTuple):
    """One classifier result. Still a plain tuple, as the brief's interface says."""

    label: str
    score: float


class Entity(NamedTuple):
    """One extracted span with its type and score."""

    span: str
    type: str
    score: float


@dataclass
class Chunk:
    """One piece of a document.

    Pages are 1-based and inclusive, so a chunk on a single page has
    page_start == page_end.
    """

    id: str
    doc_id: str
    text: str
    page_start: int
    page_end: int
    chunk_type: str | None = None  # table, definition, numeric or narrative
    type_score: float | None = None  # classifier confidence for chunk_type
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None  # set by retrieval or reranking, None after ingest

    def __post_init__(self) -> None:
        if self.page_start < 1 or self.page_end < self.page_start:
            raise ValueError(f"Chunk {self.id}: bad page range {self.page_start}-{self.page_end}")

    @property
    def pages(self) -> range:
        """Every page this chunk covers."""
        return range(self.page_start, self.page_end + 1)

    def with_score(self, score: float) -> Chunk:
        """Return a copy with a new score, so a stage never changes a chunk another stage holds.

        The copy is shallow: the metadata dict is shared, so treat it as read-only.
        """
        return replace(self, score=score)


@dataclass(frozen=True)
class Citation:
    """A cited page, linked to the document it belongs to. Frozen so it can go in a set."""

    doc_id: str
    page: int


@dataclass
class Answer:
    """Pipeline output for one question."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    verdict: Verdict | None = None
    confidence: float | None = None
    chunk_ids: list[str] = field(default_factory=list)  # chunks the SLM was given
    timings_ms: dict[str, float] = field(default_factory=dict)  # per-stage latency
    details: dict[str, Any] = field(default_factory=dict)  # debugging info: tags, widened, SLM counters
