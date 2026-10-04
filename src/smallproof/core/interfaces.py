"""Abstract base classes for every swappable stage.

Each backend (for example the ModernBERT classifier or the Ollama generator)
subclasses one of these. The pipeline only talks to these interfaces, so a
backend can be swapped through the config without touching the core.
The method signatures follow the project brief.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from smallproof.core.types import Answer, Chunk, Entity, LabelScore, Verdict


class Classifier(ABC):
    """Zero-shot classifier: scores a text against a list of labels."""

    @abstractmethod
    def classify(self, text: str, labels: list[str]) -> list[LabelScore]:
        """Return one (label, score) per label, highest score first."""


class Extractor(ABC):
    """Finds keyword and entity spans in a text."""

    @abstractmethod
    def extract(self, text: str, entity_types: list[str]) -> list[Entity]:
        """Return the spans found in text, each with its type and score."""


class Retriever(ABC):
    """Finds candidate chunks for a query."""

    @abstractmethod
    def retrieve(self, query: str, filters: dict[str, Any] | None, k: int) -> list[Chunk]:
        """Return up to k chunks, best first, each with .score set.

        filters are metadata hints, for example {"year": "2018"}. The brief treats
        them as soft boosts: a backend may reorder with them, but should not drop
        a chunk only because it does not match.
        """


class Reranker(ABC):
    """Reorders candidate chunks with a stronger (slower) model."""

    @abstractmethod
    def rerank(self, query: str, chunks: list[Chunk], k: int) -> list[Chunk]:
        """Return the k best chunks for the query, best first, with new scores."""


class Gate(ABC):
    """Evidence gate: is there enough evidence to answer?"""

    @abstractmethod
    def check(self, query: str, chunks: list[Chunk]) -> tuple[Verdict, float]:
        """Return (verdict, confidence) for answering the query from these chunks."""


class Generator(ABC):
    """Writes the final answer with the SLM."""

    @abstractmethod
    def generate(self, query: str, chunks: list[Chunk], template: str) -> Answer:
        """Write an answer to the query from the chunks, using a prompt template."""
