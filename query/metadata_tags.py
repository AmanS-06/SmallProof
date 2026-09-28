"""Query-time metadata tags and soft boosts.

Tags are only picked from values that exist in the ingest vocabulary:
- company: an exact mention of a known company, else GLiNER company spans
  fuzzy-matched to known companies (confidence = similarity x GLiNER score)
- year: a year in the question that is also a document year (latest wins)
- section: the zero-shot classifier picks one of the pack's section tags
- intent: the zero-shot classifier picks one of the pack's intents (routing)

Tags never remove chunks. A chunk that matches k confident tags gets its score
multiplied by (1 + boost * k). Tags below the confidence threshold are ignored,
which falls back to plain unfiltered search.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from core.types import Chunk, Entity

YEAR = re.compile(r"(?<!\d)(19[89]\d|20[0-4]\d)(?!\d)")


@dataclass
class QueryTags:
    company: tuple[str, float] | None = None
    year: tuple[str, float] | None = None
    section: tuple[str, float] | None = None
    intent: tuple[str, float] | None = None
    entities: list[Entity] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"company": self.company, "year": self.year, "section": self.section, "intent": self.intent,
                "entities": [tuple(entity) for entity in self.entities]}


def match_company(query: str, companies: Sequence[str], entities: Sequence[Entity]) -> tuple[str, float] | None:
    lowered = query.lower()
    for name in sorted(companies, key=len, reverse=True):  # longest first: "American Express" before "Express"
        if re.search(rf"(?<!\w){re.escape(name.lower())}(?!\w)", lowered):
            return name, 1.0
    best: tuple[str, float] | None = None
    for entity in entities:
        if entity.type != "company":
            continue
        for name in companies:
            confidence = SequenceMatcher(None, entity.span.lower(), name.lower()).ratio() * entity.score
            if best is None or confidence > best[1]:
                best = (name, round(confidence, 3))
    return best


def match_year(query: str, doc_years: Sequence[str]) -> tuple[str, float] | None:
    years = sorted(set(YEAR.findall(query)) & set(doc_years))
    return (years[-1], 1.0) if years else None


def matches(chunk: Chunk, tags: QueryTags, min_confidence: float, section_synonyms: dict[str, list[str]],
            intent_types: dict[str, list[str]]) -> int:
    """How many confident tags this chunk matches."""
    meta = chunk.metadata
    count = 0
    if tags.company and tags.company[1] >= min_confidence and meta.get("company") == tags.company[0]:
        count += 1
    if tags.year and tags.year[1] >= min_confidence and meta.get("year") == tags.year[0]:
        count += 1
    if tags.section and tags.section[1] >= min_confidence:
        section = meta.get("section", "")
        if any(word in section for word in section_synonyms.get(tags.section[0], [])):
            count += 1
    if tags.intent and tags.intent[1] >= min_confidence and chunk.chunk_type in intent_types.get(tags.intent[0], []):
        count += 1
    return count


def boost_factors(chunks: Sequence[Chunk], tags: QueryTags, boost: float, min_confidence: float,
                  section_synonyms: dict[str, list[str]], intent_types: dict[str, list[str]]) -> dict[str, float]:
    """Score multiplier per chunk id."""
    return {chunk.id: 1 + boost * matches(chunk, tags, min_confidence, section_synonyms, intent_types) for chunk in chunks}
