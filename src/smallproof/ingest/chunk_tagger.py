"""Tags each chunk as table, definition, numeric or narrative.

Two backends:
- zeroshot: the Jev-style classifier (ModernBERT NLI) with descriptive labels.
- rules: fast heuristics (share of numeric tokens, definition phrases). Used
  when classifying every chunk at ingest is too slow on CPU.
Chunks from the table parser are always tagged "table" without a model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace

from smallproof.core.interfaces import Classifier
from smallproof.core.types import Chunk

CHUNK_TYPES = ("table", "definition", "numeric", "narrative")
_DEFINITION = re.compile(r"\b(is defined as|are defined as|means|refers to|we define|is the term)\b", re.IGNORECASE)
_NUMBER = re.compile(r"^[($-]*\d[\d,.]*%?\)?$")


def numeric_share(text: str) -> float:
    """Share of whitespace tokens that are numbers like 1,234 or (5.6) or 12%."""
    tokens = text.split()
    return sum(1 for token in tokens if _NUMBER.match(token)) / len(tokens) if tokens else 0.0


def rule_chunk_type(text: str) -> str:
    lines = [line for line in text.split("\n") if line.strip()]
    rows_with_numbers = sum(1 for line in lines if sum(bool(_NUMBER.match(t)) for t in line.split()) >= 2)
    share = numeric_share(text)
    if " | " in text or (len(lines) >= 4 and rows_with_numbers / len(lines) >= 0.4):
        return "table"
    if _DEFINITION.search(text) and share < 0.05:
        return "definition"
    if share >= 0.06:
        return "numeric"
    return "narrative"


def tag_with_rules(chunks: Sequence[Chunk]) -> list[Chunk]:
    """Rule tags have no probability, so type_score stays None."""
    return [chunk if chunk.chunk_type else replace(chunk, chunk_type=rule_chunk_type(chunk.text)) for chunk in chunks]


def tag_with_classifier(chunks: Sequence[Chunk], classifier: Classifier, labels: dict[str, str]) -> list[Chunk]:
    """labels maps a chunk type to its descriptive label, for example "table" -> "a table of figures"."""
    type_of = {description: name for name, description in labels.items()}
    tagged = []
    for chunk in chunks:
        if chunk.chunk_type:  # already known, for example from the table parser
            tagged.append(chunk)
            continue
        best = classifier.classify(chunk.text, list(labels.values()))[0]
        tagged.append(replace(chunk, chunk_type=type_of[best.label], type_score=round(best.score, 4)))
    return tagged
