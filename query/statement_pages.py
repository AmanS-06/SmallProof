"""Primary statement pages: a reserved place in the final chunks.

Many financial questions need one of a filing's primary statements (balance
sheet, income statement, cash flow statement) while naming only a concept on
it: "quick ratio", "capex", "operating margin". Neither BM25 nor the dense
index connects the two well, so the statement page often ranks below the top
five even when the right filing was found (FinanceBench test: evidence found
in the right filing for 99 of 100 questions, but in the top 5 for only 46).

This module finds each filing's statement pages once, from the chunks
already in the index (the first chunk of a page whose opening lines are a
statement title and that is full of numbers), and picks which statement a
question needs from keywords in the pack config. The pipeline then gives up
to a few of those pages a reserved slot after the best reranked chunks.

Everything domain specific (titles, keywords) lives in the pack config.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from core.types import Chunk

# Lines that often come before a statement title and are not the title.
_NOISE_LINE = re.compile(r"^(table of contents|index|item \d+[a-c]?\.?|part [iv]+|\d+)$", re.I)
_NUMBER = re.compile(r"\d[\d,]{2,}")


@dataclass
class StatementIndex:
    """(company, year, kind) -> first chunk of each statement page, page order."""

    pages: dict[tuple[str, str, str], list[Chunk]] = field(default_factory=dict)

    def lookup(self, company: str, year: str, kinds: Iterable[str]) -> list[Chunk]:
        found: list[Chunk] = []
        for kind in kinds:
            found.extend(self.pages.get((company, year, kind), []))
        return found


def _title_patterns(titles: dict[str, str]) -> dict[str, re.Pattern[str]]:
    # A title is a whole line, optionally with punctuation around it.
    return {kind: re.compile(r"^\W*(?:" + pattern + r")\W*$", re.I) for kind, pattern in titles.items()}


def statement_kind(text: str, titles: dict[str, re.Pattern[str]], min_numbers: int, title_lines: int = 4) -> str | None:
    """The kind of statement a page starts with, or None for any other page."""
    lines = [line.strip() for line in text.splitlines()[: title_lines * 2] if line.strip()]
    lines = [line for line in lines if not _NOISE_LINE.match(line)][:title_lines]
    if len(_NUMBER.findall(text)) < min_numbers:
        return None
    for kind, pattern in titles.items():
        if any(pattern.match(line) for line in lines):
            return kind
    return None


def build_statement_index(chunks: Sequence[Chunk], titles: dict[str, str], min_numbers: int = 15) -> StatementIndex:
    """Find the statement pages among the first chunk of every page."""
    patterns = _title_patterns(titles)
    pages: dict[tuple[str, str, str], list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        if not chunk.id.endswith(":c0"):
            continue  # only a page's first chunk can start with its title
        kind = statement_kind(chunk.text, patterns, min_numbers)
        company, year = chunk.metadata.get("company"), chunk.metadata.get("year")
        if kind and company and year:
            pages[(company, year, kind)].append(chunk)
    for found in pages.values():
        found.sort(key=lambda c: c.page_start)
    return StatementIndex(dict(pages))


def question_kinds(question: str, keywords: dict[str, Sequence[str]]) -> list[str]:
    """The statements a question needs, from the pack's keyword lists."""
    lowered = question.lower()
    return [kind for kind, words in keywords.items()
            if any(re.search(r"(?<!\w)" + re.escape(word.lower()), lowered) for word in words)]


def reserve_slots(ranked: Sequence[Chunk], extra: Sequence[Chunk], keep: int, top_k: int) -> list[Chunk]:
    """The best `keep` ranked chunks, then the extra pages, then the rest of
    the ranking, without repeats, cut to top_k."""
    ordered = list(ranked[:keep]) + list(extra) + list(ranked[keep:])
    seen: set[str] = set()
    final: list[Chunk] = []
    for chunk in ordered:
        if chunk.id not in seen:
            seen.add(chunk.id)
            final.append(chunk)
    return final[:top_k]
