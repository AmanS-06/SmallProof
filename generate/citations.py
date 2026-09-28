"""Finds page citations in an SLM answer and checks them against the chunks it was given.

The answer prompt (Phase 2) asks the SLM to cite pages like [p. 12]. Small
models do not always follow the format exactly, so these are all accepted:

    [p. 12]  [p.12]  [p 12]  [page 12]  [pp. 12-13]  [pages 12, 14]  (p. 12)

A citation is valid only if one of the given chunks covers that page. An
invalid citation means the SLM cited a page it never saw.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from core.types import Chunk, Citation

# Opening bracket or parenthesis, a page word, then the page list, then a closing bracket.
# The page list may hold digits, commas, spaces, hyphens and en dashes (U+2013).
_CITATION_PATTERN = re.compile(
    r"[\[(]\s*(?:pages?|pp?\.?)\s*([0-9][0-9,\s\-\u2013]*?)\s*[\])]",
    re.IGNORECASE,
)
_RANGE_SEPARATOR = re.compile(r"\s*[-\u2013]\s*")

# A cited range longer than this is treated as malformed rather than expanded.
MAX_RANGE_PAGES = 20


@dataclass
class CitationCheck:
    """Result of checking one answer's citations."""

    citations: list[Citation]  # cited pages that a given chunk covers, with their document
    invalid_pages: list[int]  # cited pages that no given chunk covers

    @property
    def all_valid(self) -> bool:
        return not self.invalid_pages


def _parse_page_list(page_text: str) -> list[int]:
    """Turn '12', '12-13' or '12, 14' into a list of page numbers. Malformed parts are skipped."""
    pages: list[int] = []
    for part in page_text.split(","):
        part = part.strip()
        if not part:
            continue
        bounds = _RANGE_SEPARATOR.split(part)
        if len(bounds) == 1 and bounds[0].isdigit():
            pages.append(int(bounds[0]))
        elif len(bounds) == 2 and all(bound.isdigit() for bound in bounds):
            start, end = int(bounds[0]), int(bounds[1])
            if start <= end <= start + MAX_RANGE_PAGES:
                pages.extend(range(start, end + 1))
    return pages


def extract_cited_pages(answer_text: str) -> list[int]:
    """Every page number cited in the answer, in order of first appearance, without repeats."""
    pages: list[int] = []
    for match in _CITATION_PATTERN.finditer(answer_text):
        for page in _parse_page_list(match.group(1)):
            if page not in pages:
                pages.append(page)
    return pages


def check_citations(answer_text: str, chunks: Sequence[Chunk]) -> CitationCheck:
    """Match each cited page to the chunks the SLM was given.

    If chunks from two documents both cover a cited page number, the citation
    is linked to both documents, because a page number alone cannot tell
    them apart.
    """
    citations: list[Citation] = []
    invalid_pages: list[int] = []
    for page in extract_cited_pages(answer_text):
        doc_ids = sorted({chunk.doc_id for chunk in chunks if page in chunk.pages})
        if doc_ids:
            citations.extend(Citation(doc_id=doc_id, page=page) for doc_id in doc_ids)
        else:
            invalid_pages.append(page)
    return CitationCheck(citations=citations, invalid_pages=invalid_pages)
