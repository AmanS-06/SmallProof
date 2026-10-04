"""Splits page text into chunks of about target_tokens, with overlap.

Chunks never cross a page, so every chunk has one exact page for citations.
A page is cut into units (lines, and sentences for long lines). Units are
added to a chunk until the next one would pass target_tokens. The next chunk
starts with the last units of the previous one, up to overlap_tokens.

count_tokens is passed in, so the chunker works with any tokenizer. In the
pipeline it is the embedder's tokenizer, so chunks fit the embedder.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from smallproof.core.types import Chunk

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def split_units(page_text: str, count_tokens: Callable[[str], int], max_tokens: int) -> list[str]:
    """Lines, then sentences for lines that are too long, then word windows as a last resort."""
    units: list[str] = []
    for line in (line.strip() for line in page_text.split("\n")):
        if not line:
            continue
        if count_tokens(line) <= max_tokens:
            units.append(line)
            continue
        for sentence in _SENTENCE_END.split(line):
            if count_tokens(sentence) <= max_tokens:
                units.append(sentence)
                continue
            words = sentence.split()
            step = max(1, len(words) * max_tokens // max(1, count_tokens(sentence)))  # words per window
            units.extend(" ".join(words[i : i + step]) for i in range(0, len(words), step))
    return units


def chunk_page(
    doc_id: str,
    page: int,
    text: str,
    count_tokens: Callable[[str], int],
    target_tokens: int = 350,
    overlap_tokens: int = 50,
    min_tokens: int = 25,
) -> list[Chunk]:
    """Chunk one page. Chunk ids look like '3M_2018_10K:p12:c0'."""
    units = split_units(text, count_tokens, target_tokens)
    sizes = [count_tokens(unit) for unit in units]
    chunks: list[Chunk] = []
    start = 0
    while start < len(units):
        end, total = start, 0
        while end < len(units) and (end == start or total + sizes[end] <= target_tokens):
            total += sizes[end]
            end += 1
        if total >= min_tokens:  # drops near-empty pieces such as a lone page number
            chunks.append(Chunk(id=f"{doc_id}:p{page}:c{len(chunks)}", doc_id=doc_id,
                                text="\n".join(units[start:end]), page_start=page, page_end=page))
        if end >= len(units):
            break
        # Step back over the last units to create the overlap, but always move forward.
        back, overlap = end, 0
        while back - 1 > start and overlap + sizes[back - 1] <= overlap_tokens:
            back -= 1
            overlap += sizes[back]
        start = back
    return chunks


def chunk_document(
    doc_id: str,
    pages: list[str],
    count_tokens: Callable[[str], int],
    target_tokens: int = 350,
    overlap_tokens: int = 50,
    min_tokens: int = 25,
) -> list[Chunk]:
    """Chunk every page of a document. pages[0] is page 1. Blank pages give no chunks."""
    chunks: list[Chunk] = []
    for page_number, text in enumerate(pages, start=1):
        if text.strip():
            chunks.extend(chunk_page(doc_id, page_number, text, count_tokens, target_tokens, overlap_tokens, min_tokens))
    return chunks
