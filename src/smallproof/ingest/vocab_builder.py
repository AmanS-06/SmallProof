"""Builds the metadata vocabulary of a corpus and adds metadata to each chunk.

Per chunk: the document's metadata (company, year, type from the pack's
manifest), the years mentioned in the text, and the section it belongs to.
Section headings are found with the pack's regex patterns and carried forward
page by page, because a section runs until the next heading.

The vocabulary lists every value seen, so query-time tags are picked from
values that actually exist in the corpus.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from smallproof.core.types import Chunk

YEAR_PATTERN = re.compile(r"(?<!\d)(19[89]\d|20[0-4]\d)(?!\d)")  # also finds the year in "FY2018"


def years_in(text: str) -> list[str]:
    return sorted(set(YEAR_PATTERN.findall(text)))


def compile_section_patterns(patterns: Sequence[str]) -> list[re.Pattern]:
    return [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns]


def normalize_section(heading: str) -> str:
    return re.sub(r"\s+", " ", heading).strip(" .:").lower()[:80]


def find_section(text: str, patterns: Sequence[re.Pattern]) -> str | None:
    """The last section heading in the text, or None. The last one wins because it is still open."""
    last: tuple[int, str] | None = None
    for pattern in patterns:
        for match in pattern.finditer(text):
            heading = match.group(1) if match.groups() else match.group(0)
            if last is None or match.start() >= last[0]:
                last = (match.start(), normalize_section(heading))
    return last[1] if last else None


def add_metadata(chunks: Sequence[Chunk], doc_meta: dict[str, dict[str, Any]], section_patterns: Sequence[str]) -> list[Chunk]:
    """Return chunks with metadata: doc fields, years (comma string, Chroma needs flat values) and section."""
    patterns = compile_section_patterns(section_patterns)
    current_section: dict[str, str] = {}
    result = []
    for chunk in chunks:  # chunks are in page order within each document
        headings = sum(len(pattern.findall(chunk.text)) for pattern in patterns)
        # A chunk listing 3+ headings is a table of contents: it must not change the current section.
        section_here = find_section(chunk.text, patterns) if headings < 3 else None
        section = current_section.get(chunk.doc_id, "")
        if section_here:
            # A heading near the top of the chunk labels the whole chunk; later headings apply from now on.
            first = find_section(chunk.text[:200], patterns)
            section = first or section
            current_section[chunk.doc_id] = section_here
        metadata = {**{k: str(v) for k, v in doc_meta.get(chunk.doc_id, {}).items() if not isinstance(v, list)},
                    "years": ",".join(years_in(chunk.text)), "section": section}
        result.append(replace(chunk, metadata={**chunk.metadata, **metadata}))
    return result


def build_vocab(chunks: Sequence[Chunk], doc_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Every metadata value seen in the corpus, with how many chunks carry it."""
    years: Counter = Counter()
    sections: Counter = Counter()
    for chunk in chunks:
        years.update(year for year in chunk.metadata.get("years", "").split(",") if year)
        if chunk.metadata.get("section"):
            sections[chunk.metadata["section"]] += 1
    fields: dict[str, set] = {}
    aliases: dict[str, str] = {}  # other names for a company (a ticker, a short name) -> its manifest name
    for meta in doc_meta.values():
        for key, value in meta.items():
            if key == "aliases":
                aliases.update({str(alias): str(meta["company"]) for alias in value})
            elif not isinstance(value, list):
                fields.setdefault(key, set()).add(str(value))
    return {
        "years": dict(sorted(years.items())),
        "sections": dict(sections.most_common()),
        "doc_fields": {key: sorted(values) for key, values in fields.items()},
        "aliases": dict(sorted(aliases.items())),
        "doc_meta": doc_meta,
    }
