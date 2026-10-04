"""Table extraction with pdfplumber.

Each table becomes plain text with one row per line and cells joined by " | ".
A long table is split by rows, with the header row repeated in every piece,
so a table is never cut in the middle of a row.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

Table = list[list[str]]


def clean_table(rows: list[list[str | None]]) -> Table:
    """Collapse whitespace, turn empty cells into "", drop empty rows and columns."""
    cleaned = [[re.sub(r"\s+", " ", cell or "").strip() for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return []
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    keep = [col for col in range(width) if any(row[col] for row in cleaned)]
    return [[row[col] for col in keep] for row in cleaned]


def row_text(row: list[str]) -> str:
    return " | ".join(row)


def extract_tables(pdf_path: str | Path) -> dict[int, list[Table]]:
    """Tables found on each page, keyed by 1-based page number. Pages without tables are left out."""
    import pdfplumber

    found: dict[int, list[Table]] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            tables = [clean_table(rows) for rows in page.extract_tables()]
            tables = [table for table in tables if len(table) >= 2]  # a header alone is not a table
            if tables:
                found[number] = tables
            page.close()
    return found


def table_to_texts(table: Table, max_tokens: int, count_tokens: Callable[[str], int]) -> list[str]:
    """Split a table into text pieces under max_tokens, repeating the header row in each piece."""
    header, body = row_text(table[0]), [row_text(row) for row in table[1:]]
    pieces, current = [], [header]
    for line in body:
        if len(current) > 1 and count_tokens("\n".join(current + [line])) > max_tokens:
            pieces.append("\n".join(current))
            current = [header]
        current.append(line)
    pieces.append("\n".join(current))
    return pieces
