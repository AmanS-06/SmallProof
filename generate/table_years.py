"""Label each number in a multi-year table with its year, before the SLM sees it.

PDF text of a financial statement puts the column headers (the years) on their
own lines above the rows, and each row ends with one value per year:

    Years Ended
    December 2,
    2022
    December 3,
    2021
    Total revenue 17,606 15,785 12,868

A small model often takes a value from the wrong column. This rewrites such a
row as

    Total revenue (2022: 17,606; 2021: 15,785; 2020: 12,868)

It only changes a row whose trailing values match the number of years exactly,
and only when at least a few rows do, so ordinary text passes through
unchanged. Negative numbers in parentheses become a minus sign.
"""

from __future__ import annotations

import re

YEAR = re.compile(r"(?<!\d)(19[89]\d|20[0-4]\d)(?!\d)")
# One value: $ 1,234.5, (660), 12.5%, or a dash meaning nothing.
VALUE = re.compile(r"^\(?-?\d(?:[\d,]*\d)?(\.\d+)?\)?%?$|^[\u2014\u2013-]$")
MIN_ROWS = 3  # fewer matching rows than this: probably not a table, leave it
MAX_HEADER_LINES = 25


def _header_years(lines: list[str]) -> tuple[list[str], int]:
    """Years named above the first row of numbers, in order, and that row's index."""
    years: list[str] = []
    for index, line in enumerate(lines[:MAX_HEADER_LINES]):
        values = _trailing_values(line)
        # A header like "May 31, 2020 May 26, 2019" ends in years, not amounts.
        if len(values) >= 2 and not any(YEAR.fullmatch(value) for value in values):
            return years, index
        for year in YEAR.findall(line):
            if year not in years:
                years.append(year)
    return years, len(lines)


def _trailing_values(line: str) -> list[str]:
    """The value tokens at the end of a line ("$" signs dropped), in order."""
    tokens = [token for token in line.split() if token != "$"]
    values: list[str] = []
    for token in reversed(tokens):
        cleaned = token.lstrip("$")
        if not VALUE.match(cleaned):
            break
        values.append(cleaned)
    return list(reversed(values))


def _plain(value: str) -> str:
    """(660) -> -660; other values unchanged."""
    if value.startswith("(") and value.endswith(")"):
        return "-" + value[1:-1]
    return value


def _labelled_row(line: str, years: list[str]) -> str | None:
    values = _trailing_values(line)
    extra = values[: len(values) - len(years)]
    # Extra values before the columns are fine only if they are years in the
    # label ("Notes due 2025 1,500 1,500"); anything else is another layout.
    if len(values) < len(years) or not all(YEAR.fullmatch(value) for value in extra):
        return None
    values = values[len(extra):]
    label = line
    for value in reversed(values):  # cut the values off the end of the line
        label = label[: label.rstrip().rfind(value)].rstrip().rstrip("$").rstrip()
    if not re.search(r"[A-Za-z]", label):
        return None
    pairs = "; ".join(f"{year}: {_plain(value)}" for year, value in zip(years, values))
    return f"{label} ({pairs})"


def annotate_table_years(text: str) -> str:
    """The text with year labels on the values of a multi-year table, or the
    text unchanged when it does not look like one."""
    lines = text.splitlines()
    years, first_row = _header_years(lines)
    if len(years) < 2 or len(years) > 5:
        return text
    rewritten = list(lines)
    changed = 0
    for index in range(first_row, len(lines)):
        labelled = _labelled_row(lines[index], years)
        if labelled is not None:
            rewritten[index] = labelled
            changed += 1
    return "\n".join(rewritten) if changed >= MIN_ROWS else text
