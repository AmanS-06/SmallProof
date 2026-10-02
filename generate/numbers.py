"""Finds the numbers in a piece of text, with their position, unit and precision.

Used by the answer verifier (generate/verify.py) to check that every number an
SLM writes comes from the passages or from a calculation that code can redo.

    "$1,577 million"  -> value 1,577, scale 1e6, 0 decimals
    "(660)"           -> value -660 (accounting negative)
    "18.36%"          -> value 18.36, percent, 2 decimals

Not treated as numbers: years (2018), numbers glued to letters (FY2018, Q2,
10-K, 3M), page citations ([p. 60]) and passage markers ([1]), and small
whole numbers without a unit (list markers, "3 year average").
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_NUMBER = re.compile(
    r"(?P<sign>[-−])?(?P<currency>\$\s?)?(?P<open>\()?(?:\$\s?)?"
    r"(?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?P<close>\))?"
    r"(?:\s?(?P<unit>%|percent\b|billion\b|million\b|thousand\b|bn\b|mn\b|mm\b|[BMK]\b))?",
    re.IGNORECASE,
)
_SCALE = {"billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "mn": 1e6, "mm": 1e6, "m": 1e6,
          "thousand": 1e3, "k": 1e3}
# Citations like [p. 60], [pp. 12-13], (page 4) and passage markers like [1] or [2, 3].
# In parentheses only with a page word: "(1,577)" is an accounting negative.
_PAGES = r"\d[\d,\s\-–]*(?:,\s*(?:pages?|pp?\.?)\s*\d+)*\s*"
_CITATION = re.compile(r"\[\s*(?:(?:pages?|pp?\.?)\s*)?" + _PAGES + r"\]|\(\s*(?:pages?|pp?\.?)\s*" + _PAGES + r"\)"
                       r"|\[\s*(?:pages?|pp?\.?)\s*\d[^\]]*$"  # a citation cut off at the end of the answer
                       r"|\b(?:on|from|see|in)\s+(?:pages?|p\.)\s*\d+(?:\s*(?:and|,|-)\s*\d+)*",  # "from page 68"
                       re.IGNORECASE)
_ORDINAL = re.compile(r"(\d)(?:st|nd|rd|th)\b")
SMALL_INT = 12  # whole numbers up to this, with no unit or currency, are skipped


@dataclass(frozen=True)
class Number:
    start: int  # position in the text
    end: int
    raw: str  # as written, for example "$1,577 million"
    digits: str  # the digits part, for example "1,577"
    value: float  # signed, without the scale word
    decimals: int  # digits after the decimal point, as written
    scale: float = 1.0  # 1e6 for "million"
    percent: bool = False
    currency: bool = False

    @property
    def magnitude(self) -> float:
        return abs(self.value)


def mask_citations(text: str) -> str:
    """Replace citations with spaces, so positions in the text stay the same."""
    return _CITATION.sub(lambda m: " " * len(m.group(0)), text)


def _is_year(digits: str) -> bool:
    return digits.isdigit() and len(digits) == 4 and 1900 <= int(digits) <= 2099


def find_numbers(text: str, skip_small: bool = True) -> list[Number]:
    """Every number in the text that could be an amount, in order.
    Ordinals count as numbers ("61st" gives 61, keeping its position)."""
    masked = _ORDINAL.sub(lambda m: m.group(1) + " " * (len(m.group(0)) - 1), mask_citations(text))
    numbers = []
    for match in _NUMBER.finditer(masked):
        digits, unit = match.group("digits"), (match.group("unit") or "").lower()
        start, end = match.span()
        if unit in ("b", "m", "k") and not match.group("currency"):
            continue  # "10K" is a filing, "$10K" is an amount
        before = masked[start - 1] if start else " "
        after = masked[end] if end < len(masked) else " "
        if before.isalpha() or before in "_" or (after.isalpha() and not unit):
            continue  # FY2018, Q2, 3M, 10x
        if after == "-" and end + 1 < len(masked) and masked[end + 1].isalpha():
            continue  # 10-K, 8-K
        if before == "-" and start >= 2 and masked[start - 2].isalpha():
            continue  # COVID-19
        has_unit = bool(unit) or bool(match.group("currency"))
        if _is_year(digits) and not has_unit and "," not in digits:
            continue
        plain = digits.replace(",", "")
        value = float(plain)
        decimals = len(plain.split(".")[1]) if "." in plain else 0
        if skip_small and not has_unit and decimals == 0 and value <= SMALL_INT:
            continue
        negative = bool(match.group("sign")) or bool(match.group("open") and match.group("close"))
        raw = match.group(0).strip()
        if match.group("open") and not match.group("close"):  # "(2,112 / ..." opens an expression, not a negative
            raw = raw.lstrip("(").strip()
            start = start + match.group(0).index(raw)
        elif match.group("close") and not match.group("open"):  # "... + 116) / 3" closes an expression
            raw = raw.replace(")", "").strip()
        numbers.append(Number(start=start, end=end, raw=raw, digits=digits, value=-value if negative else value,
                              decimals=decimals, scale=_SCALE.get(unit, 1.0), percent=unit in ("%", "percent"),
                              currency=bool(match.group("currency"))))
    return numbers


def tolerance(number: Number, slack: float = 0.0) -> float:
    """How far a true value may be from the written one: half a unit in the
    last written digit (rounding), or `slack` as a share of the value,
    whichever is larger. Copied numbers get no slack; calculated ones get a
    little, because small models round intermediate steps loosely."""
    return max(0.5 * 10 ** -number.decimals, slack * number.magnitude) * (1 + 1e-9)


# A table "in millions" writes 1,577 where an answer writes $1.577 billion or
# $1,577,000 thousand: values are compared up to factors of 1,000.
SCALE_STEPS = (1.0, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9)


def same_amount(written: Number, value: float, allow_percent: bool = False, slack: float = 0.0) -> bool:
    """Is `value` (from a table or a calculation) what `written` says, allowing
    for rounding, a sign written as a word ("a decrease of 5") and a change
    of scale? With allow_percent, 0.153 also matches 15.3%.
    A change of scale needs a unit or currency in the written number: a bare
    5.12 is not 5,121.3 thousandths."""
    target, tol = written.magnitude, tolerance(written, slack)
    steps = SCALE_STEPS if written.scale != 1.0 or written.currency else (1.0,)
    candidates = [abs(value) * step for step in steps]
    if allow_percent:
        candidates += [abs(value) * 100, abs(value) / 100]
    return any(abs(target - candidate) <= tol for candidate in candidates if math.isfinite(candidate))
