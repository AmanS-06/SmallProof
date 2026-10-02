"""Checks an SLM answer after it is written. Code, not the model, has the last word.

A small model reads passages well but makes three kinds of mistakes that code
can catch:
1. Arithmetic slips: "5,121.3 / 7,491.5 = 5.12". Code redoes every written
   calculation ("a op b = c") and corrects the result when it is wrong.
2. Numbers from nowhere: every number in the answer must be in the passages,
   in the question, or be the result of a calculation on such numbers.
3. The wrong column: when a sentence names one year, a value copied from a
   multi-year table must come from that year's column.

It also refuses answers that hedge ("NOT FOUND" or "cannot be calculated"
somewhere inside) or were cut off by the token limit.

The checks know nothing about finance: they work on numbers, years and
arithmetic, so they apply to any domain pack. A pack can add hedge phrases.

    result = verify_answer(answer_text, chunks, question)
    result.ok        # False: refuse instead of answering
    result.text      # the answer, with any corrected calculations
    result.reasons   # why it failed, in plain words
    result.numbers   # one line per number: where it came from
"""

from __future__ import annotations

import ast
import itertools
import operator
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from core.types import Chunk
from generate.numbers import Number, find_numbers, mask_citations, same_amount
from generate.table_years import table_value_years

DEFAULT_HEDGES = ["not found", "cannot be calculated", "cannot be determined", "can not be calculated",
                  "not possible to calculate", "not possible to determine", "insufficient information",
                  "not enough information", "is not provided", "are not provided", "not fully supported",
                  "does not specify", "do not specify", "does not provide", "do not provide"]
CALC_SLACK = 0.005  # small models round the steps of a calculation loosely: allow 0.5 percent
MAX_CORRECTIONS = 3
_CONSTANTS = {100.0, 1000.0}  # "x 100" turns a ratio into a percent; not data that needs a source
_EQUALS = re.compile(r"=|≈|~")
_SCALE_WORDS = re.compile(r"\b(?:million|billion|thousand|bn|mn|mm)\b", re.IGNORECASE)
_EXPR_CHARS = set("0123456789.,$%()+-*/ ×÷−\t")
_AFTER_EQUALS = re.compile(r"^[\s$(]*(?:approximately|approx\.?|about|roughly|around)?[\s$(]*$", re.IGNORECASE)
_YEAR_WORD = re.compile(r"\b(?:FY\s?)?((?:19|20)\d{2})\b|\bFY\s?'?(\d{2})\b", re.IGNORECASE)
_QUARTER = re.compile(r"\bQ[1-4]\b|quarter|months ended|six months|nine months|three months", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])|\n+")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


@dataclass
class NumberCheck:
    raw: str  # the number as written in the answer
    status: str  # quoted, question, constant, calculated, derived, corrected, unsupported, wrong_year
    note: str = ""  # where it came from: a page, a calculation, or what is wrong
    start: int = 0  # position in Verification.text, for highlighting
    end: int = 0


@dataclass
class Verification:
    ok: bool
    text: str  # the answer after corrections (unchanged when there were none)
    reasons: list[str] = field(default_factory=list)
    numbers: list[NumberCheck] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# Section: safe arithmetic


def _evaluate(expression: str) -> float | None:
    """Value of an arithmetic expression with + - * / and parentheses, or None."""
    cleaned = (expression.replace("$", "").replace(",", "").replace("%", "")
               .replace("×", "*").replace("÷", "/").replace("−", "-"))
    try:
        tree = ast.parse(cleaned.strip(), mode="eval")
    except SyntaxError:
        return None

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError("not plain arithmetic")

    try:
        has_operator = any(isinstance(node, ast.BinOp) for node in ast.walk(tree))
        return walk(tree) if has_operator else None
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def _expression_before(text: str, position: int) -> tuple[int, str] | None:
    """The arithmetic just before an equals sign, as (start, expression)."""
    start = position
    while start > 0 and text[start - 1] in _EXPR_CHARS:
        start -= 1
    expression = text[start:position]
    # Trim from the left until parentheses balance and it starts with a number or "(".
    while expression and (expression.count("(") < expression.count(")") or not re.match(r"\s*[\d($.\-−]", expression)):
        expression = expression[1:]
        start += 1
    while expression.count("(") > expression.count(")"):  # an unclosed "(" opened earlier text
        cut = expression.index("(") + 1
        expression, start = expression[cut:], start + cut
    if not _has_operator(expression):
        return None
    return start, expression


def _has_operator(expression: str) -> bool:
    return bool(re.search(r"[\d)]\s*[-+*/×÷−]\s*[\d($]", expression.replace("%", "")))


def _expression_after(text: str, position: int) -> tuple[int, str] | None:
    """Arithmetic right after an equals sign, up to the next equals sign, when it
    is a middle step of a chain ("= 402 / 3 = 134"). Returns (end, expression)."""
    end = position
    while end < len(text) and text[end] in _EXPR_CHARS:
        end += 1
    if end >= len(text) or not _EQUALS.match(text[end]):
        return None
    expression = text[position:end]
    if not _has_operator(expression) or expression.count("(") != expression.count(")"):
        return None
    return end, expression


def _is_constant(number: Number) -> bool:
    return number.value in _CONSTANTS and number.scale == 1.0 and not number.currency and not number.percent


# Section: formatting a corrected value


def _format_like(value: float, written: Number) -> str:
    decimals = written.decimals
    if decimals == 0 and abs(value) < 10:
        decimals = 2  # "5" for 0.68 would hide the correction
    text = f"{abs(value):,.{decimals}f}" if "," in written.digits or abs(value) >= 10_000 else f"{abs(value):.{decimals}f}"
    return text


def _replace_number(text: str, digits: str, new_digits: str) -> str:
    """Replace a number written as `digits` wherever it stands alone."""
    pattern = re.compile(r"(?<![\d.,])" + re.escape(digits) + r"(?![\d]|[.,]\d)")
    return pattern.sub(new_digits, text)


# Section: the checks


class _Context:
    """Numbers the answer may use: from the passages (with their page) and from the question."""

    def __init__(self, chunks: Sequence[Chunk], question: str) -> None:
        self.passage_numbers = [(number, chunk.page_start) for chunk in chunks
                                for number in find_numbers(chunk.text, skip_small=False)]
        self.question_numbers = find_numbers(question, skip_small=False)
        self.year_values = [(value, year) for chunk in chunks for value, year in table_value_years(chunk.text)]

    def quoted_page(self, written: Number) -> int | None:
        for number, page in self.passage_numbers:
            if same_amount(written, number.value):
                return page
        return None

    def in_question(self, written: Number) -> bool:
        return any(same_amount(written, number.value) for number in self.question_numbers)


def _years_named(sentence: str) -> set[str]:
    years = set()
    for long_form, short_form in _YEAR_WORD.findall(sentence):
        years.add(long_form or f"20{short_form}")
    return years


def _year_problem(written: Number, sentence: str, context: _Context) -> str | None:
    """A value copied from a multi-year table must sit in the column of the one
    year its sentence names. Returns what is wrong, or None."""
    if _QUARTER.search(sentence):
        return None  # quarters: column headers are period end dates, not fiscal years
    years = _years_named(sentence)
    if len(years) != 1:
        return None
    year = next(iter(years))
    columns = {table_year for value, table_year in context.year_values if same_amount(written, value)}
    if not columns or year in columns:
        return None
    # The value must not also appear somewhere outside such tables (for example in prose).
    in_tables = sum(1 for value, _ in context.year_values if same_amount(written, value))
    anywhere = sum(1 for number, _ in context.passage_numbers if same_amount(written, number.value))
    if anywhere > in_tables:
        return None
    return f"the sentence says {year}, but this value is in the {', '.join(sorted(columns))} column"


def _derivations(values: list[float]):
    """Results of one simple step on known values: sums, differences, ratios,
    changes and averages. Yields (result, description)."""
    for a in values:
        yield a, f"{a:g} rounded"
        yield 2 * a, f"{a:g} + {a:g}"
    for a, b in itertools.permutations(values, 2):  # no products: written ones are checked as calculations
        yield a + b, f"{a:g} + {b:g}"
        yield a - b, f"{a:g} - {b:g}"
        if b:
            yield a / b, f"{a:g} / {b:g}"
            yield (a - b) / b, f"({a:g} - {b:g}) / {b:g}"
    for a, b, c in itertools.combinations(values, 3):
        yield (a + b + c) / 3, f"average of {a:g}, {b:g}, {c:g}"
    for a, b in itertools.combinations(values, 2):
        yield (a + b) / 2, f"average of {a:g} and {b:g}"


def _derived_from(written: Number, known: list[float]) -> str | None:
    distinct = list(dict.fromkeys(abs(value) for value in known if value))[:14]
    scaled = written.scale != 1.0 or written.currency
    for result, description in _derivations(distinct):
        candidates = [result, result * 100] + ([result * 1e3, result / 1e3] if scaled else [])
        # Rounding only, no slack: an unwritten step must land on the written digits.
        if any(abs(written.magnitude - abs(c)) <= 0.5 * 10 ** -written.decimals * (1 + 1e-9) for c in candidates):
            return description
    return None


def _check_once(text: str, context: _Context) -> tuple[list[NumberCheck], list[tuple[Number, float, str]]]:
    """One pass over the answer. Returns the number checks and the calculations
    that are wrong (number, correct value, expression)."""
    masked = _SCALE_WORDS.sub(lambda m: " " * len(m.group(0)), mask_citations(text))
    numbers = find_numbers(text)
    status: dict[int, NumberCheck] = {}
    known: list[float] = [n.value for n in context.question_numbers]

    for index, number in enumerate(numbers):
        page = context.quoted_page(number)
        if page is not None:
            status[index] = NumberCheck(number.raw, "quoted", f"page {page}")
            known.append(number.value)
        elif context.in_question(number):
            status[index] = NumberCheck(number.raw, "question", "given in the question")
            known.append(number.value)

    for index, number in enumerate(numbers):
        if index not in status and _is_constant(number):
            status[index] = NumberCheck(number.raw, "constant", "a constant such as x 100")

    wrong: list[tuple[Number, float, str]] = []
    for _ in range(6):  # repeat until nothing new is supported: a result can be the input of a later step
        progress = False
        for match in _EQUALS.finditer(masked):  # written calculations, left to right
            found = _expression_before(masked, match.start())
            if not found:
                continue
            start, expression = found
            operands = [i for i, n in enumerate(numbers) if start <= n.start < match.start()]
            if not all(i in status for i in operands):
                continue  # an input is not supported (yet), so the result cannot be judged
            value = _evaluate(expression)
            if value is None:
                continue
            right = _expression_after(masked, match.end())
            if right:  # a chain "a + b = 402 / 3 = 134": the middle step must equal the left side
                right_end, right_expression = right
                right_value = _evaluate(right_expression)
                inside = [i for i, n in enumerate(numbers) if match.end() <= n.start < right_end]
                if right_value is not None and abs(right_value - value) <= CALC_SLACK * max(abs(value), 1e-9):
                    for i in inside:
                        if i not in status:
                            status[i] = NumberCheck(numbers[i].raw, "calculated", expression.strip())
                            known.append(numbers[i].value)
                            progress = True
                continue
            after = [(i, n) for i, n in enumerate(numbers) if n.start >= match.end()]
            if not after or not _AFTER_EQUALS.match(masked[match.end():after[0][1].start]):
                continue
            result_index, result = after[0]
            if same_amount(result, value, allow_percent=True, slack=CALC_SLACK):
                if result_index not in status:
                    status[result_index] = NumberCheck(result.raw, "calculated", expression.strip())
                    known.append(result.value)
                    progress = True
            elif (result_index not in status or status[result_index].status == "calculated") and \
                    not any(w[0] == result for w in wrong):
                if result.percent and abs(value) <= 1.5 and "100" not in expression:
                    value *= 100
                wrong.append((result, value, expression.strip()))
        if wrong:
            break  # correct the first wrong step, then check again from the start
        for index, number in enumerate(numbers):
            if index not in status and (description := _derived_from(number, known)):
                status[index] = NumberCheck(number.raw, "derived", description)
                known.append(number.value)
                progress = True
        if not progress:
            break

    for index, number in enumerate(numbers):
        if index not in status:
            status[index] = NumberCheck(number.raw, "unsupported", "not in the passages and not a checked calculation")

    for index, number in enumerate(numbers):  # the right column of a table for the year named
        if status[index].status != "quoted":
            continue
        sentence = next((s for s in _sentences(text) if s[0] <= number.start < s[1]), (0, len(text)))
        problem = _year_problem(number, text[sentence[0]:sentence[1]], context)
        if problem:
            status[index] = NumberCheck(number.raw, "wrong_year", problem)
    for index, number in enumerate(numbers):
        status[index].start, status[index].end = number.start, number.start + len(number.raw)
    return [status[i] for i in range(len(numbers))], wrong


def _sentences(text: str) -> list[tuple[int, int]]:
    bounds, start = [], 0
    for match in _SENTENCE.finditer(text):
        bounds.append((start, match.start()))
        start = match.end()
    bounds.append((start, len(text)))
    return bounds


def verify_answer(text: str, chunks: Sequence[Chunk], question: str = "", truncated: bool = False,
                  hedges: Sequence[str] | None = None) -> Verification:
    """Check an answer against the passages it was written from."""
    reasons: list[str] = []
    lowered = text.lower()
    hedge = next((phrase for phrase in (hedges or DEFAULT_HEDGES) if phrase in lowered), None)
    if hedge:
        reasons.append(f'the model was unsure ("{hedge}" inside the answer)')
    if truncated:
        reasons.append("the answer was cut off by the length limit")

    context = _Context(chunks, question)
    corrections: list[str] = []
    checks, wrong = _check_once(text, context)
    while wrong and len(corrections) < MAX_CORRECTIONS:
        number, value, expression = wrong[0]
        new_digits = _format_like(value, number)
        corrections.append(f"{expression} is {new_digits}, not {number.digits}")
        text = _replace_number(text, number.digits, new_digits)
        checks, wrong = _check_once(text, context)
    if wrong:
        reasons.append("a calculation is wrong and could not be corrected")
    unsupported = [c.raw for c in checks if c.status == "unsupported"]
    if unsupported:
        reasons.append("numbers not found in the passages: " + ", ".join(unsupported[:4]))
    wrong_year = [c.raw for c in checks if c.status == "wrong_year"]
    if wrong_year:
        reasons.append("values taken from another year's column: " + ", ".join(wrong_year[:4]))
    if corrections:  # mark corrected results in the report
        corrected = {re.sub(r".* is (\S+), not .*", r"\1", c) for c in corrections}
        for check in checks:
            if check.status in ("calculated", "derived") and any(digits in check.raw for digits in corrected):
                check.status = "corrected"
    return Verification(ok=not reasons, text=text, reasons=reasons, numbers=checks, corrections=corrections)
