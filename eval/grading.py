"""Grading: is a predicted answer correct?

1. numeric: if the gold answer contains a number, the prediction must contain
   the gold answer's first number within 2 percent. Units are compared loosely:
   1,577 million, 1.577 billion and 1577 all match (factors of 1,000), and 25%
   matches 0.25. Years are ignored as candidate answer numbers.
2. judge: gold answers without numbers are judged by the local SLM (YES/NO).
   This is weaker and biased (same model family), so reports show the method.
Refused answers are never correct.
"""

from __future__ import annotations

import re

_NUMBER = re.compile(r"(\(?-?\$?\d[\d,]*(?:\.\d+)?\)?)\s*(%|percent|billion|million|thousand|bn|mn|mm)?", re.IGNORECASE)
_SCALE = {"billion": 1e9, "bn": 1e9, "million": 1e6, "mn": 1e6, "mm": 1e6, "thousand": 1e3}
JUDGE_PROMPT = """You are grading an answer to a financial question.
Question: {question}
Reference answer: {gold}
Candidate answer: {prediction}
Does the candidate answer agree with the reference answer on the key facts? Small wording differences are fine.
Reply with only YES or NO."""


def extract_numbers(text: str) -> list[tuple[float, bool]]:
    """(value, is_percent) for each number, skipping bare years like 2018."""
    numbers = []
    for raw, unit in _NUMBER.findall(text):
        negative = raw.startswith("(") and raw.endswith(")") or "-" in raw
        digits = re.sub(r"[^\d.]", "", raw)
        if not digits or digits == ".":
            continue
        value = float(digits)
        unit = (unit or "").lower()
        if not unit and "." not in digits and 1990 <= value <= 2049:
            continue  # a year, not an answer value
        value *= _SCALE.get(unit, 1)
        numbers.append((-value if negative else value, unit in ("%", "percent")))
    return numbers


def _close(a: float, b: float, rel_tol: float) -> bool:
    return abs(a - b) <= rel_tol * max(abs(a), abs(b), 1e-9)


def numbers_match(gold: tuple[float, bool], candidate: tuple[float, bool], rel_tol: float = 0.02) -> bool:
    (g, g_pct), (c, c_pct) = gold, candidate
    options = [c * factor for factor in (1, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)]
    if g_pct != c_pct:
        options += [c * 100, c / 100]  # 0.25 vs 25%
    return any(_close(abs(g), abs(option), rel_tol) for option in options)


def numeric_grade(gold_answer: str, prediction: str) -> bool | None:
    """True/False when the gold answer has a number, None when it does not."""
    gold_numbers = extract_numbers(gold_answer)
    if not gold_numbers:
        return None
    return any(numbers_match(gold_numbers[0], candidate) for candidate in extract_numbers(prediction))


def grade(question: str, gold_answer: str, prediction: str, refused: bool, generator=None) -> dict:
    """Returns {"correct": bool | None, "method": "refused" | "numeric" | "judge" | "ungraded"}."""
    if gold_answer == "UNANSWERABLE":  # built by `api.cli unanswerable`: refusing is the right answer
        return {"correct": refused, "method": "unanswerable"}
    if refused:
        return {"correct": False, "method": "refused"}
    numeric = numeric_grade(gold_answer, prediction)
    if numeric is not None:
        return {"correct": numeric, "method": "numeric"}
    if generator is None:
        return {"correct": None, "method": "ungraded"}
    reply, _ = generator.complete(JUDGE_PROMPT.format(question=question, gold=gold_answer, prediction=prediction),
                                  num_predict=3)
    return {"correct": reply.strip().upper().startswith("YES"), "method": "judge"}
