"""Evaluation metrics.

All functions work on plain Python values, so metrics can be recomputed from
saved run records without running any model again.

Implemented now:
    rate (accuracy, refusal rate and other yes/no shares), recall@k, hit@k,
    citation match, confusion matrix, bootstrap confidence interval.
Not yet:
    hallucination rate. Its definition is agreed with Aman in Phase 4
    (docs/PLAN.md), so it is added then.

Evidence is identified by (doc_id, page), with 1-based pages like Chunk.
When FinanceBench is loaded in Phase 2, its gold pages must be converted to
the same base (to check then, not assumed).
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from smallproof.core.profiling import percentile
from smallproof.core.types import Chunk

EvidenceKey = tuple[str, int]  # (doc_id, page)


def rate(flags: Sequence[bool]) -> float:
    """Share of True values, for example accuracy or refusal rate."""
    if not flags:
        raise ValueError("Cannot compute a rate over zero items")
    return sum(1 for flag in flags if flag) / len(flags)


def chunk_evidence_keys(chunk: Chunk) -> set[EvidenceKey]:
    """All (doc_id, page) pairs a chunk covers."""
    return {(chunk.doc_id, page) for page in chunk.pages}


def recall_at_k(retrieved: Sequence[Chunk], gold: set[EvidenceKey], k: int) -> float:
    """Share of gold evidence pages covered by the top k retrieved chunks."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if not gold:
        raise ValueError("The question has no gold evidence pages")
    covered: set[EvidenceKey] = set()
    for chunk in retrieved[:k]:
        covered |= chunk_evidence_keys(chunk)
    return len(gold & covered) / len(gold)


def hit_at_k(retrieved: Sequence[Chunk], gold: set[EvidenceKey], k: int) -> bool:
    """True if the top k chunks cover at least one gold evidence page."""
    return recall_at_k(retrieved, gold, k) > 0


def citation_match(cited: set[EvidenceKey], gold: set[EvidenceKey]) -> dict[str, float]:
    """Compare the pages an answer cites with the gold evidence pages.

    precision: share of cited pages that are gold pages
    recall:    share of gold pages that were cited
    any_match: 1.0 if at least one cited page is a gold page

    An answer with no citations scores 0 on all three. Refused answers should
    be left out by the caller. Which number is the headline is decided in Phase 2.
    """
    if not gold:
        raise ValueError("The question has no gold evidence pages")
    overlap = cited & gold
    return {
        "precision": len(overlap) / len(cited) if cited else 0.0,
        "recall": len(overlap) / len(gold),
        "any_match": 1.0 if overlap else 0.0,
    }


def confusion_matrix(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    labels: Sequence[str] | None = None,
) -> tuple[list[str], list[list[int]]]:
    """Count (true, predicted) pairs. Rows are true labels, columns are predicted labels.

    Returns the label order used and the matrix. Without a labels list, every
    label seen is used, sorted.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError(f"Got {len(true_labels)} true labels but {len(predicted_labels)} predictions")
    label_order = list(labels) if labels is not None else sorted(set(true_labels) | set(predicted_labels))
    index = {label: position for position, label in enumerate(label_order)}
    matrix = [[0] * len(label_order) for _ in label_order]
    for true_label, predicted in zip(true_labels, predicted_labels):
        for label in (true_label, predicted):
            if label not in index:
                raise ValueError(f"Label '{label}' is not in the labels list")
        matrix[index[true_label]][index[predicted]] += 1
    return label_order, matrix


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of values.

    For accuracy-style metrics, pass 1.0 or 0.0 per question. The interval is
    seeded, so the same input always gives the same result.
    """
    if not values:
        raise ValueError("Need at least one value")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    rng = random.Random(seed)
    n = len(values)
    # Resample the questions with replacement many times and record each mean.
    means = [sum(rng.choices(values, k=n)) / n for _ in range(n_resamples)]
    tail = (1 - confidence) / 2 * 100
    return percentile(means, tail), percentile(means, 100 - tail)
