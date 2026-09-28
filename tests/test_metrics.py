import pytest

from core.types import Chunk
from eval.metrics import (
    bootstrap_ci,
    citation_match,
    confusion_matrix,
    hit_at_k,
    rate,
    recall_at_k,
)


def test_rate():
    assert rate([True, False, True, True]) == 0.75
    with pytest.raises(ValueError):
        rate([])


def test_recall_and_hit_at_k():
    retrieved = [
        Chunk("c1", "doc", "x", 5, 5),
        Chunk("c2", "other", "y", 6, 6),  # right page, wrong document
        Chunk("c3", "doc", "z", 6, 7),
    ]
    gold = {("doc", 5), ("doc", 6)}
    assert recall_at_k(retrieved, gold, k=1) == 0.5
    assert recall_at_k(retrieved, gold, k=2) == 0.5
    assert recall_at_k(retrieved, gold, k=3) == 1.0
    assert hit_at_k(retrieved, gold, k=1)
    assert not hit_at_k(retrieved[1:], gold, k=1)


def test_recall_rejects_bad_input():
    with pytest.raises(ValueError):
        recall_at_k([], {("doc", 1)}, k=0)
    with pytest.raises(ValueError):
        recall_at_k([], set(), k=1)


def test_citation_match():
    gold = {("doc", 5), ("doc", 6)}
    scores = citation_match({("doc", 5), ("doc", 9)}, gold)
    assert scores == {"precision": 0.5, "recall": 0.5, "any_match": 1.0}
    assert citation_match(set(), gold) == {"precision": 0.0, "recall": 0.0, "any_match": 0.0}


def test_confusion_matrix():
    labels, matrix = confusion_matrix(["a", "a", "b"], ["a", "b", "b"])
    assert labels == ["a", "b"]
    assert matrix == [[1, 1], [0, 1]]
    with pytest.raises(ValueError):
        confusion_matrix(["a"], ["c"], labels=["a", "b"])
    with pytest.raises(ValueError):
        confusion_matrix(["a"], [])


def test_bootstrap_ci_is_seeded_and_sensible():
    values = [1.0] * 30 + [0.0] * 20  # mean 0.6
    low, high = bootstrap_ci(values, seed=1)
    assert (low, high) == bootstrap_ci(values, seed=1)
    assert low < 0.6 < high
    assert bootstrap_ci([1.0, 1.0, 1.0]) == (1.0, 1.0)
