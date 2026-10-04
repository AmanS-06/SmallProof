import pytest

from smallproof.core.types import Chunk
from smallproof.query.hybrid import fuse_chunks, reciprocal_rank_fusion


def test_rrf_prefers_ids_ranked_well_by_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], k=60)
    assert [item_id for item_id, _ in fused] == ["b", "a", "c"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_includes_ids_found_by_one_list_only():
    fused = dict(reciprocal_rank_fusion([["a"], ["b"]], k=60))
    assert fused == {"a": pytest.approx(1 / 61), "b": pytest.approx(1 / 61)}


def test_rrf_ties_break_by_best_rank_then_id():
    # a and b get identical scores; both have best rank 1, so id order decides.
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    assert [item_id for item_id, _ in fused] == ["a", "b"]


def test_rrf_weights_and_repeated_ids():
    fused = dict(reciprocal_rank_fusion([["a", "a", "b"], ["b"]], k=10, weights=[1.0, 2.0]))
    assert fused["a"] == pytest.approx(1 / 11)  # the repeat of "a" is ignored
    assert fused["b"] == pytest.approx(1 / 13 + 2 / 11)


def test_rrf_rejects_bad_settings():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[-1.0])


def test_fuse_chunks_returns_scored_copies():
    a = Chunk("a", "doc", "text a", 1, 1)
    b = Chunk("b", "doc", "text b", 2, 2)
    fused = fuse_chunks([[a, b], [b]], top_n=1)
    assert [chunk.id for chunk in fused] == ["b"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert b.score is None  # the original chunk is untouched
