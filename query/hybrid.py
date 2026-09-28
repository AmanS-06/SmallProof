"""Hybrid retrieval helpers: reciprocal rank fusion (RRF).

BM25 and dense search give scores on different scales, so we merge them by
rank instead of by score. Each list gives a chunk weight / (k + rank) points
and the points are added up (Cormack, Clarke and Buettcher, 2009). A chunk
ranked well by both lists beats one ranked first by only one list.

This file holds the fusion math. The HybridRetriever class that calls BM25 and
dense search, and the metadata boosts, are built in Phase 3.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.types import Chunk

# Standard starting value from the RRF paper. Tuned on the dev split in Phase 3 if needed.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Merge ranked lists of ids into one list of (id, fused score), best first.

    ranked_lists: each inner list holds ids, best first. Lengths can differ.
    k: damping constant. A larger k makes low ranks count almost as much as top ranks.
    weights: one weight per list, default 1.0 each.

    Ties are broken by the best rank the id got in any list, then by id,
    so the output order is always the same for the same input.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(f"Got {len(weights)} weights for {len(ranked_lists)} lists")
    if any(weight < 0 for weight in weights):
        raise ValueError("Weights must not be negative")

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ids, weight in zip(ranked_lists, weights):
        seen: set[str] = set()
        for rank, item_id in enumerate(ids, start=1):
            if item_id in seen:
                continue  # an id repeated inside one list only counts once
            seen.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)
            best_rank[item_id] = min(best_rank.get(item_id, rank), rank)

    return sorted(scores.items(), key=lambda pair: (-pair[1], best_rank[pair[0]], pair[0]))


def fuse_chunks(
    ranked_chunk_lists: Sequence[Sequence[Chunk]],
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
    top_n: int | None = None,
) -> list[Chunk]:
    """RRF over lists of chunks. Returns copies with .score set to the fused score."""
    # Remember one Chunk object per id, so ids can be turned back into chunks.
    chunk_by_id: dict[str, Chunk] = {}
    for chunks in ranked_chunk_lists:
        for chunk in chunks:
            chunk_by_id.setdefault(chunk.id, chunk)

    id_lists = [[chunk.id for chunk in chunks] for chunks in ranked_chunk_lists]
    fused = reciprocal_rank_fusion(id_lists, k, weights)
    if top_n is not None:
        fused = fused[:top_n]
    return [chunk_by_id[chunk_id].with_score(score) for chunk_id, score in fused]
