"""Dense retrieval: bge-small embeddings stored in Chroma.

Embeddings are normalised, so Chroma's default squared L2 distance d relates to
cosine similarity by cos = 1 - d / 2. Scores returned here are cosine similarities.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.interfaces import Retriever
from core.registry import register
from core.types import Chunk

COLLECTION = "chunks"
CHROMA_MAX_BATCH = 2000


class Embedder:
    """SentenceTransformer wrapper. Queries get the model's retrieval instruction, passages do not."""

    def __init__(self, model_path: str, query_instruction: str = "", batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_path, device="cpu")
        self.query_instruction = query_instruction
        self.batch_size = batch_size

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=False, verbose=False))

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                                    show_progress_bar=False)
        return vectors.tolist()

    def encode_query(self, text: str) -> list[float]:
        return self.model.encode(self.query_instruction + text, normalize_embeddings=True).tolist()


def flat_metadata(chunk: Chunk) -> dict[str, Any]:
    """Chroma accepts only str, int, float and bool values."""
    meta = {key: str(value) for key, value in chunk.metadata.items()}
    meta.update({"doc_id": chunk.doc_id, "page": chunk.page_start, "chunk_type": chunk.chunk_type or ""})
    return meta


class DenseIndex:
    """A persistent Chroma collection of chunk embeddings."""

    def __init__(self, path: str | Path) -> None:
        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(COLLECTION)

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        for start in range(0, len(chunks), CHROMA_MAX_BATCH):
            batch = chunks[start : start + CHROMA_MAX_BATCH]
            self.collection.add(ids=[chunk.id for chunk in batch],
                                embeddings=[list(e) for e in embeddings[start : start + CHROMA_MAX_BATCH]],
                                metadatas=[flat_metadata(chunk) for chunk in batch])

    def search(self, query_embedding: Sequence[float], k: int, where: dict | None = None) -> list[tuple[str, float]]:
        result = self.collection.query(query_embeddings=[list(query_embedding)], n_results=k, where=where)
        return [(chunk_id, 1 - distance / 2) for chunk_id, distance in zip(result["ids"][0], result["distances"][0])]


@register("retriever", "dense")
class DenseRetriever(Retriever):
    """Plain dense search. Metadata filters are applied as an extra filtered search by the hybrid retriever."""

    def __init__(self, embedder: Embedder, index: DenseIndex, chunk_store: dict[str, Chunk]) -> None:
        self.embedder, self.index, self.chunk_store = embedder, index, chunk_store

    def retrieve(self, query: str, filters: dict[str, Any] | None, k: int) -> list[Chunk]:
        conditions = [{key: value} for key, value in (filters or {}).items()]
        where = None if not conditions else conditions[0] if len(conditions) == 1 else {"$and": conditions}
        hits = self.index.search(self.embedder.encode_query(query), k, where=where)
        return [self.chunk_store[chunk_id].with_score(score) for chunk_id, score in hits]
