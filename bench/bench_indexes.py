"""Index benchmark: BM25 (rank_bm25) and Chroma build time, query latency and size.

Needs no models. Chroma is fed random unit vectors with the bge-small size
(384), so this measures the cost of the index itself, not of embedding.
The Chroma folder is temporary and deleted afterwards.

Run: .venv\\Scripts\\python.exe bench\\bench_indexes.py
"""

from __future__ import annotations

import re
import shutil
from itertools import cycle

from common import ROOT, folder_size_bytes, load_fixtures, load_sample_chunks, run_in_child, save_result, time_calls, time_once

EMBEDDING_DIM = 384  # bge-small-en-v1.5
TOP_K = 50
WORK_DIR = ROOT / "data" / "indexes" / "_bench_tmp"


def tokenize(text: str) -> list[str]:
    """Simple BM25 tokenizer: lowercase words and numbers."""
    return re.findall(r"\w+", text.lower())


def bench_bm25(chunks: list[str], questions: list[str]) -> dict:
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(chunk) for chunk in chunks]
    index = BM25Okapi(tokenized)
    question_cycle = cycle(questions)

    def search() -> list[int]:
        scores = index.get_scores(tokenize(next(question_cycle)))
        return sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:TOP_K]

    return {
        "chunks": len(chunks),
        "corpus_tokens": sum(len(tokens) for tokens in tokenized),
        "build": time_calls(lambda: BM25Okapi(tokenized), warmup=1, runs=5),
        f"query_top{TOP_K}": time_calls(search),
    }


def bench_chroma(chunks: list[str], work_dir: str) -> dict:
    import chromadb
    import numpy as np
    from chromadb.config import Settings

    rng = np.random.default_rng(0)

    def unit_vectors(count: int) -> np.ndarray:
        vectors = rng.normal(size=(count, EMBEDDING_DIM)).astype("float32")
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    vectors = unit_vectors(len(chunks)).tolist()
    ids = [f"c{i}" for i in range(len(chunks))]
    client = chromadb.PersistentClient(path=work_dir, settings=Settings(anonymized_telemetry=False))

    def build():
        try:
            client.delete_collection("bench")
        except Exception:  # the collection does not exist on the first build
            pass
        collection = client.create_collection("bench")
        collection.add(ids=ids, embeddings=vectors, documents=chunks)
        return collection

    build()  # warmup
    collection, build_ms = time_once(build)
    query = time_calls(lambda: collection.query(query_embeddings=unit_vectors(1).tolist(), n_results=TOP_K))
    return {
        "chunks": len(chunks),
        "vectors": f"random unit vectors, dim {EMBEDDING_DIM} (index cost only)",
        "build_ms": build_ms,
        f"query_top{TOP_K}": query,
        "chromadb_version": chromadb.__version__,
    }


def main() -> None:
    chunks = [text for _, text in load_sample_chunks()][:500]
    questions = load_fixtures()["questions"]
    bm25 = run_in_child(bench_bm25, chunks, questions)
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    chroma = run_in_child(bench_chroma, chunks, str(WORK_DIR))
    chroma["disk_mb"] = round(folder_size_bytes(WORK_DIR) / 2**20, 1)
    shutil.rmtree(WORK_DIR, ignore_errors=True)  # after the child exits, so no file is locked
    save_result("indexes", {"bm25": bm25, "chroma": chroma})
    print(f"BM25: {bm25['chunks']} chunks, build median {bm25['build']['median_ms']} ms, "
          f"query median {bm25[f'query_top{TOP_K}']['median_ms']} ms, peak RAM {bm25['peak_ram_mb']} MB")
    print(f"Chroma: build {chroma['build_ms']} ms, query median {chroma[f'query_top{TOP_K}']['median_ms']} ms, "
          f"disk {chroma['disk_mb']} MB, peak RAM {chroma['peak_ram_mb']} MB")


if __name__ == "__main__":
    main()
