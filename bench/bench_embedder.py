"""Embedder benchmark (bge-small): single query latency and 100-chunk batch throughput.

The batch is slow compared with a query, so it uses 1 warmup and 3 timed runs.

Run: .venv\\Scripts\\python.exe bench\\bench_embedder.py
"""

from __future__ import annotations

from itertools import cycle
from statistics import mean

from common import load_fixtures, load_sample_chunks, require_model, run_in_child, save_result, time_calls, time_once

BATCH_CHUNKS = 100
BATCH_SIZE = 32


def bench(model_path: str, questions: list[str], chunks: list[str]) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer

    model, load_ms = time_once(lambda: SentenceTransformer(model_path, device="cpu"))
    question_cycle = cycle(questions)
    query = time_calls(lambda: model.encode(next(question_cycle), normalize_embeddings=True))
    batch = time_calls(
        lambda: model.encode(chunks, batch_size=BATCH_SIZE, normalize_embeddings=True), warmup=1, runs=3
    )
    tokens = [len(ids) for ids in model.tokenizer(chunks)["input_ids"]]
    return {
        "load_ms": load_ms,
        "query": query,
        f"batch_{len(chunks)}_chunks": batch,
        "chunks_per_second": round(len(chunks) / (batch["median_ms"] / 1000), 1),
        "chunk_tokens_mean": round(mean(tokens)),
        "chunk_tokens_max": max(tokens),
        "chunks_over_max_length": sum(1 for count in tokens if count > model.max_seq_length),
        "max_seq_length": model.max_seq_length,
        "embedding_dim": model.get_sentence_embedding_dimension(),
        "torch_threads": torch.get_num_threads(),
    }


def main() -> None:
    model_path = require_model("embedder")
    chunks = [text for _, text in load_sample_chunks()][:BATCH_CHUNKS]
    result = run_in_child(bench, model_path, load_fixtures()["questions"], chunks)
    save_result("embedder", result)
    print(f"load {result['load_ms']} ms, query median {result['query']['median_ms']} ms, "
          f"{result['chunks_per_second']} chunks/s, peak RAM {result['peak_ram_mb']} MB")


if __name__ == "__main__":
    main()
