"""Reranker benchmark (MiniLM cross-encoder): latency for 20 and 50 candidates.

Run: .venv\\Scripts\\python.exe bench\\bench_reranker.py
"""

from __future__ import annotations

from itertools import cycle

from common import load_fixtures, load_sample_chunks, require_model, run_in_child, save_result, time_calls, time_once

CANDIDATE_COUNTS = (20, 50)


def bench(model_path: str, questions: list[str], chunks: list[str]) -> dict:
    import torch
    from sentence_transformers import CrossEncoder

    model, load_ms = time_once(lambda: CrossEncoder(model_path, device="cpu"))
    question_cycle = cycle(questions)
    result: dict = {"load_ms": load_ms, "torch_threads": torch.get_num_threads()}
    for count in CANDIDATE_COUNTS:

        def score(count: int = count):
            question = next(question_cycle)
            return model.predict([(question, chunk) for chunk in chunks[:count]])

        result[f"candidates_{count}"] = time_calls(score)
    return result


def main() -> None:
    model_path = require_model("reranker")
    chunks = [text for _, text in load_sample_chunks()][: max(CANDIDATE_COUNTS)]
    result = run_in_child(bench, model_path, load_fixtures()["questions"], chunks)
    save_result("reranker", result)
    summary = ", ".join(f"{n} candidates median {result[f'candidates_{n}']['median_ms']} ms" for n in CANDIDATE_COUNTS)
    print(f"load {result['load_ms']} ms, {summary}, peak RAM {result['peak_ram_mb']} MB")


if __name__ == "__main__":
    main()
