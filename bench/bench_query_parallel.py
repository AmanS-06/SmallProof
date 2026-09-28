"""CPU query stage: sequential vs parallel.

Runs the three per-question CPU tasks (classifier with 8 labels, GLiNER,
query embedding) one after another, then in 3 threads, at two torch thread
settings. Parallel only helps if the tasks do not slow each other down.

Run: .venv\\Scripts\\python.exe bench\\bench_query_parallel.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import cycle

from common import load_fixtures, require_model, run_in_child, save_result, time_calls, time_once

THREAD_SETTINGS = (4, 2)  # capped to keep the laptop cool


def bench(paths: dict, fixtures: dict, torch_threads: int | None) -> dict:
    import torch
    from gliner import GLiNER
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline

    if torch_threads:
        torch.set_num_threads(torch_threads)

    def load_all():
        classifier = pipeline("zero-shot-classification", model=paths["classifier"], device=-1,
                              model_kwargs={"reference_compile": False})
        extractor = GLiNER.from_pretrained(paths["extractor"], local_files_only=True)
        embedder = SentenceTransformer(paths["embedder"], device="cpu")
        return classifier, extractor, embedder

    (classifier, extractor, embedder), load_ms = time_once(load_all)
    labels = fixtures["latency_labels"][:8]
    tasks = [
        lambda question: classifier(question, labels),
        lambda question: extractor.predict_entities(question, fixtures["entity_types"], threshold=0.5),
        lambda question: embedder.encode(question, normalize_embeddings=True),
    ]
    question_cycle = cycle(fixtures["questions"])

    def sequential():
        question = next(question_cycle)
        return [task(question) for task in tasks]

    pool = ThreadPoolExecutor(max_workers=len(tasks))

    def parallel():
        question = next(question_cycle)
        return [future.result() for future in [pool.submit(task, question) for task in tasks]]

    try:
        result = {"torch_threads": torch.get_num_threads(), "load_all_ms": load_ms,
                  "sequential": time_calls(sequential), "parallel": time_calls(parallel)}
    finally:
        pool.shutdown()
    return result


def main() -> None:
    paths = {key: require_model(key) for key in ("classifier", "extractor", "embedder")}
    require_model("extractor_tokenizer")
    fixtures = load_fixtures()
    results = [run_in_child(bench, paths, fixtures, setting) for setting in THREAD_SETTINGS]
    save_result("query_parallel", {"results": results})
    for result in results:
        print(f"threads {result['torch_threads']}: sequential median {result['sequential']['median_ms']} ms, "
              f"parallel median {result['parallel']['median_ms']} ms, peak RAM {result['peak_ram_mb']} MB")


if __name__ == "__main__":
    main()
