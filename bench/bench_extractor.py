"""Extractor benchmark (GLiNER small): latency on a question and on a chunk, plus a sanity check.

GLiNER has a maximum input length; the chunk's word count is recorded so a
truncated input is visible in the result.

Run: .venv\\Scripts\\python.exe bench\\bench_extractor.py
"""

from __future__ import annotations

from itertools import cycle

from common import load_fixtures, load_sample_chunks, require_model, run_in_child, save_result, time_calls, time_once

THRESHOLD = 0.5  # GLiNER's usual default


def bench(model_path: str, fixtures: dict, chunk: str) -> dict:
    import torch
    from gliner import GLiNER

    model, load_ms = time_once(lambda: GLiNER.from_pretrained(model_path, local_files_only=True))
    types = fixtures["entity_types"]
    question_cycle = cycle(fixtures["questions"])
    question = time_calls(lambda: model.predict_entities(next(question_cycle), types, threshold=THRESHOLD))
    chunk_stats = time_calls(lambda: model.predict_entities(chunk, types, threshold=THRESHOLD))
    sanity = [
        {"text": sentence,
         "entities": [(e["text"], e["label"], round(e["score"], 2)) for e in model.predict_entities(sentence, types, threshold=THRESHOLD)]}
        for sentence in fixtures["entity_sentences"]
    ]
    return {
        "load_ms": load_ms,
        "torch_threads": torch.get_num_threads(),
        "entity_types": types,
        "question": question,
        "chunk": {"words": len(chunk.split()), "max_len": getattr(model.config, "max_len", None), **chunk_stats},
        "sanity_check": sanity,
    }


def main() -> None:
    model_path = require_model("extractor")
    require_model("extractor_tokenizer")
    chunk = load_sample_chunks()[0][1]
    result = run_in_child(bench, model_path, load_fixtures(), chunk)
    save_result("extractor", result)
    print(f"load {result['load_ms']} ms, question median {result['question']['median_ms']} ms, "
          f"chunk ({result['chunk']['words']} words) median {result['chunk']['median_ms']} ms, peak RAM {result['peak_ram_mb']} MB")


if __name__ == "__main__":
    main()
