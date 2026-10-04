"""SLM benchmark (Ollama on the GPU).

Measures time to first token, output tokens per second, total latency and GPU
memory for a prompt of 5 chunks (about 2,000 tokens) at context 4096.

The model is unloaded first, so the first run includes a cold load. Each run
uses different chunks: Ollama reuses cached prompt prefixes, and repeating the
same passages would make prompt processing look faster than it really is.

Run: .venv\\Scripts\\python.exe bench\\bench_slm.py
"""

from __future__ import annotations

import sys
import time

from common import WARMUP_RUNS, GpuMemorySampler, load_config, load_fixtures, load_sample_chunks, save_result
from smallproof.core.profiling import summarize_latencies

TIMED_RUNS = 10  # fewer than other benches: SLM runs heat the GPU

CHUNKS_PER_PROMPT = 5
MAX_OUTPUT_TOKENS = 256
PROMPT = """Answer the question using only the passages below. Cite the page of each fact like [p. 3].
If the passages do not contain the answer, say that you cannot answer.

{passages}

Question: {question}
Answer:"""


def build_prompt(chunks: list[tuple[int, str]], run_index: int, question: str) -> str:
    start = (run_index * CHUNKS_PER_PROMPT) % max(1, len(chunks) - CHUNKS_PER_PROMPT)
    picked = chunks[start : start + CHUNKS_PER_PROMPT]
    passages = "\n\n".join(f"[Passage {i + 1}, p. {page}]\n{text}" for i, (page, text) in enumerate(picked))
    return PROMPT.format(passages=passages, question=question)


def run_once(ollama, model: str, prompt: str, options: dict) -> dict:
    """One streamed generation. Ollama reports token counts and durations in nanoseconds."""
    start = time.perf_counter()
    first_token = None
    pieces, final = [], None
    for part in ollama.generate(model=model, prompt=prompt, options=options, stream=True):
        if first_token is None and part.response:
            first_token = time.perf_counter()
        pieces.append(part.response)
        if part.done:
            final = part
    end = time.perf_counter()
    return {
        "ttft_ms": round((first_token - start) * 1000, 1) if first_token else None,
        "total_ms": round((end - start) * 1000, 1),
        "prompt_tokens": final.prompt_eval_count,
        "output_tokens": final.eval_count,
        "tokens_per_second": round(final.eval_count / (final.eval_duration / 1e9), 1) if final.eval_duration else None,
        "load_ms": round((final.load_duration or 0) / 1e6, 1),
        "text": "".join(pieces),
    }


def main() -> None:
    from smallproof.core.ollama_server import OllamaServer
    from smallproof.core.config import load_config as load_full_config

    with OllamaServer.from_config(load_full_config()):  # server only lives for this benchmark
        _run()


def _run() -> None:
    import ollama

    config = load_config()["generator"]
    model = config["model"]
    try:
        pulled = [entry.model for entry in ollama.list().models]
    except Exception as error:
        sys.exit(f"Ollama is not reachable ({error}). See docs/PLAN.md step 0.3.")
    if model not in pulled:
        sys.exit(f"Ollama model {model} is not pulled yet. See docs/PLAN.md step 0.3.")

    chunks = load_sample_chunks()
    questions = load_fixtures()["questions"]
    options = {"num_ctx": config["num_ctx"], "temperature": config["temperature"],
               "num_predict": MAX_OUTPUT_TOKENS, "seed": 0}

    ollama.generate(model=model, prompt="", keep_alive=0)  # unload, so the first run is cold
    time.sleep(3)
    with GpuMemorySampler() as gpu:
        cold = run_once(ollama, model, build_prompt(chunks, 0, questions[0]), options)
        for index in range(1, WARMUP_RUNS + 1):
            run_once(ollama, model, build_prompt(chunks, index, questions[index % len(questions)]), options)
        warm = [
            run_once(ollama, model, build_prompt(chunks, index, questions[index % len(questions)]), options)
            for index in range(WARMUP_RUNS + 1, WARMUP_RUNS + 1 + TIMED_RUNS)
        ]
        running = [entry for entry in ollama.ps().models if entry.model == model]

    def stats(key: str) -> dict:
        return {k: round(v, 1) for k, v in summarize_latencies([run[key] for run in warm if run[key] is not None]).items()}

    placement = None
    if running:
        placement = {"size_mb": round(running[0].size / 2**20), "size_vram_mb": round(running[0].size_vram / 2**20),
                     "gpu_fraction": round(running[0].size_vram / running[0].size, 3)}
    result = {
        "model": model,
        "options": options,
        "cold_run": {key: value for key, value in cold.items() if key != "text"},
        "warm_runs": {"ttft_ms": stats("ttft_ms"), "total_ms": stats("total_ms"),
                      "tokens_per_second": stats("tokens_per_second"), "prompt_tokens": stats("prompt_tokens"),
                      "output_tokens": stats("output_tokens")},
        "gpu_memory": gpu.result(),
        "placement": placement,
        "think_tag_seen": any("<think>" in run["text"] for run in [cold, *warm]),
        "sample_answer": cold["text"][:600],
    }
    save_result("slm", result)
    print(f"cold: load {cold['load_ms']} ms, ttft {cold['ttft_ms']} ms | warm: ttft median {result['warm_runs']['ttft_ms']['median_ms']} ms, "
          f"{result['warm_runs']['tokens_per_second']['median_ms']} tok/s, total median {result['warm_runs']['total_ms']['median_ms']} ms")
    print(f"prompt tokens median {result['warm_runs']['prompt_tokens']['median_ms']} | VRAM {result['gpu_memory']} | placement {placement}")


if __name__ == "__main__":
    main()
