"""Generator backend: a local Ollama model on the GPU.

The prompt lists the chunks as numbered passages with their document and page.
The SLM must cite pages like [p. 12] and reply NOT FOUND when the passages do
not hold the answer. Citations are checked against the pages it was given.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from core.interfaces import Generator
from core.registry import register
from core.types import Answer, Chunk
from generate.citations import check_citations
from generate.table_years import annotate_table_years

REFUSAL_MARKER = "NOT FOUND"


def source_label(chunk: Chunk) -> str:
    meta = chunk.metadata
    parts = [meta.get("company"), meta.get("year"), meta.get("doc_type")]
    return " ".join(part for part in parts if part) or chunk.doc_id


def format_passages(chunks: Sequence[Chunk], table_years: bool = False) -> str:
    """Numbered passages with their source. With table_years, values in
    multi-year tables are labelled with their year (generate/table_years.py)."""
    def text_of(chunk: Chunk) -> str:
        return annotate_table_years(chunk.text) if table_years else chunk.text

    return "\n\n".join(f"[{n}] {source_label(chunk)}, page {chunk.page_start}\n{text_of(chunk)}"
                       for n, chunk in enumerate(chunks, start=1))


def is_refusal(text: str) -> bool:
    return REFUSAL_MARKER in text.upper()[:200]


@register("generator", "ollama")
class OllamaGenerator(Generator):
    def __init__(self, model: str, host: str = "http://127.0.0.1:11434", num_ctx: int = 4096,
                 temperature: float = 0.0, num_predict: int = 256, keep_alive: str = "30m",
                 table_years: bool = False) -> None:
        import ollama

        self.client = ollama.Client(host=host)
        self.model, self.keep_alive, self.table_years = model, keep_alive, table_years
        self.options = {"num_ctx": num_ctx, "temperature": temperature, "num_predict": num_predict, "seed": 0}

    def complete(self, prompt: str, num_predict: int | None = None) -> tuple[str, dict]:
        """Raw completion. Returns the text and Ollama's own timing counters (ms)."""
        options = {**self.options, **({"num_predict": num_predict} if num_predict else {})}
        start = time.perf_counter()
        response = self.client.generate(model=self.model, prompt=prompt, options=options, keep_alive=self.keep_alive)
        stats = {
            "slm_total_ms": round((time.perf_counter() - start) * 1000, 1),
            "slm_prompt_tokens": response.prompt_eval_count,
            "slm_output_tokens": response.eval_count,
        }
        return response.response.strip(), stats

    def generate(self, query: str, chunks: Sequence[Chunk], template: str) -> Answer:
        text, stats = self.complete(template.format(passages=format_passages(chunks, self.table_years), question=query))
        refused = is_refusal(text)
        citations = [] if refused else check_citations(text, chunks).citations
        return Answer(text=text, citations=citations, refused=refused, chunk_ids=[chunk.id for chunk in chunks],
                      timings_ms={"generate": stats["slm_total_ms"]}, details={"slm": stats})
