"""Runs a system variant over eval questions and saves one JSONL record per question.

Modes:
- answer: full question answering (retrieval, gate, SLM), graded afterwards
- retrieval: retrieval and gate only, no SLM (cheap; used for recall@k and ablations)

Variant "A_slm_only" is Baseline A. Other variant names come from core.pipeline.VARIANTS.
Runs resume: questions already in the output file are skipped.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from core.profiling import GpuMemorySampler, Profiler


def load_eval_set(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _chunk_refs(chunks) -> list[dict]:
    return [{"id": c.id, "doc_id": c.doc_id, "page": c.page_start, "type": c.chunk_type} for c in chunks]


def run(pipeline, rows: Sequence[dict], variant: str, out_path: str | Path, mode: str = "answer",
        log: Callable[[str], None] = print, guard=None) -> Path:
    """guard (core.thermal.ThermalGuard) is checked before each question; the run stops cleanly at its time limit."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out_path.read_text(encoding="utf-8").splitlines()} if out_path.exists() else set()
    todo = [row for row in rows if row["id"] not in done]
    log(f"{variant} ({mode}): {len(todo)} to run, {len(done)} already done")
    with out_path.open("a", encoding="utf-8") as handle:
        for number, row in enumerate(todo, start=1):
            if guard and not guard.checkpoint():
                log(f"  stopped after {number - 1}/{len(todo)} (time limit); run again to resume")
                break
            record = {"id": row["id"], "group": row.get("group"), "question": row["question"], "gold": row["answer"],
                      "evidence": row.get("evidence", []), "variant": variant, "mode": mode}
            if mode == "retrieval":
                profiler = Profiler()
                result = pipeline.retrieve_with_fallback(row["question"], variant, profiler)
                record.update({"chunks": _chunk_refs(result.chunks), "candidates": _chunk_refs(result.candidates),
                               "verdict": result.verdict, "confidence": result.confidence,
                               "widened": result.widened, "tags": result.tags, "timings_ms": profiler.timings_ms})
            else:
                with GpuMemorySampler() as gpu:
                    if variant == "A_slm_only":
                        answer = pipeline.ask_slm_only(row["question"], row["doc_id"])
                    else:
                        answer = pipeline.ask(row["question"], variant)
                record.update({"prediction": answer.text, "refused": answer.refused, "verdict": answer.verdict,
                               "confidence": answer.confidence,
                               "citations": [[c.doc_id, c.page] for c in answer.citations],
                               "chunks": answer.details.get("chunks") or [{"id": i} for i in answer.chunk_ids],
                               "timings_ms": answer.timings_ms, "details": answer.details, "gpu": gpu.result()})
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
            if number % 10 == 0 or number == len(todo):
                log(f"  {number}/{len(todo)}")
    return out_path
