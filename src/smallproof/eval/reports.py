"""Turns saved run records into metrics.

Hallucination rate (definition proposed in docs/PLAN.md, Phase 4): the share of
questions where the system answered (did not refuse) and the answer was graded
wrong. Answered-but-wrong over answered questions is also reported.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from smallproof.core.profiling import summarize_latencies
from smallproof.eval.grading import grade
from smallproof.eval.metrics import bootstrap_ci, citation_match, hit_at_k, rate
from smallproof.core.types import Chunk


def load_records(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def add_grades(records: list[dict], generator=None) -> list[dict]:
    """Grade answer records that have no grade yet."""
    for record in records:
        if "prediction" in record and "grade" not in record:
            record["grade"] = grade(record["question"], record["gold"], record["prediction"], record["refused"], generator)
    return records


def save_records(records: list[dict], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(r, default=str) + "\n" for r in records), encoding="utf-8")


def _gold_keys(record: dict) -> set[tuple[str, int]]:
    return {(e["doc_id"], int(e["page"])) for e in record.get("evidence", [])}


def _as_chunks(refs: list[dict]) -> list[Chunk]:
    return [Chunk(id=r["id"], doc_id=r["doc_id"], text="", page_start=r["page"], page_end=r["page"])
            for r in refs if "doc_id" in r]


def retrieval_metrics(records: list[dict]) -> dict:
    with_gold = [r for r in records if _gold_keys(r)]
    out = {"questions_with_evidence": len(with_gold)}
    for k in (1, 3, 5):
        out[f"hit@{k}"] = round(rate([hit_at_k(_as_chunks(r["chunks"]), _gold_keys(r), k) for r in with_gold]), 3) if with_gold else None
    candidates = [r for r in with_gold if r.get("candidates")]
    if candidates:
        out["hit@candidates"] = round(rate([hit_at_k(_as_chunks(r["candidates"]), _gold_keys(r), len(r["candidates"])) for r in candidates]), 3)
    return out


def answer_metrics(records: list[dict]) -> dict:
    graded = [r for r in records if r.get("grade", {}).get("correct") is not None]
    if not graded:
        return {}
    correct = [bool(r["grade"]["correct"]) for r in graded]
    answered = [r for r in graded if not r["refused"]]
    low, high = bootstrap_ci([1.0 if c else 0.0 for c in correct])
    by_method = defaultdict(list)
    for r in graded:
        by_method[r["grade"]["method"]].append(bool(r["grade"]["correct"]))
    by_group = defaultdict(list)
    for r in graded:
        by_group[r.get("group") or "none"].append(bool(r["grade"]["correct"]))
    cite = [citation_match({tuple(c) for c in r["citations"]}, _gold_keys(r))["any_match"]
            for r in answered if _gold_keys(r)]
    return {
        "graded": len(graded),
        "accuracy": round(rate(correct), 3), "accuracy_ci95": [round(low, 3), round(high, 3)],
        "correct": sum(correct),
        "refusal_rate": round(rate([r["refused"] for r in graded]), 3),
        "hallucination_rate": round(sum(1 for r in answered if not r["grade"]["correct"]) / len(graded), 3),
        "wrong_when_answered": round(rate([not r["grade"]["correct"] for r in answered]), 3) if answered else None,
        "citation_any_match": round(sum(cite) / len(cite), 3) if cite else None,
        "by_method": {m: f"{sum(v)}/{len(v)}" for m, v in by_method.items()},
        "by_group": {g: f"{sum(v)}/{len(v)}" for g, v in by_group.items()},
    }


def latency_metrics(records: list[dict]) -> dict:
    stages = defaultdict(list)
    for r in records:
        for stage, ms in (r.get("timings_ms") or {}).items():
            stages[stage].append(ms)
        if r.get("timings_ms"):
            stages["total"].append(sum(r["timings_ms"].values()))
    out = {stage: {"median_ms": round(s["median_ms"], 1), "p95_ms": round(s["p95_ms"], 1)}
           for stage, values in stages.items() for s in [summarize_latencies(values)]}
    vram = [r["gpu"]["peak_mib"] for r in records if r.get("gpu")]
    if vram:
        out["peak_vram_mib"] = max(vram)
        out["vram_baseline_mib"] = min(r["gpu"]["baseline_mib"] for r in records if r.get("gpu"))
    return out


def summarize(records: list[dict]) -> dict:
    return {"n": len(records), "retrieval": retrieval_metrics(records), "answers": answer_metrics(records),
            "latency": latency_metrics(records)}
