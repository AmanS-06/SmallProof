"""FastAPI server: the HTTP API and the interactive demo (demo/web).

    .venv\\Scripts\\python.exe -m uvicorn api.server:app --port 8000
    then open http://127.0.0.1:8000

JSON API
    GET  /health
    POST /ask {"question": "...", "variant": "C_verified"}: answer, verdict, citations, timings
Demo API (used by demo/web)
    GET  /api/info                  pack, variants, corpus size, example questions
    GET  /api/ask/stream?question=  server-sent events, one per pipeline stage
    GET  /api/chunk/{id}            one chunk's text and metadata
    GET  /api/runs                  saved evaluation runs with their metrics
    GET  /api/runs/{name}           one run's questions and outcomes

One question runs at a time (it is a laptop). The Ollama server starts on the
first question and stops after 5 idle minutes (see core/ollama_server.py).
The pack comes from the PACK environment variable (default: financebench).
Evaluation data stays on this machine: the demo reads it from data/, which is
never committed.
"""

from __future__ import annotations

import json
import os
import queue
import random
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from core.config import resolve_path  # noqa: E402
from core.ollama_server import OnDemandOllama  # noqa: E402
from core.pipeline import DEFAULT_VARIANT, VARIANTS, Pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "demo" / "web"
pipeline = Pipeline.from_pack(os.environ.get("PACK", "financebench"))
ollama = OnDemandOllama(pipeline.config)
busy = threading.Lock()
app = FastAPI(title="Local SLM + Jev-style RAG")

# Saved runs shown in the demo's comparison table, in order, with a readable name.
# "reviewed" runs are graded by reading each answer (eval/review.py); the others by the automatic grader.
RUN_LABELS = {
    "A_slm_only_test_answer": "A: SLM only",
    "B_dense_rag_test_answer": "B: dense RAG",
    "C_full_test_answer": "C: full system",
    "C_lean_test_answer": "C_lean: before the answer check",
    "C_verified_test_answer": "C_verified: with the answer check",
    "C_lean_test_answer_reviewed": "C_lean: before the answer check",
    "C_verified_test_answer_verified_reviewed": "C_verified: with the answer check",
}


class AskRequest(BaseModel):
    question: str
    variant: str = DEFAULT_VARIANT


class VerifyRequest(BaseModel):
    answer: str
    passage: str
    question: str = ""


def _check_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise HTTPException(400, f"Unknown variant. Choose one of {sorted(VARIANTS)}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "pack": pipeline.config.get("pack_name"), "variants": sorted(VARIANTS)}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    _check_variant(request.variant)
    with busy:
        ollama.ensure()
        answer = pipeline.ask(request.question, request.variant)
    return {"answer": answer.text, "refused": answer.refused, "verdict": answer.verdict,
            "confidence": answer.confidence, "citations": [[c.doc_id, c.page] for c in answer.citations],
            "timings_ms": {k: round(v) for k, v in answer.timings_ms.items()}, "details": answer.details}


# Section: demo API


def _runs_dir() -> Path:
    return resolve_path(pipeline.config, pipeline.config["paths"]["data_dir"]) / "runs" / pipeline.config["pack_name"]


def _eval_rows(split: str) -> list[dict]:
    """Questions of a split, if the pack's data is present on this machine."""
    data = pipeline.config.get("data", {})
    try:
        rows = [json.loads(line) for line in resolve_path(pipeline.config, data["eval_set"]).read_text(encoding="utf-8").splitlines()]
        split_ids = json.loads((resolve_path(pipeline.config, data["raw_dir"]) / "split.json").read_text(encoding="utf-8"))
    except (KeyError, OSError, ValueError):
        return []
    wanted = set(split_ids.get(split, []))
    return [r for r in rows if r["id"] in wanted]


@app.get("/api/info")
def info() -> dict:
    index_dir = pipeline.index_dir
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8")) if (index_dir / "manifest.json").exists() else {}
    vocab = pipeline.vocab if (index_dir / "vocab.json").exists() else {}
    dev = _eval_rows("dev")
    examples = [{"id": r["id"], "question": r["question"], "group": r.get("group")}
                for r in random.Random(7).sample(dev, min(len(dev), 50))]
    return {"pack": pipeline.config.get("pack_name"), "variants": sorted(VARIANTS), "default_variant": DEFAULT_VARIANT,
            "documents": manifest.get("documents"), "chunks": manifest.get("chunks"),
            "companies": sorted(vocab.get("doc_fields", {}).get("company", [])),
            "slm": pipeline.config["generator"]["model"], "examples": examples}


def _chunk_view(chunk, retrieval=None) -> dict:
    meta = chunk.metadata
    return {"id": chunk.id, "doc_id": chunk.doc_id, "page": chunk.page_start, "type": chunk.chunk_type,
            "company": meta.get("company"), "year": meta.get("year"), "section": meta.get("section"),
            "score": retrieval.scores.get(chunk.id) if retrieval else None,
            "statement_slot": bool(retrieval and chunk.id in retrieval.statement_ids),
            "text": chunk.text}


def _events(question: str, variant: str):
    """Run the pipeline in a thread and yield one server-sent event per stage."""
    events: queue.Queue = queue.Queue()

    def on_stage(name: str, data: dict) -> None:
        if name == "retrieved":
            retrieval = data["retrieval"]
            data = {"chunks": [_chunk_view(c, retrieval) for c in retrieval.chunks],
                    "candidates": len(retrieval.candidates), "confidence": retrieval.confidence,
                    "verdict": retrieval.verdict, "widened": retrieval.widened, "timings_ms": data["timings_ms"]}
        events.put((name, data))

    def work() -> None:
        try:
            with busy:
                events.put(("status", {"message": "Starting the local SLM server" if not ollama.server else "Running"}))
                ollama.ensure()
                answer = pipeline.ask(question, variant, on_stage=on_stage)
            events.put(("done", {"text": answer.text, "refused": answer.refused,
                                 "citations": [[c.doc_id, c.page] for c in answer.citations],
                                 "timings_ms": {k: round(v) for k, v in answer.timings_ms.items()},
                                 "verification": answer.details.get("verification"),
                                 "unverified_answer": answer.details.get("unverified_answer"),
                                 "slm": answer.details.get("slm")}))
        except Exception as error:  # shown in the page, not hidden
            events.put(("error", {"message": f"{type(error).__name__}: {error}"}))
        events.put(None)

    threading.Thread(target=work, daemon=True).start()
    if busy.locked():
        yield _sse("status", {"message": "Waiting for the previous question to finish"})
    while (item := events.get()) is not None:
        yield _sse(*item)


def _sse(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, default=str)}\n\n"


@app.get("/api/ask/stream")
def ask_stream(question: str, variant: str = DEFAULT_VARIANT) -> StreamingResponse:
    _check_variant(variant)
    if not question.strip():
        raise HTTPException(400, "Empty question")
    return StreamingResponse(_events(question.strip()[:1000], variant), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/verify")
def verify(request: VerifyRequest) -> dict:
    """The answer check on its own, for the demo's "Try the check" panel. No model runs."""
    from core.types import Chunk
    from generate.verify import verify_answer

    passage = Chunk(id="demo:p1:c0", doc_id="demo", text=request.passage[:20000], page_start=1, page_end=1)
    return verify_answer(request.answer[:5000], [passage], request.question[:1000]).as_dict()


@app.get("/api/chunk/{chunk_id}")
def chunk(chunk_id: str) -> dict:
    found = pipeline.chunk_store.get(chunk_id)
    if found is None:
        raise HTTPException(404, "No such chunk")
    return _chunk_view(found)


def _outcome(record: dict) -> str:
    if record.get("refused"):
        return "refused"
    correct = (record.get("grade") or {}).get("correct")
    return "ungraded" if correct is None else ("correct" if correct else "wrong")


@app.get("/api/runs")
def runs() -> list[dict]:
    from eval.reports import answer_metrics, load_records, retrieval_metrics

    found = []
    for path in sorted(_runs_dir().glob("*_answer*.jsonl")):
        records = load_records(path)
        answers = answer_metrics(records)
        if not answers:
            continue
        hit5 = retrieval_metrics(records).get("hit@5")
        found.append({"name": path.stem, "label": RUN_LABELS.get(path.stem, path.stem), "featured": path.stem in RUN_LABELS,
                      "order": list(RUN_LABELS).index(path.stem) if path.stem in RUN_LABELS else 99,
                      "n": len(records), "accuracy": answers["accuracy"], "ci95": answers["accuracy_ci95"],
                      "refusal_rate": answers["refusal_rate"], "hallucination_rate": answers["hallucination_rate"],
                      "precision": round(1 - answers["wrong_when_answered"], 3) if answers["wrong_when_answered"] is not None else None,
                      "hit5": hit5, "by_method": answers["by_method"],
                      "grading": "reviewed" if "review" in answers["by_method"] else "automatic"})
    return sorted(found, key=lambda r: (r["order"], r["name"]))


@app.get("/api/runs/{name}")
def run(name: str) -> list[dict]:
    from eval.reports import load_records

    path = _runs_dir() / f"{Path(name).name}.jsonl"
    if not path.exists():
        raise HTTPException(404, "No such run")
    out = []
    for r in load_records(path):
        gold_pages = {(e["doc_id"], int(e["page"])) for e in r.get("evidence", [])}
        verification = r.get("verification") or (r.get("details") or {}).get("verification")
        out.append({"id": r["id"], "group": r.get("group"), "question": r["question"], "gold": r["gold"],
                    "prediction": r.get("prediction"), "unverified": r.get("unverified_prediction") or
                    (r.get("details") or {}).get("unverified_answer"),
                    "outcome": _outcome(r), "method": (r.get("grade") or {}).get("method"),
                    "why": (r.get("grade") or {}).get("why"),
                    "evidence_in_top5": any((c.get("doc_id"), c.get("page")) in gold_pages for c in r.get("chunks", [])),
                    "evidence": sorted(gold_pages), "chunks": r.get("chunks", []), "verification": verification})
    return out


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")
