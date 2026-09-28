"""FastAPI server.

    .venv\\Scripts\\python.exe -m uvicorn api.server:app --port 8000

POST /ask {"question": "...", "variant": "C_full"} returns the answer, verdict,
confidence, citations and per-stage timings. The Ollama server starts on the
first request and stops after 5 idle minutes (see core/ollama_server.py).
The pack comes from the PACK environment variable (default: financebench).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from core.ollama_server import OnDemandOllama  # noqa: E402
from core.pipeline import VARIANTS, Pipeline  # noqa: E402

pipeline = Pipeline.from_pack(os.environ.get("PACK", "financebench"))
ollama = OnDemandOllama(pipeline.config)
app = FastAPI(title="Local SLM + Jev-style RAG")


class AskRequest(BaseModel):
    question: str
    variant: str = "C_full"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "pack": pipeline.config.get("pack_name"), "variants": sorted(VARIANTS)}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    if request.variant not in VARIANTS:
        raise HTTPException(400, f"Unknown variant. Choose one of {sorted(VARIANTS)}")
    ollama.ensure()
    answer = pipeline.ask(request.question, request.variant)
    return {"answer": answer.text, "refused": answer.refused, "verdict": answer.verdict,
            "confidence": answer.confidence, "citations": [[c.doc_id, c.page] for c in answer.citations],
            "timings_ms": {k: round(v) for k, v in answer.timings_ms.items()}, "details": answer.details}
