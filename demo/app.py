"""Minimal Streamlit demo. The main demo is demo/web (served by api/server.py).

    .venv\\Scripts\\streamlit.exe run demo/app.py

Ask a question over an ingested pack and see the answer, the evidence gate's
verdict, the cited pages, the chunks the SLM saw and per-stage latency.
The Ollama server starts on demand and stops after 5 idle minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st  # noqa: E402

from smallproof.core.ollama_server import OnDemandOllama  # noqa: E402
from smallproof.core.pipeline import DEFAULT_VARIANT, VARIANTS, Pipeline  # noqa: E402

PACKS = sorted(p.name for p in (Path(__file__).resolve().parent.parent / "packs").iterdir() if (p / "config.yaml").exists())


@st.cache_resource
def load(pack: str):
    pipeline = Pipeline.from_pack(pack)
    return pipeline, OnDemandOllama(pipeline.config)


st.title("SmallProof")
pack = st.sidebar.selectbox("Domain pack", PACKS)
variant = st.sidebar.selectbox("System variant", sorted(VARIANTS), index=sorted(VARIANTS).index(DEFAULT_VARIANT))
question = st.text_input("Question", "What is the FY2018 capital expenditure amount (in USD millions) for 3M?")

if st.button("Ask") and question.strip():
    pipeline, ollama = load(pack)
    with st.spinner("Retrieving evidence and asking the local SLM ..."):
        ollama.ensure()
        answer = pipeline.ask(question, variant)
    (st.warning if answer.refused else st.success)(answer.text)
    left, right = st.columns(2)
    left.metric("Gate verdict", str(answer.verdict or "gate off"))
    right.metric("Evidence score", f"{answer.confidence:.2f}" if answer.confidence is not None else "n/a")
    st.write("**Citations:**", ", ".join(f"{doc} p. {page}" for doc, page in [(c.doc_id, c.page) for c in answer.citations]) or "none")
    st.write("**Query tags:**", answer.details.get("tags"))
    with st.expander("Chunks given to the SLM"):
        store = pipeline.chunk_store
        for ref in answer.details.get("chunks", []):
            chunk = store.get(ref["id"])
            st.markdown(f"**{ref['doc_id']}, page {ref['page']}** ({chunk.chunk_type if chunk else ''})")
            st.text(chunk.text[:1500] if chunk else "")
    st.write("**Latency (ms):**", {k: round(v) for k, v in answer.timings_ms.items()})
