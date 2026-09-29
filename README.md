# Local SLM + Jev-style RAG (name TBD)

A local-first Python library for question answering over documents. Small CPU
models do the heavy lifting: a zero-shot classifier (the Jev-style decision
layer), an entity extractor, hybrid retrieval and a reranker. A small language
model (SLM) on the GPU only writes the final answer from 3 to 5 chunks, with
page citations. No paid APIs; everything runs on one laptop (RTX 4060 Laptop,
8 GB VRAM).

Status: Phases 0 to 5 implemented and measured on FinanceBench; Phase 6
(packaging) partly done. See [docs/PLAN.md](docs/PLAN.md) for the plan and
decision log.

## How it works

```
Ingest (once per document set)
  PDF -> page text (pypdfium2) -> chunks (350 tokens, 50 overlap, never crossing a page)
      -> chunk type (table / definition / numeric / narrative) -> metadata (company, year,
         section, years mentioned) -> bge-small embeddings -> Chroma + BM25

Query (per question)
  1. CPU analysis: GLiNER entities, metadata tags picked from the ingest vocabulary
     (company, year, section), intent label (router)
  2. Hybrid retrieval: BM25 + dense, plus an extra search inside the tagged filing,
     merged with reciprocal rank fusion
  3. Soft boosts from confident tags (nothing is filtered out), then the MiniLM reranker;
     when the question needs a primary statement (balance sheet, income or cash flow)
     of the tagged filing, that page gets one of the last two slots of the top 5
  4. Evidence gate (score threshold calibrated on dev): widen once, then refuse
  5. The SLM (Qwen3 4B Instruct, Ollama, GPU) answers from the top 5 chunks, citing pages
```

Every stage sits behind an interface in `core/interfaces.py` and can be
switched off or swapped through the config; `core/pipeline.py` defines the
variants used for baselines and ablations.

## Stack

| Stage | Model or library | Size | License |
|---|---|---|---|
| Zero-shot classifier | MoritzLaurer/ModernBERT-base-zeroshot-v2.0 | 0.30 GB | Apache-2.0 (training data: see note) |
| Extractor | urchade/gliner_small-v2.1 (+ deberta-v3-small tokenizer) | 0.61 GB | Apache-2.0 |
| Embeddings | BAAI/bge-small-en-v1.5 | 0.14 GB | MIT |
| Reranker | cross-encoder/ms-marco-MiniLM-L6-v2 | 0.09 GB | Apache-2.0 (MS MARCO data: see note) |
| SLM | qwen3:4b-instruct-2507-q4_K_M via Ollama 0.34.4 | 2.5 GB | Apache-2.0 |
| Parsing | pypdfium2 (PyMuPDF optional, AGPL) | | Apache/BSD |
| Index | Chroma 1.5, rank_bm25 | | Apache-2.0 |

Notes: the ModernBERT zero-shot card says it was trained on the zeroshot-v2.0
data mix, and the reranker on MS MARCO, whose terms (to confirm) are
non-commercial. Check both before any commercial use. FinanceBench is
CC BY-NC 4.0 and is never shipped.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # CPU-only torch
```

Models: see the pinned `hf download` commands in docs/PLAN.md step 0.2 (they
go to `models/hf`). Ollama: extract the standalone Windows zip to
`tools/ollama`; the code starts and stops it by itself.

## Usage

```powershell
.\.venv\Scripts\python.exe packs\financebench\prepare.py --download --check   # data
.\.venv\Scripts\python.exe -m api.cli ingest --pack financebench
.\.venv\Scripts\python.exe -m api.cli ask --pack financebench "What is the FY2018 capital expenditure amount (in USD millions) for 3M?"
.\.venv\Scripts\python.exe -m api.cli eval --pack financebench --variant C_full --split test --mode answer
.\.venv\Scripts\python.exe -m uvicorn api.server:app --port 8000               # HTTP API
.\.venv\Scripts\streamlit.exe run demo/app.py                                   # demo
```

## Laptop safety (heat)

Long jobs overheated the laptop once, so all heavy work runs under limits
(`configs/default.yaml`, section `runtime`):
- torch uses 4 of 14 CPU threads
- duty cycle: 60 s of work, 30 s of rest
- the job pauses when the GPU reaches 80 C and resumes below 70 C
- every job stops after 25 minutes and resumes when run again
- the Ollama server runs only during a job, is killed by a 25 minute watchdog,
  unloads the model after 1 idle minute, and is tied to its parent process so
  it dies even if the job is force-killed (Windows job object)

## Phase 0 measurements

Laptop on AC power, Windows "Balanced" plan, i7-13650HX, RTX 4060 Laptop.
Median of 20 runs after 2 warmups unless noted. Thread counts differ: the
first four rows ran before the 4-thread cap was introduced.

| Component | Threads | Result |
|---|---|---|
| pypdfium2 / PyMuPDF text | 1 | 185 / 446 pages per s (3M 2018 10-K) |
| pdfplumber tables | 1 | 4.3 pages per s: too slow for ingest, so tables are off |
| bge-small | 14 | 13 ms per query, 17.8 chunks per s, 925 MB RAM |
| MiniLM reranker | 14 | 504 ms for 20 candidates, 1,475 ms for 50 |
| ModernBERT zero-shot | 4 | 572 / 1,164 / 2,334 / 4,845 ms for 4 / 8 / 16 / 32 labels; 3.5 s for a 2,000-token input |
| GLiNER small | 4 | 39 ms per question, 237 ms per 260-word chunk, 1.7 GB RAM |
| Query stage, 3 CPU tasks | 4 | sequential 1,118 ms vs parallel threads 1,209 ms: parallel not used |
| SLM (Qwen3 4B Q4) | GPU | warm: 1.36 s to first token, 53.9 tokens per s, 2.15 s total (1,905-token prompt); 100% on GPU; +3.1 GB VRAM |

## Results on FinanceBench (test split, 100 questions)

84 filings in one shared index. Details, ablations and decisions: docs/PLAN.md.

| System | Accuracy (95% CI) | Refused | Hallucinated | Evidence in top 5 | Median latency |
|---|---|---|---|---|---|
| A: SLM only | 0.12 (0.06 to 0.19) | 0.74 | 0.14 | n/a | 2.3 s |
| B: dense RAG | 0.30 (0.21 to 0.39) | 0.57 | 0.13 | 0.36 | 2.0 s |
| C: full system | 0.33 (0.24 to 0.42) | 0.48 | 0.19 | 0.46 | 6.1 s |
| C_lean: without router and GLiNER | 0.36 (0.27 to 0.46) | 0.43 | 0.21 | 0.48 | 5.2 s |
| C_lean + statement slots (2026-09-30, default) | 0.39 (0.30 to 0.49) | 0.34 | 0.27 | 0.61 | 5.3 s |

Statement slots (query/statement_pages.py): the right filing was found for 99 of 100
test questions, but its balance sheet, income or cash flow statement often ranked
below the top 5 because questions name a concept ("quick ratio", "capex") rather
than the statement. Giving those pages a reserved slot raised evidence in the top 5
from 0.46 to 0.62 on dev (0.48 to 0.61 on test) without changing hit@1 or hit@3.
Answers: dev accuracy 0.26 to 0.38, test 0.36 to 0.39. The cost is more attempted
answers that go wrong (hallucination 0.16 to 0.20 on dev, 0.21 to 0.27 on test):
the 4B model now sees the statement and sometimes misreads a column or a ratio.
Turn it off with the `C_lean_no_statements` variant or `statements: {}` in the pack.

On 30 unanswerable questions (company not in the corpus) the systems refused
29 to 30. Metadata tags are the component that matters most for retrieval.
