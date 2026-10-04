# Project plan

Version 1, 2026-09-27. Status: scaffold created, Phase 0 not started.

2026-10-04: named SmallProof. The library moved under src/smallproof/ (import
smallproof.core, smallproof.query, ...), pyproject.toml added (editable
install, `smallproof` CLI), no license yet. Module paths in the log below
(core/..., query/..., generate/...) predate the move and now live under
src/smallproof/.

2026-10-03: the answer check (decision 13) and an interactive web demo
(demo/web, served by api/server.py) added. Headline, test split, reviewed
grading: precision when answered 0.65 to 0.78 at the same 43 correct; wrong
answers 23 to 12 of 100.

2026-09-28: standard-library-only modules implemented ahead of Phase 0 (nothing
installed): core/types.py, core/interfaces.py, core/registry.py,
core/profiling.py, query/hybrid.py (RRF), generate/citations.py,
eval/metrics.py (without hallucination rate), eval/splits.py, with tests.

2026-09-28, Phase 0 progress:
- 0.1 done: CPU torch 2.14.0 and all Phase 0 libraries installed, 113 packages
  pinned in requirements.txt, venv measured at 1.33 GB, 38 tests pass under pytest.
- 0.4 to 0.7 scripts written (bench/). Model-free checks run and pass; the
  parser and index scripts were run on a generated PDF to prove the code works
  (numbers not reported).
- Waiting on Aman: model downloads (0.2), Ollama and the SLM (0.3), sample PDF.

2026-09-29, overnight run (Aman delegated all decisions and allowed downloads):
Phases 0 to 5 are done and measured; Phase 6 is partly done (CLI, FastAPI,
Streamlit, README; no pyproject yet because the package name is Aman's call).

### Results on FinanceBench (test split, 100 questions)

Setup: 84 filings, 12,010 pages, 31,999 chunks in one shared index; Qwen3 4B
Instruct 2507 Q4_K_M, context 4096, temperature 0; CPU models on 4 threads.
Grading: numeric match within 2% where the gold answer has a number, else the
same SLM as judge (few questions). Hallucination = answered and graded wrong,
as a share of all questions. "Evidence in top 5" = a gold evidence page is
among the 5 chunks given to the SLM.

| System | Accuracy (95% CI) | Refused | Hallucinated | Evidence in top 5 | Cited a gold page | Median latency |
|---|---|---|---|---|---|---|
| A: SLM only (start of the gold filing, ~3,000 tokens) | 0.12 (0.06 to 0.19) | 0.74 | 0.14 | n/a | 0.19 | 2.3 s |
| B: dense RAG, top 5 | 0.30 (0.21 to 0.39) | 0.57 | 0.13 | 0.36 | 0.40 | 2.0 s |
| C: full system | 0.33 (0.24 to 0.42) | 0.48 | 0.19 | 0.46 | 0.40 | 6.1 s |
| C_lean: C without router and GLiNER | 0.36 (0.27 to 0.46) | 0.43 | 0.21 | 0.48 | 0.44 | 5.2 s (p95 9.3 s) |

Unanswerable questions (30, company swapped for one outside the corpus): B
refused 29, C refused 30, C_lean refused 29. SLM GPU memory: +3.1 GB (whole GPU
peak 7.4 of 8 GB with other apps). Disk: 6.53 GB of the 15 GB budget.

Retrieval ablations on test (no SLM; share of questions with evidence in top 1 / 3 / 5, and in the 30 candidates):

| Variant | top 1 | top 3 | top 5 | candidates | median retrieval time |
|---|---|---|---|---|---|
| C_full | 0.27 | 0.40 | 0.46 | 0.68 | 3.5 s |
| without BM25 | 0.25 | 0.39 | 0.46 | 0.68 | 3.3 s |
| without dense | 0.18 | 0.31 | 0.34 | 0.47 | 3.5 s |
| without reranker | 0.23 | 0.41 | 0.46 | 0.68 | 3.6 s |
| without metadata tags | 0.14 | 0.30 | 0.37 | 0.52 | 2.0 s |
| without router | 0.27 | 0.42 | 0.47 | 0.68 | 2.7 s |
| without GLiNER | 0.27 | 0.42 | 0.48 | 0.71 | 3.4 s |
| B: dense only | 0.13 | 0.31 | 0.36 | 0.62 | 0.02 s |

Reading: metadata tags (company and year, with an extra search inside the
tagged filing) are the component that matters; dense beats BM25; the reranker
mostly helps top 1; router and GLiNER cost time without helping retrieval on
this data. C vs B accuracy differences are inside the confidence intervals.
The main remaining weakness is the SLM refusing (NOT FOUND) when the evidence
needs a calculation.

### Decisions made (with the evidence)

1. SLM: qwen3:4b-instruct-2507-q4_K_M. Ollama: standalone zip in tools/ollama,
   started and stopped by the code (no installer, no setx). CUDA 13 libraries
   deleted: driver 560 uses CUDA 12 (confirmed in the server log).
2. Parser: pypdfium2 (permissive license). pdfplumber tables off (4.3 pages/s).
3. Chunk tagger: rules. Zero-shot tagging ran at 0.09 chunks/s on 4 threads,
   infeasible for 32k chunks (sanity check 7/10 on obvious examples).
4. Query stage runs sequentially: 3 threads were slower (1,209 vs 1,118 ms).
5. Gate: reranker-probability mode (NLI on a 2,000-token input takes 3.5 s).
   Threshold calibrated on dev (50 answerable + 30 unanswerable) = 0.0: the
   SLM's own NOT FOUND already refused all 30 dev unanswerable questions, so
   the gate adds nothing on this data. Kept in the code, off in effect.
6. Retrieval: extra search inside the tagged filing and tag boost 1.0 raised
   dev evidence-in-top-5 from 0.36 to 0.44. 45 fused candidates tested (0.40),
   kept 30.
7. Prompt v2 (pushes calculations) tested on dev: accuracy 0.24 vs 0.28,
   hallucination 0.24 vs 0.14. Kept the original prompt.
8. FinanceBench evidence pages are 0-based (52 of 60 checked, none 1-based);
   converted to 1-based.
9. Heat: 4 threads, 60/30 s duty cycle, GPU pause at 80 C, 25 min job limit,
   Ollama watchdog and job-object tie (see README).

10. 2026-09-30: statement slots (query/statement_pages.py, on by default in
    C_lean and C_full). Measured on dev before test. Dev: evidence in top 5
    0.46 to 0.62, accuracy 0.26 to 0.38, refusals 0.58 to 0.42, hallucination
    0.16 to 0.20. Test: evidence 0.48 to 0.61, accuracy 0.36 to 0.39,
    hallucination 0.21 to 0.27. Kept on: accuracy rises on both splits; the
    extra wrong answers are the SLM misreading statement tables. The keyword
    lists were drafted from test failures, then checked on dev.
11. Calculator post-check considered and dropped: of 11 wrong numeric answers
    with the evidence present, only 1 or 2 were arithmetic slips; the rest
    picked the wrong figure or period.

12. Year labels on multi-year table values (generate/table_years.py, config
    generator.table_years) tested on dev and turned off: accuracy 0.38 to
    0.36, hallucination 0.20 to 0.26. The SLM answered more often but not
    better. Kept in the code, off by default.

13. 2026-10-03: answer check (generate/verify.py, variant C_verified, now the
    default). Diagnosis first: on the saved C_lean test run, 27 of 100
    answers were confidently wrong and nothing checked the SLM's output.
    Of those, arithmetic slips, numbers with no source in the passages,
    values from the wrong year column, hedged answers ending in NOT FOUND,
    and answers cut off at 256 tokens made up most of the cases code can
    detect. The check redoes written calculations (and corrects the result),
    requires every number to come from the passages, the question or a
    checked calculation, compares table values with the year column the
    sentence names, and refuses hedged or cut-off answers. It knows numbers,
    years and arithmetic, nothing about finance. Built and tuned on dev only
    (six saved dev runs, rechecked offline with `api.cli recheck`). Then
    applied once to test; three parser gaps seen in test refusals were fixed
    afterwards (page numbers written in prose, ordinals like "61st", a sum
    of two equal values) and are disclosed here. Cost: about 3 ms per answer.
    Dev (automatic grading, fresh GPU run): accuracy 0.38 to 0.36,
    hallucination 0.20 to 0.10, precision when answered 0.66 to 0.78.
    Test (fresh GPU run, reviewed grading, see 15): correct 43 to 43, wrong
    23 to 12, refused 34 to 45, precision when answered 0.65 to 0.78. The
    check refused 10 wrong answers, corrected 1, refused 1 right one (a CAGR
    needs a square root). The 12 left use real numbers for the wrong line
    item, segment or formula, or judge a yes/no question wrongly. Leakage
    (bench/bench_verify.py): with one copied number made wrong by 2 to 25
    percent, 5 to 14 percent of 156 passing answers still pass, almost all
    because the wrong value also appears elsewhere in the passages.
    Not applied, to avoid more tuning on test: answers that correct
    themselves ("Correction:", "Wait:") look unreliable and could count as
    hedges; to be checked on a held-out set.
14. num_predict 256 to 768: long calculations were cut off mid-answer (3 of
    50 dev answers). With temperature 0 this only changes answers that hit
    the limit.
15. Grading reviewed by reading. The 4B self-judge passed answers that
    contradict themselves (dev 00517, 00807), and the numeric grader uses the
    gold answer's first number, which misgrades prose gold answers in both
    directions. eval/review.py stores one verdict per answered question with
    a reason (an LLM reviewer read each question, gold answer and answer; numbers
    within 1 percent of gold count as right, lists must be complete,
    contradictory, hedged or cut-off answers are wrong). A verdict is tied
    to a hash of the answer text. The reported test numbers use it.

### Still open for Aman

- ~~Package name~~ done 2026-10-04: SmallProof (src/smallproof/, pyproject.toml).
- Intent labels and hand-labelled ground truth: the router's labels in
  packs/financebench/config.yaml are provisional (LLM-drafted), so router
  accuracy and its confusion matrix are not measured yet.
- A clean held-out set: the test split has now informed two changes
  (statement-slot keywords, three verifier parser gaps). Any public claim is
  stronger with questions that never influenced a design decision.
- Spot-check of the review verdicts (LLM-graded; a second reader helps).
- Licenses before any commercial use: ModernBERT zero-shot training data mix,
  MS MARCO terms for the reranker.
- Second domain pack (Phase 6).

Facts marked **verified** were looked up on 2026-09-27 from the source named.
Everything else is an estimate or a proposal and is labelled that way.
No measured numbers exist yet.

## 1. Goal

A reusable, local-first Python library that combines a Jev-style decision layer
(small zero-shot classifiers on CPU), hybrid RAG and a local SLM on the GPU.
Classifiers and retrieval do the heavy lifting. The SLM only writes the final
answer from 3 to 5 chunks, with page citations. The standalone core comes first.
Integration into other projects is a separate, later plan.

## 2. Who does what

| Task | Who |
|---|---|
| Model downloads (Hugging Face models, Ollama app, SLM) | Aman, with the commands in this plan |
| Library installs (pip, including CPU PyTorch) | Assistant |
| Code, tests, benchmarks, eval runs | Assistant |
| Intent labels and hand-labelled ground truth | Aman |
| Approving each phase checkpoint | Aman |
| Small data files (for example the sample PDF) | Aman, or the assistant after a yes |

Safety net: all code and benchmarks run with `HF_HUB_OFFLINE=1`. If a model is
missing, the run stops with an error instead of downloading it, and the assistant asks
for the download.

## 3. Repo layout

The scaffold follows the flat layout from the brief. Before Phase 1 code starts,
everything moves under `src/<package_name>/` once the name is chosen. That move
costs nothing now, and names like `core`, `eval` and `api` are too generic for
an installed package.

| Folder | Holds | Phase |
|---|---|---|
| core/ | types, interfaces, config loader, backend registry, pipeline, profiling | 1 |
| ingest/ | PDF text and table parsers, chunker, chunk tagger, vocab builder, indexer | 1 |
| query/ | classifier, metadata tags, extractor, BM25, dense, hybrid (RRF), reranker, gate | 1, 3, 4 |
| generate/ | Ollama backend, citation checker, prompt templates | 2, 4 |
| eval/ | harness, splits, grading, metrics, ablations, reports | 2 to 5 |
| packs/ | domain packs, FinanceBench first | 1, 2, 6 |
| api/ | CLI and FastAPI server | 6 |
| demo/ | Streamlit app | 6 |
| bench/ | Phase 0 benchmark scripts and results | 0 |
| configs/ | default stack config | 0, 1 |
| tests/ | pytest | 1 onward |
| docs/ | this plan, later the domain pack guide | all |
| data/, models/ | local only, git-ignored | all |

## 4. Components (verified 2026-09-27)

| Stage | Candidate | Source, revision | Download | License (repo metadata) |
|---|---|---|---|---|
| Zero-shot classifier | MoritzLaurer/ModernBERT-base-zeroshot-v2.0 | HF, d421c45 | 0.30 GB | Apache-2.0 |
| Zero-shot, optional comparison | knowledgator/gliclass-modern-base-v3.0 | HF, ac36922 | 0.61 GB | Apache-2.0 |
| Extractor | urchade/gliner_small-v2.1 | HF, 4e09141 | 0.61 GB | Apache-2.0 |
| Extractor tokenizer | microsoft/deberta-v3-small (tokenizer and config only) | HF, a36c739 | about 2.5 MB | MIT |
| Embeddings | BAAI/bge-small-en-v1.5 | HF, 5c38ec7 | 0.14 GB | MIT |
| Reranker | cross-encoder/ms-marco-MiniLM-L6-v2 | HF, 233902d | 0.09 GB | Apache-2.0 |
| SLM, recommended | qwen3:4b-instruct-2507-q4_K_M | Ollama library | 2.5 GB | Apache-2.0 (Qwen/Qwen3-4B-Instruct-2507) |
| SLM, alternative | qwen3.5:4b (q4_K_M) | Ollama library | 3.4 GB | Apache-2.0 (Qwen/Qwen3.5-4B) |
| SLM, fallback | llama3.2:3b | Ollama library | check before pull | Llama 3.2 Community License |
| Ollama app | v0.34.4 (released 2026-09-23) | GitHub releases | installer 1.57 GB | confirm in 0.6 |
| PyTorch | 2.14.0 CPU build | PyTorch CPU index | PyPI Windows cp311 wheel is 124 MB | confirm in 0.6 |
| Eval data | FinanceBench open sample | HF PatronusAI/financebench, GitHub pdfs/ | questions 1 MB, all 368 PDFs 705 MB | CC BY-NC 4.0 |

Notes:
- **Why Qwen3 4B Instruct 2507.** It is Apache-2.0, text only, and the smallest
  4B option at 2.5 GB. It is the non-thinking variant, so it should not spend
  tokens on hidden reasoning. That behaviour gets confirmed in 0.5.
- **Qwen2.5-3B is ruled out.** Its license is "qwen-research" (verified), which
  restricts commercial use.
- **qwen3.5:4b** is newer and also Apache-2.0. It is a text plus image model
  (the vision part is unused here) and 0.9 GB larger. I have not verified how
  its quality compares to Qwen3. It can be tested in Phase 5 as a swap, one
  model at a time.
- **The license in repo metadata is only a first check.** Step 0.6 reads each
  model card for notes on its training data.
- **Some repos also carry ONNX and OpenVINO copies.** The ModernBERT repo is
  2.2 GB in total and the reranker repo is 0.89 GB. The commands below download
  only the files that are needed.
- **GLiNER small v2.1 has no tokenizer files.** The current GLiNER code (GitHub
  main) builds the model from the checkpoint and loads the tokenizer from the
  backbone name. So only the backbone's tokenizer and config files are needed,
  not its 286 MB of weights. This gets confirmed with the installed version in 0.4.
- **Zero-shot cost grows with the number of labels.** ModernBERT zero-shot is an
  NLI cross-encoder, so it runs one forward pass per label. GLiClass scores all
  labels in one pass. This matters for metadata tags, where the label list can
  be long. Step 0.5 measures 4, 8, 16 and 32 labels to decide.

## 5. Storage budget (15 GB cap, target under 10 GB)

| Item | Size | Basis |
|---|---|---|
| venv (CPU torch and libraries) | 2 to 3 GB | estimate from the brief, measured in 0.1 |
| Ollama app | unknown, the installer alone is 1.57 GB | measured in 0.3 |
| SLM (qwen3 4b instruct q4_K_M) | 2.5 GB | verified listing |
| HF models (classifier, GLiNER, tokenizer, embedder, reranker) | 1.14 GB | verified |
| GLiClass (optional) | 0.61 GB | verified |
| FinanceBench | at most 0.71 GB (the subset will be smaller) | verified total, subset computed in Phase 2 |
| Indexes and run outputs | unknown | measured in Phases 1 and 2 |

The verified subtotal is 4.35 GB, or 4.96 GB with GLiClass. It leaves out the
venv, the Ollama app and the indexes. The biggest unknown is Ollama's installed
size. The brief assumed 1 to 2 GB, but the installer is already 1.57 GB
compressed.

Current state: the `.venv` is 19 MB and D: has about 50 GB free.

## 6. Phase 0: setup and feasibility

Goal: prove every component runs on this laptop and measure what it costs,
before writing library code.

### 0.1 Library install (assistant)

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install transformers sentence-transformers gliner chromadb rank_bm25 pypdfium2 pdfplumber pymupdf ollama psutil nvidia-ml-py pyyaml pytest
```

- Torch goes in first, so no later package can pull a CUDA build.
- The global pip cache is reused, not purged (decided 2026-09-27).
- FastAPI, Streamlit and KeyBERT wait until a phase needs them. `gliclass` is
  installed only if the GLiClass comparison is chosen.
- PyMuPDF is installed for the parser comparison. Whether it stays depends on
  the license decision.
- After install: pin exact versions into `requirements.txt`, and run
  `hf download --help` to confirm the 0.2 commands match the installed CLI.
  Report the venv size.

### 0.2 Model downloads (Aman)

Run these in PowerShell from the project folder. `HF_HOME` puts the files in
`models\hf`, so they count toward the budget and stay off C:. Each revision is
pinned so results are reproducible.

```powershell
cd D:\SLM_Jev_Rag
.\.venv\Scripts\Activate.ps1
$env:HF_HOME = "D:\SLM_Jev_Rag\models\hf"

# Zero-shot classifier, 0.30 GB
hf download MoritzLaurer/ModernBERT-base-zeroshot-v2.0 config.json model.safetensors special_tokens_map.json tokenizer.json tokenizer_config.json --revision d421c4545a438fd006fb43f8b981c5d908faa1e1

# GLiNER small, 0.61 GB
hf download urchade/gliner_small-v2.1 gliner_config.json pytorch_model.bin --revision 4e091416cf7c3481db542c2a3d26156916f3a47f

# GLiNER tokenizer and config from its backbone, about 2.5 MB (no weights).
# No --revision here on purpose: GLiNER looks this model up by name, which needs the
# cache's "main" reference, and a download pinned to a commit does not create it.
hf download microsoft/deberta-v3-small config.json spm.model tokenizer_config.json

# Embeddings, 0.14 GB
hf download BAAI/bge-small-en-v1.5 config.json config_sentence_transformers.json modules.json sentence_bert_config.json special_tokens_map.json tokenizer.json tokenizer_config.json vocab.txt model.safetensors 1_Pooling/config.json --revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a

# Reranker, 0.09 GB
hf download cross-encoder/ms-marco-MiniLM-L6-v2 config.json model.safetensors special_tokens_map.json tokenizer.json tokenizer_config.json vocab.txt --revision 233902d25c440f23af6f7d6e94d2946bac0bee0a

# Optional comparison: GLiClass, 0.61 GB
hf download knowledgator/gliclass-modern-base-v3.0 config.json model.safetensors special_tokens_map.json tokenizer.json tokenizer_config.json --revision ac369222ca4375ca66ebaf7fb5220f223514c035

# Sample PDF for parser benchmarks: FinanceBench 3M 2018 10-K, 1.25 MB
Invoke-WebRequest "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/3M_2018_10K.pdf" -OutFile data\samples\3M_2018_10K.pdf
```

Checked on 2026-09-28 against the installed CLI (huggingface-hub 1.33.0):
`hf download REPO_ID [FILENAMES]... --revision` matches these commands.
After downloading, run `.venv\Scripts\python.exe bench\env_check.py`.

### 0.3 Ollama and the SLM (Aman)

1. Download `OllamaSetup.exe` (1.57 GB, v0.34.4) from ollama.com/download.
   Optional: install the app on D: with the documented flag
   `OllamaSetup.exe /DIR="D:\Ollama"`.
2. Store models inside the project:
   `setx OLLAMA_MODELS "D:\SLM_Jev_Rag\models\ollama"`. Then quit Ollama from
   the tray and start it again, because it reads the variable at startup.
3. In a new terminal: `ollama pull qwen3:4b-instruct-2507-q4_K_M` (2.5 GB).
4. Tell the assistant when this is done.

Driver 560.94 meets Ollama's minimum of 550 (verified in Ollama's GPU docs).

### 0.4 Environment check (assistant, `bench/env_check.py`)

- The torch version ends in `+cpu`, `torch.cuda.is_available()` is False, and
  no `nvidia-*` pip packages are installed.
- Every HF model loads with `HF_HUB_OFFLINE=1`. This proves nothing is missing,
  including the GLiNER tokenizer.
- No new files appear in `C:\Users\AMAN\.cache\huggingface`. This proves the
  downloads went to the project folder.
- Ollama responds, and `ollama ps` shows the model running 100% on the GPU.

### 0.5 Benchmarks (assistant)

Protocol:
- The laptop is on AC power, in the same Windows power mode for every run.
  Both are recorded in each result file.
- Each component runs in its own process. Cold load time is measured once.
  Then come 2 warmup runs and 20 timed runs, reported as median and p95.
- Peak RAM is the Windows peak working set of the process (from psutil).
- VRAM: total GPU memory used, sampled every 50 ms. The report gives the idle
  baseline (955 MiB was in use by other apps when checked), the peak and the
  difference. Windows often does not report VRAM per process, so the
  difference is the number we use.
- Inputs: pages and chunks from the 3M 2018 10-K, plus 10 hand-written
  questions in `bench/fixtures/`. The assistant writes the questions and shows them
  to you.
- Every result JSON records hardware, versions, thread counts and settings.

| Script | Measures |
|---|---|
| bench_parsers.py | pages per second for PyMuPDF and pypdfium2, pdfplumber table time per page, text samples to check quality by eye |
| bench_embedder.py | one query; 100 chunks in batches of 32 |
| bench_reranker.py | 20 and 50 candidates |
| bench_classifier.py | one query against 4, 8, 16 and 32 labels; tagging 100 chunks with the 4 chunk types; one 2,000-token input (the size the gate will see); GLiClass too if downloaded |
| bench_extractor.py | GLiNER on a question and on a 400-token chunk, 5 entity types |
| bench_indexes.py | BM25 and Chroma build over about 500 chunks, query latency, index size on disk |
| bench_slm.py | time to first token, tokens per second, total time for a 5-chunk prompt of about 2,000 tokens, context 4096, temperature 0, VRAM |
| bench_query_parallel.py | classifier, GLiNER and query embedding run sequentially vs in threads, at a few torch thread settings |
| disk_report.py | size of each folder vs the budget |

Smoke check (no accuracy claims): chunk-type tagging on 10 chunks whose type is
obvious, using the four types from the brief, and GLiNER on 10 sentences. This
is only a sanity check. Accuracy is measured in Phase 3.

### 0.6 License review (assistant)

One table: component, license, whether commercial use is allowed, source link,
and notes from the model card. It covers models, libraries (from installed
package metadata), Ollama and FinanceBench.

### 0.7 Checkpoint

A chat summary with:
- the final stack
- a table of measured latency, RAM, VRAM and disk, with the setup used
- the license table
- open issues

Raw JSON stays in `bench/results/`.

Decisions for Aman at this point:
- parser backend
- SLM
- classifier (ModernBERT or GLiClass)
- whether running the query stage in parallel is worth it

## 7. Phase 1: core and ingest

Goal: interfaces, config and an ingest pipeline that runs end to end on one PDF.

Tasks:
1. **Package setup.** Aman picks the package name. Move folders under
   `src/<name>/` and add `pyproject.toml` (editable install). Run `git init`
   if Aman wants it.
2. **Types** (`core/types.py`): `Chunk` (id, doc_id, text, page_start,
   page_end, chunk_type, type_score, metadata), `Answer` and `Verdict`.
3. **Interfaces** (`core/interfaces.py`): the six abstract classes from the brief.
4. **Config** (`core/config.py`): load YAML, merge the default config with a
   pack, and validate. Pinned model revisions come from the config.
5. **Registry** (`core/registry.py`): maps a backend name to a class.
6. **Parsers:** page text using the backend chosen in 0.7, plus pdfplumber
   tables. Each table becomes one text chunk. A long table is split by rows,
   with the header repeated.
7. **Chunker:** counts tokens with the bge tokenizer, so chunks fit the
   embedder's 512-token limit. Proposed start: 350 tokens with 50 tokens of
   overlap. This is a starting point to tune in Phase 3 using recall@k, not a
   known best value. Chunks never cut through a table, and each chunk keeps its
   page range.
8. **Chunk tagger (proposal):** tables found by the parser are tagged `table`
   directly, and the classifier tags everything else. Store the top label and
   its score.
9. **Vocab builder:**
   - years: regex
   - sections: regex for 10-K "Item N" headings, plus heading detection
   - entities: document-level from the pack's document list (for example the
     company name); chunk-level from GLiNER only if 0.5 shows the ingest cost
     is acceptable
10. **Indexer:** normalized bge embeddings, one persistent Chroma collection per
    pack (fed our own embeddings), and the BM25 corpus saved to disk.
11. **Tests:**
    - chunker: no empty chunks, valid page ranges, correct overlap, tables kept whole
    - config validation and registry
    - parser on a small fixture PDF
    - BM25 save and load
    - Tests that need models are marked and skipped when a model is missing.

Checkpoint: spot-check 20 chunks from the 3M 10-K, including at least 5 tables
and every chunk type that appears. They can be shown in chat or saved to a
file (your choice). Also reported: ingest time and index size for one document.

Decisions: package name, parser backend (from 0.7), starting chunk size.

## 8. Phase 2: baselines and eval harness

Goal: measure how far simple systems get, so the full system (C) has something
to beat.

Tasks:
1. **Eval data:** download the FinanceBench questions (1 MB). Work out exactly
   which PDFs the 150 questions reference and their total size, then ask before
   downloading them.
2. **Corpus setup (decision):** one shared index over all the needed documents,
   or one index per document. Proposal: a shared index. It is harder, more
   realistic, and it is where metadata tags earn their place.
3. **Splits:** fixed seed, dev and test, stratified by FinanceBench question
   type. Proposal: 50 dev and 100 test. All tuning and threshold calibration
   use dev only. Results are reported on test.
4. **Unanswerable set (proposal):** for some questions, remove the gold
   document from the index, so the right behaviour is to refuse. Without this,
   refusal quality can't be measured, because all 150 questions are answerable.
5. **Grading (decision).** Options:
   - a) Aman grades by hand. Most reliable, but it means 100+ answers per variant.
   - b) Automatic numeric match with a tolerance, plus manual checks for the rest.
   - c) A local LLM judge. Weak, and biased because it would be the same model family.

   Proposal: b, with Aman checking a random sample to estimate the automatic
   grader's error rate.
6. **Harness:** runs one variant over a split and saves one JSONL record per
   question (answer, citations, verdict, retrieved chunk ids, per-stage
   timings, peak VRAM). Metrics are computed from the saved records, so they
   can be recomputed without running the models again.
7. **Baseline A, SLM only.** Interpretation to confirm: the gold document's
   text, cut to fit the 4096-token context, is passed to the SLM with the
   question.
8. **Baseline B, plain dense RAG:** bge-small top 5 chunks, then the SLM, with
   the same answer prompt as C.
9. **Answer prompt and citation checker:** answers cite pages like `[p. 12]`.
   A citation to a page that is not in the given chunks gets flagged.

Checkpoint: A and B numbers with their setup. Metrics are answer accuracy,
citation match (cited pages vs FinanceBench evidence pages), latency and VRAM.

Decisions: corpus setup, split sizes, grading method, Baseline A
interpretation, unanswerable set.

## 9. Phase 3: query stage

Goal: better retrieval, with CPU classifiers doing the work.

Tasks:
1. **Intent labels:** Aman defines the label set and hand-labels the ground
   truth. The assistant can build a small labelling helper if you want one.
2. **Intent classifier:** zero-shot, using label descriptions as hypotheses.
   What each intent changes (for example chunk-type boosts or candidate
   counts) gets agreed with Aman once the labels exist.
3. **Metadata tags:** picked from the ingest vocabulary. They act as a soft
   boost during fusion, with no boost below a confidence threshold tuned on
   dev. If 0.5 shows zero-shot is too slow with many labels, company and year
   come from GLiNER plus fuzzy matching against the vocabulary instead.
4. **GLiNER extraction:** entities and keywords feed BM25 query terms and the
   metadata tags.
5. **Hybrid retrieval:** BM25 top N plus dense top N, merged with reciprocal
   rank fusion (starting constant k = 60), plus metadata boosts.
6. **Reranker:** reorders the top M candidates and keeps the top 5.
7. **Retrieval eval:** recall@k, meaning the gold evidence page appears in the
   retrieved chunks. Measured for dense only, BM25 only, hybrid, hybrid with
   tags, and with the reranker.

Checkpoint: router accuracy and confusion matrix against Aman's labels, a
recall@k table, and per-stage latency.

## 10. Phase 4: evidence gate and fallback

Tasks:
1. **Gate signal (proposal, compared on dev):**
   - a) NLI entailment of "these passages answer the question" over the top chunks
   - b) the highest reranker score
   - c) both combined
2. **Verdicts:** sufficient, partial or insufficient, each with a confidence.
   Thresholds are calibrated on dev only.
3. **Fallback:** if the verdict is not sufficient, widen once (drop the
   metadata boosts, take more candidates, rerank again) and run the gate again.
   If it is still not sufficient, refuse clearly using the refusal template.
4. **Measured on test:**
   - hallucination rate
   - false refusals on answerable questions
   - correct refusals on unanswerable questions

Definition to agree before running (proposal): a hallucination is an answer
that is not a refusal and is graded wrong, or that cites a page that does not
support it.

Checkpoint: hallucination and refusal rates, with and without the gate.

Decision: when the second pass still says partial, answer with a caveat or
refuse (the brief says refuse).

## 11. Phase 5: full evaluation

- A vs B vs C on the test split, with the same prompt, SLM and settings.
- Ablations of C, removing one component at a time:
  - intent routing
  - metadata tags
  - GLiNER
  - BM25 (dense only)
  - dense (BM25 only)
  - reranker
  - gate
- Metrics:
  - answer accuracy
  - citation match
  - hallucination rate
  - refusal rate
  - per-stage latency (median and p95)
  - peak VRAM
- Counts are reported next to percentages. Key metrics get bootstrap
  confidence intervals, because a 100-question test set is small.
- Optional, one model at a time and if disk allows: an SLM swap (qwen3.5:4b or
  llama3.2:3b).
- Checkpoint: the results table. A written report only if Aman asks.

## 12. Phase 6: packaging

- `pyproject.toml` with extras: the core install, `[api]` (FastAPI), `[demo]`
  (Streamlit) and `[keybert]` (the extractor fallback).
- CLI (ingest, ask, eval), FastAPI server and Streamlit demo.
- README and a "write your own domain pack" guide.
- A second domain pack, picked by Aman, with its own small eval set.
- Package license, which depends on the PyMuPDF decision.
- Checkpoint: a clean install in a fresh venv, the demo runs, and the second
  pack runs end to end.

## 13. Open decisions

| # | Decision | Needed by | Recommendation |
|---|---|---|---|
| 1 | SLM | 0.3 | qwen3:4b-instruct-2507-q4_K_M |
| 2 | Download GLiClass for comparison | 0.2 | Yes if 0.61 GB is acceptable. It settles the long-label-list question with data. |
| 3 | Install the Ollama app on D: | 0.3 | Yes |
| 4 | Parser backend and package license | 0.7 | Decide with the 0.5 numbers |
| 5 | Package name | Phase 1 start | Aman: SmallProof (2026-10-04) |
| 6 | `git init` | Phase 1 start | Yes, local only, no remote until you say |
| 7 | Corpus setup, splits, grading, Baseline A, unanswerable set | Phase 2 | See Phase 2 |
| 8 | Intent labels and ground truth | Phase 3 | Aman |
| 9 | Handling a partial verdict, hallucination definition | Phase 4 | See Phase 4 |
| 10 | Second domain pack | Phase 6 | Aman |

## 14. Risks

| Risk | Mitigation |
|---|---|
| Ollama's installed size is larger than planned | Measure in 0.3, install on D:, drop GLiClass and unneeded PDFs if over budget |
| Zero-shot is too slow with many labels | Measure 4 to 32 labels; use GLiClass, or GLiNER plus vocabulary matching, for company and year |
| Zero-shot intent accuracy is weak | Tune label descriptions on dev; a small fine-tune only if Aman approves (needs labelled data and more disk) |
| The automatic grader makes mistakes | Aman checks a random sample; report agreement between grader and Aman |
| A small test set makes differences noisy | Report counts and bootstrap confidence intervals |
| FinanceBench is non-commercial | Never shipped in the package; downloaded by a script |
| Other apps use VRAM (955 MiB at the time of the check) | Record the baseline in every run; close heavy GPU apps during benchmarks |
| Parallel CPU stages slow each other down | Measured in 0.5 before designing around it |
