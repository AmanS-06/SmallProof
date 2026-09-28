# Phase 0 feasibility benchmarks

One script per component. Each writes a JSON file to bench/results/ with the
setup (hardware, versions, settings) and the measured numbers.

Protocol: laptop on AC power, 2 warmup runs, 20 timed runs, median and p95.
Each script runs in its own process so peak RAM is per component.
Models load offline (HF_HUB_OFFLINE=1), so nothing downloads silently.
Details: docs/PLAN.md, step 0.5.

## Run order

From the project root (no activation needed):

    .venv\Scripts\python.exe bench\env_check.py          # step 0.4, any time
    .venv\Scripts\python.exe bench\bench_parsers.py      # needs data\samples\3M_2018_10K.pdf
    .venv\Scripts\python.exe bench\bench_indexes.py      # needs the sample PDF, no models
    .venv\Scripts\python.exe bench\bench_embedder.py     # needs bge-small
    .venv\Scripts\python.exe bench\bench_reranker.py     # needs the MiniLM reranker
    .venv\Scripts\python.exe bench\bench_classifier.py   # needs ModernBERT zero-shot
    .venv\Scripts\python.exe bench\bench_extractor.py    # needs GLiNER and its tokenizer
    .venv\Scripts\python.exe bench\bench_query_parallel.py
    .venv\Scripts\python.exe bench\bench_slm.py          # needs Ollama and the SLM
    .venv\Scripts\python.exe bench\license_report.py     # step 0.6
    .venv\Scripts\python.exe bench\disk_report.py        # step 0.7

A script whose model is missing stops with a message pointing to docs/PLAN.md.
Inputs live in bench/fixtures/fixtures.json.
