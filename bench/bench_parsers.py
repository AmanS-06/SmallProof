"""Parser benchmark: PyMuPDF vs pypdfium2 text speed, and pdfplumber table extraction.

Each parser runs in its own process. Text parsers: one untimed warmup pass over
the whole PDF, then one timed pass that records every page, which gives the
per-page median and p95. pdfplumber is much slower, so it gets a single timed
pass (no warmup); this is recorded in the result.

Run: .venv\\Scripts\\python.exe bench\\bench_parsers.py
"""

from __future__ import annotations

import time
from collections.abc import Callable

from common import SAMPLE_PDF, load_sample_chunks, run_in_child, save_result, summarize_latencies


def _timed_pass(extract_page: Callable[[int], str], n_pages: int) -> tuple[list[float], list[str]]:
    """Extract every page, timing each one."""
    times, texts = [], []
    for index in range(n_pages):
        start = time.perf_counter()
        texts.append(extract_page(index))
        times.append((time.perf_counter() - start) * 1000)
    return times, texts


def _summary(parser: str, times: list[float], texts: list[str], method: str) -> dict:
    return {
        "parser": parser,
        "method": method,
        "pages": len(times),
        "pages_per_second": round(len(times) / (sum(times) / 1000), 1),
        "page_ms": {key: round(value, 2) for key, value in summarize_latencies(times).items()},
        "chars_total": sum(len(text) for text in texts),
        "empty_pages": sum(1 for text in texts if not text.strip()),
        "sample_page_2": texts[min(1, len(texts) - 1)][:400],
    }


def bench_pymupdf(pdf_path: str) -> dict:
    import pymupdf

    doc = pymupdf.open(pdf_path)
    extract = lambda index: doc[index].get_text()  # noqa: E731
    _timed_pass(extract, len(doc))  # warmup
    return _summary("pymupdf", *_timed_pass(extract, len(doc)), "warmup pass + timed pass")


def bench_pypdfium2(pdf_path: str) -> dict:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    extract = lambda index: pdf[index].get_textpage().get_text_range()  # noqa: E731
    _timed_pass(extract, len(pdf))  # warmup
    return _summary("pypdfium2", *_timed_pass(extract, len(pdf)), "warmup pass + timed pass")


def bench_pdfplumber_tables(pdf_path: str) -> dict:
    import pdfplumber

    times, tables_per_page = [], []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            start = time.perf_counter()
            tables_per_page.append(page.extract_tables())
            times.append((time.perf_counter() - start) * 1000)
            page.close()  # frees the page's cached objects
    first_table = next((tables[0] for tables in tables_per_page if tables), None)
    result = _summary("pdfplumber tables", times, [""] * len(times), "single timed pass, no warmup")
    for key in ("chars_total", "empty_pages", "sample_page_2"):
        result.pop(key)
    result.update({
        "tables_found": sum(len(tables) for tables in tables_per_page),
        "pages_with_tables": [page + 1 for page, tables in enumerate(tables_per_page) if tables],
        "sample_table_rows": first_table[:4] if first_table else None,
    })
    return result


def main() -> None:
    load_sample_chunks()  # exits with a clear message if the PDF is missing
    results = [run_in_child(bench, str(SAMPLE_PDF)) for bench in (bench_pymupdf, bench_pypdfium2, bench_pdfplumber_tables)]
    save_result("parsers", {"pdf": SAMPLE_PDF.name, "results": results})
    for result in results:
        print(f"{result['parser']}: {result['pages_per_second']} pages/s, "
              f"median {result['page_ms']['median_ms']} ms/page, peak RAM {result['peak_ram_mb']} MB")


if __name__ == "__main__":
    main()
