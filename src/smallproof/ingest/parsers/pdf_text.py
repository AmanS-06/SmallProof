"""Page-level text extraction.

pypdfium2 is the default (Apache/BSD license). PyMuPDF is AGPL, so it is only
an opt-in backend. Both return one string per page, in page order.
"""

from __future__ import annotations

import re
from pathlib import Path

BACKENDS = ("pypdfium2", "pymupdf")


def clean_page_text(text: str) -> str:
    """Normalise line endings and whitespace, keep line breaks (they separate table rows)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def extract_pages(pdf_path: str | Path, backend: str = "pypdfium2") -> list[str]:
    """Text of every page. Index 0 is page 1. HTML files use the HTML parser."""
    if Path(pdf_path).suffix.lower() in (".html", ".htm"):
        from smallproof.ingest.parsers.html_text import extract_html_pages

        return extract_html_pages(pdf_path)
    if backend == "pypdfium2":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            return [clean_page_text(pdf[i].get_textpage().get_text_range()) for i in range(len(pdf))]
        finally:
            pdf.close()
    if backend == "pymupdf":
        import pymupdf

        with pymupdf.open(str(pdf_path)) as doc:
            return [clean_page_text(page.get_text()) for page in doc]
    raise ValueError(f"Unknown text backend '{backend}'. Choose one of {BACKENDS}")
