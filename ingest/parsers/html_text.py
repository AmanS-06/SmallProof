"""Page-level text from an HTML filing (for example an SEC inline-XBRL 10-K).

Uses only the standard library. Pages are split where the HTML asks the
printer for a page break (CSS page-break-before / page-break-after, or
break-before / break-after: page), so page numbers match the printed filing.
Each table row becomes one line with its cells separated by spaces, which is
the same shape PDF text gives for a statement row:

    Total revenues 3,773.5 3,958.5 3,494.8

Hidden content (display:none, the XBRL header) is skipped.
"""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from ingest.parsers.pdf_text import clean_page_text

_BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "ul", "ol"}
_CELL = {"td", "th"}
_SKIP = {"script", "style", "head", "title", "ix:header"}
_BREAK = re.compile(r"(?:page-)?break-(before|after)\s*:\s*(always|page)", re.IGNORECASE)
_HIDDEN = re.compile(r"display\s*:\s*none", re.IGNORECASE)
_VOID = {"br", "hr", "img", "meta", "link", "input", "col"}


class _PageCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pages: list[list[str]] = [[]]
        self.stack: list[tuple[str, bool, bool]] = []  # (tag, hides content, breaks the page after it)
        self.hidden = 0
        self.cells = 0  # open table cells: inside one, block tags are only spaces

    def _new_page(self) -> None:
        if any(part.strip() for part in self.pages[-1]):
            self.pages.append([])

    def handle_starttag(self, tag, attrs):
        style = dict(attrs).get("style") or ""
        breaks = [m.group(1).lower() for m in _BREAK.finditer(style)]
        if "before" in breaks:
            self._new_page()
        hides = tag in _SKIP or bool(_HIDDEN.search(style))
        if tag in _VOID:
            if tag == "br" and not self.hidden:
                self.pages[-1].append("\n")
            if "after" in breaks:
                self._new_page()
            return
        self.stack.append((tag, hides, "after" in breaks))
        if hides:
            self.hidden += 1
        if tag in _CELL:
            self.cells += 1
        self._separator(tag)

    def handle_endtag(self, tag):
        # Close up to the matching tag (HTML in filings is not always well nested).
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                for open_tag, hides, breaks_after in reversed(self.stack[index:]):
                    if hides:
                        self.hidden -= 1
                    if open_tag in _CELL:
                        self.cells -= 1
                    if breaks_after:
                        self._new_page()
                del self.stack[index:]
                break
        self._separator(tag)

    def _separator(self, tag: str) -> None:
        if self.hidden:
            return
        if tag in ("tr", "table") or (tag in _BLOCK and not self.cells):
            self.pages[-1].append("\n")
        elif tag in _BLOCK or tag in _CELL:
            self.pages[-1].append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.pages[-1].append(data.replace("\n", " "))


def html_pages(html: str) -> list[str]:
    """Text of every page of an HTML document. Index 0 is page 1."""
    collector = _PageCollector()
    collector.feed(html)
    collector.close()
    pages = []
    for parts in collector.pages:
        text = unescape("".join(parts)).replace("\xa0", " ").replace(chr(0x200B), "")
        # Cells: "$ 1,445.1" split over cells reads better without the gap after "$".
        text = re.sub(r"\$\s+(?=[\d(])", "$", text)
        text = clean_page_text(text)
        if text:
            pages.append(text)
    return pages


def extract_html_pages(path: str | Path) -> list[str]:
    return html_pages(Path(path).read_text(encoding="utf-8", errors="ignore"))


def xbrl_facts(html: str) -> dict[str, str]:
    """A few cover-page facts tagged in inline XBRL (company name, fiscal year, ticker)."""
    facts = {}
    for name in ("EntityRegistrantName", "DocumentFiscalYearFocus", "TradingSymbol", "DocumentType"):
        match = re.search(rf'name="dei:{name}"[^>]*>(.*?)</ix:', html, re.IGNORECASE | re.DOTALL)
        if match:
            facts[name] = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", match.group(1)))).strip()
    return facts
