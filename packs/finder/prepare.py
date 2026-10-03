"""Prepares the FinDER pack: a held-out set that no design decision has seen.

FinDER (Linq AI Research, CC BY-NC 4.0): 5,703 expert questions on 10-K
filings, with the evidence excerpts and the filings themselves (inline-XBRL
HTML, one per ticker). Downloaded to data/raw/finder/ (never committed).

The sample is drawn once and frozen, before any system runs on it:
1. 40 filings picked at random (seed 42) from the 497.
2. A question belongs to a filing when its evidence text is found in that
   filing (and in no other of the 40); the pages holding it are the gold pages.
3. 2 questions per filing, taking categories in turn so all 8 are covered.

Writes to data/raw/finder/: eval.jsonl, manifest.json, docs/<TICKER>.html,
split.json (all 80 questions are test; there is no dev split on purpose).

Run: .venv\\Scripts\\python.exe packs\\finder\\prepare.py
"""

from __future__ import annotations

import json
import random
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ingest.parsers.html_text import html_pages, xbrl_facts  # noqa: E402

RAW = ROOT / "data" / "raw" / "finder"
SEED, N_FILINGS, PER_FILING = 42, 40, 2
_SUFFIX = re.compile(r"[,.]?\s+(inc|incorporated|corp|corporation|co|company|holdings|plc|ltd|limited|n\.v|s\.a)\.?$",
                     re.IGNORECASE)


def norm(text: str) -> str:
    """Letters and digits only, lower case: survives any whitespace or table formatting."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def probes(reference: str) -> list[str]:
    """Three 40-character pieces of an evidence excerpt (start, middle, end)."""
    text = norm(reference)
    if len(text) < 60:
        return []
    size = 40
    return [text[int(at * (len(text) - size)):][:size] for at in (0.1, 0.5, 0.9)]


def short_name(registrant: str) -> str:
    name = registrant.strip()
    for _ in range(2):
        name = _SUFFIX.sub("", name).strip(" ,.")
    return name


def main() -> None:
    import pandas as pd

    questions = pd.read_parquet(RAW / "data" / "train-00000-of-00001.parquet").to_dict("records")
    archive = zipfile.ZipFile(RAW / "10-k.zip")
    files = sorted(n for n in archive.namelist() if n.startswith("10k/") and n.endswith(".html"))
    rng = random.Random(SEED)
    picked = rng.sample(files, N_FILINGS)

    filings = {}
    for name in picked:
        html = archive.read(name).decode("utf-8", errors="ignore")
        pages = html_pages(html)
        facts = xbrl_facts(html)
        ticker = Path(name).stem
        filings[ticker] = {"html": html, "pages": [norm(page) for page in pages], "facts": facts,
                           "full": norm(" ".join(pages))}
        print(f"{ticker}: {len(pages)} pages, {facts.get('EntityRegistrantName')} {facts.get('DocumentFiscalYearFocus')}")

    by_filing: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        refs = [r for r in q["references"] if len(norm(r)) >= 60]
        if not refs:
            continue
        pieces = [p for r in refs for p in probes(r)]
        owners = [t for t, f in filings.items() if sum(p in f["full"] for p in pieces) >= max(2, len(pieces) * 2 // 3)]
        if len(owners) != 1:
            continue
        ticker = owners[0]
        pages = sorted({i + 1 for i, page in enumerate(filings[ticker]["pages"]) for p in pieces if p in page})
        if pages:
            by_filing[ticker].append({**q, "pages": pages})

    rows = []
    category_turn = 0
    categories = sorted({q["category"] for q in questions})
    for ticker in sorted(by_filing):
        candidates = by_filing[ticker][:]
        rng.shuffle(candidates)
        chosen = []
        for _ in range(len(categories)):  # take the next category in turn that this filing has
            if len(chosen) == PER_FILING:
                break
            wanted = categories[category_turn % len(categories)]
            category_turn += 1
            match = next((c for c in candidates if c["category"] == wanted and c not in chosen), None)
            if match:
                chosen.append(match)
        chosen += [c for c in candidates if c not in chosen][: PER_FILING - len(chosen)]
        for c in chosen:
            rows.append({"id": f"finder_{c['_id']}", "question": c["text"], "answer": c["answer"],
                         "group": c["category"], "type": c["type"], "doc_id": ticker,
                         "evidence": [{"doc_id": ticker, "page": page} for page in c["pages"]]})

    docs = RAW / "docs"
    docs.mkdir(exist_ok=True)
    manifest = {}
    for ticker in sorted({r["doc_id"] for r in rows}):
        f = filings[ticker]
        registrant = f["facts"].get("EntityRegistrantName", ticker)
        manifest[ticker] = {"company": short_name(registrant), "year": f["facts"].get("DocumentFiscalYearFocus", ""),
                            "doc_type": "10-K", "aliases": sorted({ticker, registrant})}
        (docs / f"{ticker}.html").write_text(f["html"], encoding="utf-8")
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (RAW / "eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (RAW / "split.json").write_text(json.dumps({"seed": SEED, "note": "held-out: every question is test",
                                                "dev": [], "test": [r["id"] for r in rows]}, indent=1), encoding="utf-8")
    counts = defaultdict(int)
    for r in rows:
        counts[r["group"]] += 1
    print(f"{len(rows)} questions over {len(manifest)} filings; by category {dict(counts)}")
    print(f"filings with no matched question: {sorted(set(filings) - set(by_filing))}")


if __name__ == "__main__":
    main()
