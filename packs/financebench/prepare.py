"""Prepares the FinanceBench pack from financebench_merged.jsonl.

Writes to data/raw/financebench/:
- manifest.json: doc_id -> {company, year, doc_type}
- eval.jsonl: one row per question in a pack-neutral format:
  id, question, answer, group (question_type), doc_id, evidence [{doc_id, page}] with 1-based pages
- pdfs/<doc_id>.pdf: only the documents the questions need

FinanceBench's evidence_page_num base (0 or 1) is not assumed: --check finds
each evidence text in the PDF and reports which base matches.

Run: .venv\\Scripts\\python.exe packs\\financebench\\prepare.py --download --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "raw" / "financebench"
PDF_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{}.pdf"


def load_questions() -> list[dict]:
    lines = (RAW / "financebench_merged.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def needed_docs(questions: list[dict]) -> list[str]:
    docs = {q["doc_name"] for q in questions}
    docs |= {e["doc_name"] for q in questions for e in q.get("evidence", []) if e.get("doc_name")}
    return sorted(docs)


def build_manifest(questions: list[dict]) -> dict[str, dict]:
    manifest = {}
    for q in questions:
        manifest.setdefault(q["doc_name"], {"company": q["company"], "year": str(q["doc_period"]),
                                            "doc_type": str(q["doc_type"]).upper()})
    return manifest


def download(doc_ids: list[str]) -> None:
    pdf_dir = RAW / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for number, doc_id in enumerate(doc_ids, start=1):
        target = pdf_dir / f"{doc_id}.pdf"
        if not target.exists():
            urllib.request.urlretrieve(PDF_URL.format(doc_id), target)
        print(f"[{number}/{len(doc_ids)}] {doc_id} {target.stat().st_size / 2**20:.1f} MB")


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def check_page_base(questions: list[dict], sample: int = 60) -> Counter:
    """For each evidence item, is its text on page (n + 1) or page n (1-based)? Counts the answers."""
    from smallproof.ingest.parsers.pdf_text import extract_pages

    votes: Counter = Counter()
    page_cache: dict[str, list[str]] = {}
    for q in questions:
        for evidence in q.get("evidence", []):
            if sum(votes.values()) >= sample:
                return votes
            doc_id, n = evidence.get("doc_name") or q["doc_name"], evidence.get("evidence_page_num")
            path = RAW / "pdfs" / f"{doc_id}.pdf"
            if n is None or not path.exists():
                continue
            if doc_id not in page_cache:
                page_cache[doc_id] = [_alnum(page) for page in extract_pages(path)]
            pages = page_cache[doc_id]
            probe = _alnum(evidence.get("evidence_text", ""))[:60]
            if len(probe) < 30:
                continue
            zero_based = n < len(pages) and probe in pages[n]
            one_based = 0 <= n - 1 < len(pages) and probe in pages[n - 1]
            votes["0-based" if zero_based and not one_based else "1-based" if one_based and not zero_based
                  else "both" if zero_based else "neither"] += 1
    return votes


def write_eval(questions: list[dict], zero_based: bool) -> None:
    offset = 1 if zero_based else 0
    with (RAW / "eval.jsonl").open("w", encoding="utf-8") as handle:
        for q in questions:
            evidence = [{"doc_id": e.get("doc_name") or q["doc_name"], "page": e["evidence_page_num"] + offset}
                        for e in q.get("evidence", []) if e.get("evidence_page_num") is not None]
            row = {"id": q["financebench_id"], "question": q["question"], "answer": q["answer"],
                   "group": q.get("question_type", ""), "doc_id": q["doc_name"], "evidence": evidence,
                   "justification": q.get("justification", "")}
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="download the PDFs the questions need")
    parser.add_argument("--check", action="store_true", help="check the evidence page base against the PDFs")
    args = parser.parse_args()

    questions = load_questions()
    docs = needed_docs(questions)
    manifest = build_manifest(questions)
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(questions)} questions, {len(docs)} documents, groups {dict(Counter(q.get('question_type') for q in questions))}")
    if args.download:
        download(docs)
    zero_based = True
    if args.check:
        votes = check_page_base(questions)
        print(f"Evidence page base check: {dict(votes)}")
        zero_based = votes["0-based"] >= votes["1-based"]
    write_eval(questions, zero_based)
    print(f"Wrote eval.jsonl with pages converted from {'0' if zero_based else '1'}-based to 1-based")


if __name__ == "__main__":
    main()
