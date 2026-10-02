"""Manual review of graded answers, stored next to the run.

The automatic grader has two weak spots found on 2026-10-03:
- numeric: it takes the gold answer's first number, which is wrong for prose
  gold answers ("No, margins fell from 36.8% to 34.6%" vs "fell to 34.6%")
- judge: the same 4B SLM passed answers that contradict themselves
So for reported numbers every answered question is reviewed by a reader (here
an LLM reviewer, reading the question, gold answer and answer) and the verdict is saved
in <run>.review.json as {id: {"correct": bool, "why": "..."}}. A review
applies only while the answer text is unchanged (a hash is stored).

    python -m eval.review dump  data/runs/financebench/C_verified_test_answer.jsonl
    python -m eval.review apply data/runs/financebench/C_verified_test_answer.jsonl
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.reports import load_records, save_records, summarize  # noqa: E402


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def review_path(run: Path) -> Path:
    return run.with_suffix(".review.json")


def dump(run: Path) -> None:
    """Print the answered questions that need a verdict."""
    reviews = json.loads(review_path(run).read_text(encoding="utf-8")) if review_path(run).exists() else {}
    for r in load_records(run):
        if r.get("refused") or r["gold"] == "UNANSWERABLE":
            continue
        known = reviews.get(r["id"])
        if known and known.get("hash") == text_hash(r["prediction"]):
            continue
        print(f"### {r['id']} [{r.get('group')}] auto={r.get('grade', {}).get('correct')} ({r.get('grade', {}).get('method')})")
        print("Q:", r["question"])
        print("GOLD:", r["gold"])
        print("ANSWER:", r["prediction"])
        print()


def apply(run: Path, verdicts: dict | None = None) -> Path:
    """Write <run>_reviewed.jsonl with reviewed grades. verdicts {id: [bool, why]} are added first."""
    records = load_records(run)
    path = review_path(run)
    reviews = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    by_id = {r["id"]: r for r in records}
    for rid, (correct, why) in (verdicts or {}).items():
        reviews[rid] = {"correct": bool(correct), "why": why, "hash": text_hash(by_id[rid]["prediction"])}
    path.write_text(json.dumps(reviews, indent=1), encoding="utf-8")
    missing = []
    for r in records:
        if r.get("refused") or r["gold"] == "UNANSWERABLE":
            continue
        review = reviews.get(r["id"])
        if review and review["hash"] == text_hash(r["prediction"]):
            r["grade"] = {"correct": review["correct"], "method": "review", "why": review["why"], "auto": r.get("grade")}
        else:
            missing.append(r["id"])
    out = run.with_name(run.stem + "_reviewed.jsonl")
    save_records(records, out)
    print(f"{out.name}: {len(records)} records, {len(missing)} answered without a review {missing[:5]}")
    print(json.dumps(summarize(records)["answers"], indent=1))
    return out


if __name__ == "__main__":
    command, run_file = sys.argv[1], Path(sys.argv[2])
    if command == "dump":
        dump(run_file)
    elif command == "apply":
        extra = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8")) if len(sys.argv) > 3 else None
        apply(run_file, extra)
