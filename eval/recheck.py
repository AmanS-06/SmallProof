"""Applies the answer verifier (generate/verify.py) to a saved run, without the GPU.

Each answered record is checked against the chunks the SLM saw:
- failed checks: the record becomes a refusal; the old answer is kept
- corrected calculations: the record gets the corrected answer and is graded again
- passed: unchanged
Records whose old grade still holds keep it, so no SLM judge is needed unless a
corrected answer has a gold answer without numbers (left ungraded then).
"""

from __future__ import annotations

from collections.abc import Callable

from core.types import Chunk
from eval.grading import grade
from generate.verify import verify_answer

REFUSAL = "I found a possible answer but could not verify it against the documents, so I am not giving it."


def recheck_record(record: dict, chunk_of: Callable[[str], Chunk | None], num_predict: int | None = None,
                   hedges: list[str] | None = None) -> dict:
    if record.get("refused") or "prediction" not in record:
        return record
    chunks = [chunk for ref in record.get("chunks", []) if (chunk := chunk_of(ref["id"])) is not None]
    output_tokens = (record.get("details") or {}).get("slm", {}).get("slm_output_tokens")
    truncated = bool(num_predict and output_tokens and output_tokens >= num_predict)
    result = verify_answer(record["prediction"], chunks, record["question"], truncated, hedges)
    new = {**record, "verification": result.as_dict()}
    if not result.ok:
        new.update({"unverified_prediction": record["prediction"], "prediction": REFUSAL, "refused": True,
                    "citations": [], "grade": {"correct": False, "method": "refused"}})
        if record["gold"] == "UNANSWERABLE":
            new["grade"] = {"correct": True, "method": "unanswerable"}
    elif result.text != record["prediction"]:
        new.update({"unverified_prediction": record["prediction"], "prediction": result.text})
        new["grade"] = grade(record["question"], record["gold"], result.text, False, None)
    return new
