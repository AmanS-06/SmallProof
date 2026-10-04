"""Applies the answer verifier (generate/verify.py) to a saved run, without the GPU.

Each answered record is checked against the chunks the SLM saw:
- failed checks: the record becomes a refusal; the old answer is kept
- corrected calculations: the record gets the corrected answer and is graded again
- passed: unchanged
Records whose old grade still holds keep it, so no SLM judge is needed unless a
corrected answer has a gold answer without numbers (left ungraded then).
A run made with the check on (variant C_verified) is checked again from the
SLM's own text, so a newer verifier can be applied to an older run.
"""

from __future__ import annotations

from collections.abc import Callable

from smallproof.core.types import Chunk
from smallproof.eval.grading import grade
from smallproof.generate.verify import verify_answer

REFUSAL = "I found a possible answer but could not verify it against the documents, so I am not giving it."


def _slm_text(record: dict) -> tuple[str, bool]:
    """The SLM's own answer and whether the SLM itself refused, before any check."""
    details = record.get("details") or {}
    if details.get("verification"):  # checked when it ran: start again from the SLM's text
        return details.get("unverified_answer") or record["prediction"], False
    return record["prediction"], bool(record.get("refused"))


def recheck_record(record: dict, chunk_of: Callable[[str], Chunk | None], num_predict: int | None = None,
                   hedges: list[str] | None = None) -> dict:
    if "prediction" not in record:
        return record
    text, refused_by_slm = _slm_text(record)
    if refused_by_slm:
        return record
    chunks = [chunk for ref in record.get("chunks", []) if (chunk := chunk_of(ref["id"])) is not None]
    output_tokens = (record.get("details") or {}).get("slm", {}).get("slm_output_tokens")
    truncated = bool(num_predict and output_tokens and output_tokens >= num_predict)
    result = verify_answer(text, chunks, record["question"], truncated, hedges)
    new = {**record, "verification": result.as_dict()}
    if not result.ok:
        new.update({"unverified_prediction": text, "prediction": REFUSAL, "refused": True,
                    "citations": [], "grade": {"correct": False, "method": "refused"}})
        if record["gold"] == "UNANSWERABLE":
            new["grade"] = {"correct": True, "method": "unanswerable"}
    else:
        new.update({"prediction": result.text, "refused": False})
        if result.text != text:
            new["unverified_prediction"] = text
        if result.text != record["prediction"] or record.get("refused"):
            new["grade"] = grade(record["question"], record["gold"], result.text, False, None)
    return new
