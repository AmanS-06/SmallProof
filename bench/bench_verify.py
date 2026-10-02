"""How often does a wrong number get past the answer check?

Takes saved answers that pass the check, replaces one number copied from a
passage with a wrong value (off by 2 to 25 percent, same format), and checks
the answer again. A good check refuses almost all of them. No GPU, no models.

    python bench/bench_verify.py data/runs/financebench/C_lean_dev_answer.jsonl ...
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.pipeline import Pipeline  # noqa: E402
from generate.verify import verify_answer  # noqa: E402

OFFSETS = (0.02, 0.05, 0.10, 0.25)


def wrong_value(raw_digits: str, factor: float) -> str:
    plain = raw_digits.replace(",", "")
    decimals = len(plain.split(".")[1]) if "." in plain else 0
    value = float(plain) * factor
    text = f"{value:,.{decimals}f}" if "," in raw_digits else f"{value:.{decimals}f}"
    return text


def main(paths: list[str]) -> None:
    pipeline = Pipeline.from_pack("financebench")
    store = pipeline.chunk_store
    rng = random.Random(0)
    trials = {offset: [0, 0] for offset in OFFSETS}  # offset -> [accepted, total]
    answers = 0
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("refused") or "prediction" not in record:
                continue
            chunks = [store[c["id"]] for c in record.get("chunks", []) if c.get("id") in store]
            text = record["prediction"]
            base = verify_answer(text, chunks, record["question"])
            quoted = [n for n in base.numbers if n.status == "quoted"]
            if not base.ok or not quoted:
                continue
            answers += 1
            target = rng.choice(quoted)
            digits = text[target.start:target.end]
            core = "".join(ch for ch in digits if ch.isdigit() or ch in ",.").strip(",.")
            for offset in OFFSETS:
                factor = 1 + offset * rng.choice((-1, 1))
                changed = text[:target.start] + digits.replace(core, wrong_value(core, factor), 1) + text[target.end:]
                if changed == text:
                    continue
                trials[offset][1] += 1
                trials[offset][0] += verify_answer(changed, chunks, record["question"]).ok
    print(f"answers that pass the check and copy at least one number: {answers}")
    for offset, (accepted, total) in trials.items():
        print(f"  one copied number off by {offset:>4.0%}: wrong answer accepted {accepted}/{total}"
              f" ({accepted / total:.0%})" if total else "")


if __name__ == "__main__":
    main(sys.argv[1:])
