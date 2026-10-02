"""Command line interface.

    python -m api.cli ingest   --pack financebench
    python -m api.cli ask      --pack financebench "What was 3M's FY2018 capex?"
    python -m api.cli split    --pack financebench
    python -m api.cli unanswerable --pack financebench
    python -m api.cli eval     --pack financebench --variant C_verified --split test --mode answer
    python -m api.cli report   data/runs/financebench/C_full_test_answer.jsonl
    python -m api.cli recheck  --pack financebench --num-predict 256 data/runs/financebench/C_lean_dev_answer.jsonl
    python -m api.cli calibrate --pack financebench --run data/runs/financebench/C_no_gate_dev_answer.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import get, load_config, resolve_path  # noqa: E402
from core.ollama_server import OllamaServer  # noqa: E402
from core.pipeline import DEFAULT_VARIANT, VARIANTS, Pipeline  # noqa: E402
from core.thermal import ThermalGuard  # noqa: E402

SPLIT_SEED, DEV_SIZE = 42, 50
# Well-known companies used to build unanswerable questions; any that appear in the corpus are skipped.
OUTSIDE_COMPANIES = ["Tesla", "Toyota", "Nestle", "Siemens", "Samsung", "Unilever", "Sony", "Shell", "Airbus", "Heineken"]
UNANSWERABLE = "UNANSWERABLE"


def _pack_paths(config: dict) -> dict[str, Path]:
    data = config["data"]
    raw = resolve_path(config, data["raw_dir"])
    return {"raw": raw, "eval": resolve_path(config, data["eval_set"]), "split": raw / "split.json",
            "manifest": resolve_path(config, data["manifest"]), "pdfs": resolve_path(config, data["pdf_dir"]),
            "unanswerable": raw / "unanswerable.jsonl",
            "runs": resolve_path(config, config["paths"]["data_dir"]) / "runs" / config["pack_name"]}


def cmd_ingest(args) -> None:
    pipeline = Pipeline.from_pack(args.pack)
    paths = _pack_paths(pipeline.config)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    pdfs = sorted(p for p in paths["pdfs"].glob("*.pdf") if p.stem in manifest)[: args.limit or None]
    with ThermalGuard.from_config(pipeline.config) as guard:
        stats = pipeline.ingest(pdfs, manifest, guard=guard)
    print(json.dumps({k: v for k, v in stats.items() if k != "settings"}, indent=1))


def cmd_ask(args) -> None:
    pipeline = Pipeline.from_pack(args.pack)
    with OllamaServer.from_config(pipeline.config):
        answer = pipeline.ask(args.question, args.variant)
    print(answer.text)
    print(f"\nverdict={answer.verdict} confidence={answer.confidence} refused={answer.refused}")
    print("citations:", [(c.doc_id, c.page) for c in answer.citations])
    print("timings_ms:", {k: round(v) for k, v in answer.timings_ms.items()})
    print("tags:", answer.details.get("tags"))


def cmd_split(args) -> None:
    from eval.harness import load_eval_set
    from eval.splits import save_split, stratified_split

    paths = _pack_paths(load_config(args.pack))
    rows = load_eval_set(paths["eval"])
    dev, test = stratified_split([r["id"] for r in rows], [r["group"] for r in rows], DEV_SIZE, SPLIT_SEED)
    save_split(paths["split"], dev, test, SPLIT_SEED, note=f"stratified by group, dev {DEV_SIZE}", overwrite=args.force)
    print(f"dev {len(dev)}, test {len(test)} -> {paths['split']}")


def cmd_unanswerable(args) -> None:
    """Test questions with the company swapped for one that is not in the corpus: the right move is to refuse."""
    from eval.harness import load_eval_set
    from eval.splits import load_split

    config = load_config(args.pack)
    paths = _pack_paths(config)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    known = {meta["company"].lower() for meta in manifest.values()}
    outside = [name for name in OUTSIDE_COMPANIES if name.lower() not in known]
    dev_ids, test_ids = load_split(paths["split"])
    source_ids = dev_ids if args.split == "dev" else test_ids
    rows = {r["id"]: r for r in load_eval_set(paths["eval"])}
    rng, made = random.Random(SPLIT_SEED), []
    for row_id in rng.sample(sorted(source_ids), len(source_ids)):
        row = rows[row_id]
        company = manifest.get(row["doc_id"], {}).get("company", "")
        if company and re.search(re.escape(company), row["question"], re.IGNORECASE):
            new_company = outside[len(made) % len(outside)]
            question = re.sub(re.escape(company), new_company, row["question"], flags=re.IGNORECASE)
            made.append({"id": f"unans_{row_id}", "question": question, "answer": UNANSWERABLE,
                         "group": "unanswerable", "doc_id": "", "evidence": [], "source_id": row_id})
        if len(made) >= args.n:
            break
    target = paths["unanswerable"] if args.split == "test" else paths["raw"] / "unanswerable_dev.jsonl"
    target.write_text("".join(json.dumps(r) + "\n" for r in made), encoding="utf-8")
    print(f"{len(made)} unanswerable questions -> {target}")


def _rows_for(paths: dict, split: str) -> list[dict]:
    from eval.harness import load_eval_set
    from eval.splits import load_split

    if split == "unanswerable":
        return load_eval_set(paths["unanswerable"])
    if split == "unanswerable_dev":
        return load_eval_set(paths["raw"] / "unanswerable_dev.jsonl")
    rows = load_eval_set(paths["eval"])
    if split == "all":
        return rows
    dev, test = load_split(paths["split"])
    wanted = set(dev if split == "dev" else test)
    return [r for r in rows if r["id"] in wanted]


def cmd_eval(args) -> None:
    from eval.harness import run
    from eval.reports import add_grades, load_records, save_records, summarize

    pipeline = Pipeline.from_pack(args.pack)
    paths = _pack_paths(pipeline.config)
    rows = _rows_for(paths, args.split)[: args.limit or None]
    out = paths["runs"] / f"{args.variant}_{args.split}_{args.mode}.jsonl"
    with ThermalGuard.from_config(pipeline.config) as guard:
        if args.mode == "retrieval":
            run(pipeline, rows, args.variant, out, args.mode, guard=guard)
            records = load_records(out)
        else:  # the Ollama server lives only for this block (watchdog-limited)
            with OllamaServer.from_config(pipeline.config):
                run(pipeline, rows, args.variant, out, args.mode, guard=guard)
                records = add_grades(load_records(out), pipeline.generator)
            save_records(records, out)
    print(json.dumps(summarize(records), indent=1))


def cmd_report(args) -> None:
    from eval.reports import load_records, summarize

    print(json.dumps(summarize(load_records(args.run)), indent=1))


def cmd_recheck(args) -> None:
    """Run the answer verifier over saved runs (no GPU) and compare before and after."""
    from eval.recheck import recheck_record
    from eval.reports import answer_metrics, load_records, save_records

    pipeline = Pipeline.from_pack(args.pack)
    store = pipeline.chunk_store
    num_predict = args.num_predict or get(pipeline.config, "generator.num_predict", None)  # old runs: 256
    hedges = get(pipeline.config, "verify.hedges", None)
    for path in map(Path, args.run):
        before = load_records(path)
        after = [recheck_record(r, store.get, num_predict, hedges) for r in before]
        out = path.with_name(path.stem + "_verified.jsonl")
        save_records(after, out)
        keys = ("accuracy", "correct", "refusal_rate", "hallucination_rate", "wrong_when_answered", "by_method")
        print(path.name, "->", out.name)
        for name, records in (("before", before), ("after", after)):
            metrics = answer_metrics(records)
            print(f"  {name:6}", json.dumps({k: metrics.get(k) for k in keys}))
        changed = [(b, a) for b, a in zip(before, after) if a.get("verification") and
                   (a["refused"] != b["refused"] or a["prediction"] != b["prediction"])]
        for b, a in changed:
            outcome = lambda r: "refused" if r["refused"] else ("correct" if r["grade"]["correct"] else "wrong")
            print(f"  {a['id']}: {outcome(b)} -> {outcome(a)}; {'; '.join(a['verification']['reasons'] or a['verification']['corrections'])}")


def cmd_calibrate(args) -> None:
    """Pick the gate threshold t from dev runs made with the gate off (the evidence score is still recorded).

    The gate refuses when score < t. Utility per question:
    answerable:   +1 correct answer, -1 wrong answer, 0 refused (by the gate or by the SLM itself)
    unanswerable: +1 refused, -1 answered
    """
    from eval.reports import load_records

    records = [r for path in args.run for r in load_records(path) if r.get("confidence") is not None and "grade" in r]

    def utility(t: float) -> int:
        total = 0
        for r in records:
            refused = r["confidence"] < t or r["refused"]
            if r["gold"] == UNANSWERABLE:
                total += 1 if refused else -1
            elif not refused:
                total += 1 if r["grade"]["correct"] else -1
        return total

    candidates = sorted({0.0} | {r["confidence"] for r in records})
    t = max(candidates, key=lambda c: (utility(c), -c))  # on a tie, the lower threshold (answer more)
    unanswerable = [r for r in records if r["gold"] == UNANSWERABLE]
    print(json.dumps({"questions": len(records), "threshold": round(t, 4), "utility": utility(t),
                      "utility_without_gate": utility(0.0),
                      "answered": sum(1 for r in records if r["confidence"] >= t and not r["refused"]),
                      "unanswerable_refused": f"{sum(1 for r in unanswerable if r['confidence'] < t or r['refused'])}/{len(unanswerable)}"},
                     indent=1))
    if args.write:  # store the calibrated thresholds in the pack's config.yaml
        path = Path(load_config(args.pack)["pack_dir"]) / "config.yaml"
        old_block = re.compile(r"^gate:.*(?:\n  .*)*\n?", re.MULTILINE)  # a previous calibration, if any
        lines = [old_block.sub("", path.read_text(encoding="utf-8")).rstrip(), "",
                 f"gate: # calibrated on the dev split ({len(records)} questions) by `api.cli calibrate`",
                 f"  sufficient_threshold: {round(t, 4)}", f"  partial_threshold: {round(t / 2, 4)}", ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote gate thresholds to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="api.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ingest"); p.add_argument("--pack", required=True); p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_ingest)
    p = sub.add_parser("ask"); p.add_argument("question"); p.add_argument("--pack", required=True)
    p.add_argument("--variant", default=DEFAULT_VARIANT, choices=sorted(VARIANTS)); p.set_defaults(func=cmd_ask)
    p = sub.add_parser("split"); p.add_argument("--pack", required=True); p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_split)
    p = sub.add_parser("unanswerable"); p.add_argument("--pack", required=True); p.add_argument("--n", type=int, default=30)
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.set_defaults(func=cmd_unanswerable)
    p = sub.add_parser("eval"); p.add_argument("--pack", required=True)
    p.add_argument("--variant", default=DEFAULT_VARIANT, choices=sorted(VARIANTS) + ["A_slm_only"])
    p.add_argument("--split", default="dev", choices=["dev", "test", "all", "unanswerable", "unanswerable_dev"])
    p.add_argument("--mode", default="answer", choices=["answer", "retrieval"]); p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_eval)
    p = sub.add_parser("report"); p.add_argument("run"); p.set_defaults(func=cmd_report)
    p = sub.add_parser("recheck"); p.add_argument("--pack", required=True); p.add_argument("run", nargs="+")
    p.add_argument("--num-predict", type=int, help="the token limit the run used (runs before 2026-10-03: 256)")
    p.set_defaults(func=cmd_recheck)
    p = sub.add_parser("calibrate"); p.add_argument("--pack", required=True); p.add_argument("--run", required=True, nargs="+")
    p.add_argument("--write", action="store_true"); p.set_defaults(func=cmd_calibrate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
