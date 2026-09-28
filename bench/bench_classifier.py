"""Zero-shot classifier benchmark (ModernBERT NLI, CPU).

Measures:
- one question against 4, 8, 16 and 32 labels (an NLI model runs one pass per label)
- tagging 100 chunks with the 4 chunk types
- one long input of about 2,000 tokens with one hypothesis (the size the gate will see)
- a sanity check on 10 hand-written chunks with obvious types (not an accuracy number)

Run: .venv\\Scripts\\python.exe bench\\bench_classifier.py
"""

from __future__ import annotations

from itertools import cycle

from common import load_fixtures, load_sample_chunks, require_model, run_in_child, save_result, time_calls, time_once

LABEL_COUNTS = (4, 8, 16, 32)
TAG_CHUNKS = 100
LONG_INPUT_TOKENS = 2000
TOPIC_TEMPLATE = "This text is about {}."  # template used on the model card of this model family


def bench(model_path: str, fixtures: dict, chunks: list[str]) -> dict:
    import torch
    from transformers import pipeline

    # reference_compile=False: skip torch.compile, which targets GPUs.
    classifier, load_ms = time_once(lambda: pipeline(
        "zero-shot-classification", model=model_path, device=-1, model_kwargs={"reference_compile": False}
    ))
    question_cycle = cycle(fixtures["questions"])

    label_scaling = {}
    for count in LABEL_COUNTS:
        labels = fixtures["latency_labels"][:count]
        label_scaling[f"labels_{count}"] = time_calls(
            lambda labels=labels: classifier(next(question_cycle), labels, hypothesis_template=TOPIC_TEMPLATE)
        )

    # Chunk tagging with descriptive labels, mapped back to the four type names.
    type_config = fixtures["chunk_types"]
    descriptions = list(type_config["labels"].values())
    type_of = {description: name for name, description in type_config["labels"].items()}

    def tag(texts):
        return classifier(texts, descriptions, hypothesis_template=type_config["template"], batch_size=8)

    tag(chunks[:4])  # warmup
    _, tag_ms = time_once(lambda: tag(chunks[:TAG_CHUNKS]))

    # Long input: the first ~2,000 tokens of several chunks joined together.
    token_ids = classifier.tokenizer(" ".join(chunks[:10]))["input_ids"][:LONG_INPUT_TOKENS]
    long_text = classifier.tokenizer.decode(token_ids, skip_special_tokens=True)
    long_input = time_calls(
        lambda: classifier(long_text, ["the passages answer the question"], multi_label=True), runs=10
    )

    sanity = []
    for example, output in zip(fixtures["chunk_type_examples"], tag([ex["text"] for ex in fixtures["chunk_type_examples"]])):
        sanity.append({"expected": example["type"], "predicted": type_of[output["labels"][0]],
                       "score": round(output["scores"][0], 3)})

    return {
        "load_ms": load_ms,
        "torch_threads": torch.get_num_threads(),
        "label_scaling": label_scaling,
        "chunk_tagging": {"chunks": min(TAG_CHUNKS, len(chunks)), "labels": 4, "total_ms": tag_ms,
                          "chunks_per_second": round(min(TAG_CHUNKS, len(chunks)) / (tag_ms / 1000), 2)},
        "long_input": {"tokens": len(token_ids), **long_input},
        "sanity_check": {"agree": sum(row["expected"] == row["predicted"] for row in sanity),
                         "of": len(sanity), "rows": sanity},
    }


def main() -> None:
    model_path = require_model("classifier")
    chunks = [text for _, text in load_sample_chunks()]
    result = run_in_child(bench, model_path, load_fixtures(), chunks)
    save_result("classifier", result)
    scaling = ", ".join(f"{key.split('_')[1]} labels {value['median_ms']} ms" for key, value in result["label_scaling"].items())
    print(f"load {result['load_ms']} ms | {scaling}")
    print(f"tagging {result['chunk_tagging']['chunks_per_second']} chunks/s | "
          f"{result['long_input']['tokens']}-token input median {result['long_input']['median_ms']} ms | "
          f"sanity {result['sanity_check']['agree']}/{result['sanity_check']['of']} | peak RAM {result['peak_ram_mb']} MB")


if __name__ == "__main__":
    main()
