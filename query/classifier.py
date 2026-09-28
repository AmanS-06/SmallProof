"""Zero-shot classifier backend: ModernBERT trained on NLI, run on CPU.

An NLI model scores "does the text entail this hypothesis?". Each label is
turned into a hypothesis with a template, for example "This text is about
{revenue}.", so the cost grows with the number of labels.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.interfaces import Classifier
from core.registry import register
from core.types import LabelScore


@register("classifier", "modernbert_nli")
class ZeroShotNLIClassifier(Classifier):
    def __init__(self, model_path: str, template: str = "This text is about {}.", batch_size: int = 8) -> None:
        from transformers import pipeline

        # reference_compile=False: skip torch.compile, which targets GPUs.
        self.pipe = pipeline("zero-shot-classification", model=model_path, device=-1,
                             model_kwargs={"reference_compile": False})
        self.template = template
        self.batch_size = batch_size

    def classify(self, text: str, labels: Sequence[str], template: str | None = None,
                 multi_label: bool = False) -> list[LabelScore]:
        """Scores sum to 1 across labels, unless multi_label is True (then each label is scored alone)."""
        output = self.pipe(text, list(labels), hypothesis_template=template or self.template, multi_label=multi_label)
        return [LabelScore(label, float(score)) for label, score in zip(output["labels"], output["scores"])]

    def entailment(self, premise: str, hypothesis: str) -> float:
        """Probability that premise entails hypothesis (entailment vs contradiction)."""
        output = self.pipe(premise, [hypothesis], hypothesis_template="{}", multi_label=True)
        return float(output["scores"][0])
