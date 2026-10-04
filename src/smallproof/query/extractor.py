"""Entity and keyword extraction with GLiNER small (CPU).

GLiNER finds spans for any entity types given at call time, for example
"company" or "financial metric". The KeyBERT fallback from the brief is not
installed; add it only if GLiNER turns out to be unsuitable.
"""

from __future__ import annotations

from collections.abc import Sequence

from smallproof.core.interfaces import Extractor
from smallproof.core.registry import register
from smallproof.core.types import Entity


@register("extractor", "gliner")
class GlinerExtractor(Extractor):
    def __init__(self, model_path: str, cache_dir: str | None = None, threshold: float = 0.5) -> None:
        from gliner import GLiNER

        # cache_dir: GLiNER loads its tokenizer by backbone name from the project cache.
        self.model = GLiNER.from_pretrained(model_path, local_files_only=True, cache_dir=cache_dir)
        self.threshold = threshold

    def extract(self, text: str, entity_types: Sequence[str]) -> list[Entity]:
        found = self.model.predict_entities(text, list(entity_types), threshold=self.threshold)
        return [Entity(item["text"], item["label"], float(item["score"])) for item in found]
