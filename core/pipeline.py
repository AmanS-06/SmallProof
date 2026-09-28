"""Pipeline orchestrator: ingest (once per document set) and ask (once per question).

Query flow for the full system (variant "C_full"):
1. CPU analysis: GLiNER entities, metadata tags (company, year, section) and
   the intent label (routing), optionally in parallel threads.
2. Hybrid retrieval: BM25 + dense, plus an extra company-filtered search when
   the company tag is confident, merged with reciprocal rank fusion.
3. Soft boosts from the tags, then the cross-encoder reranker.
4. Evidence gate. Not sufficient: widen once (more candidates, no boosts).
   Still not sufficient: refuse clearly.
5. The SLM answers from the top chunks with page citations.

Each component can be switched off through a variant, which is how the
baselines and ablations are run. Models load lazily on first use.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.config import apply_runtime_env, get, load_config, model_path, resolve_path
from core.profiling import Profiler
from core.types import Answer, Chunk, Verdict
from query.hybrid import fuse_chunks

FULL = {"bm25": True, "dense": True, "extractor": True, "tags": True, "router": True, "reranker": True, "gate": True}
VARIANTS: dict[str, dict[str, bool]] = {
    "C_full": FULL,
    "B_dense_rag": {key: False for key in FULL} | {"dense": True},
    **{f"C_no_{name}": FULL | {name: False} for name in FULL},
    # Phase 5 finding: on FinanceBench the router and GLiNER cost latency without retrieval gains.
    "C_lean": FULL | {"router": False, "extractor": False},
}


@dataclass
class Retrieval:
    chunks: list[Chunk]  # final chunks for the SLM, best first
    candidates: list[Chunk]  # fused candidates before reranking
    verdict: Verdict | None = None
    confidence: float | None = None
    widened: bool = False
    tags: dict[str, Any] = field(default_factory=dict)


def _chunk_from_dict(data: dict) -> Chunk:
    return Chunk(**data)


class Pipeline:
    def __init__(self, config: dict) -> None:
        self.config = config
        apply_runtime_env(config)
        self.index_dir = resolve_path(config, config["paths"]["index_dir"]) / config.get("pack_name", "default")
        self._cache: dict[str, Any] = {}

    @classmethod
    def from_pack(cls, pack: str, overrides: dict | None = None) -> Pipeline:
        return cls(load_config(pack, overrides))

    def _lazy(self, name: str, factory: Callable[[], Any]) -> Any:
        if name not in self._cache:
            self._cache[name] = factory()
        return self._cache[name]

    # Section: components (loaded on first use)

    @property
    def embedder(self):
        from query.dense_index import Embedder

        return self._lazy("embedder", lambda: Embedder(model_path(self.config, "embedder"),
                                                        get(self.config, "embedder.query_instruction", ""),
                                                        get(self.config, "embedder.batch_size", 32)))

    @property
    def classifier(self):
        from query.classifier import ZeroShotNLIClassifier

        return self._lazy("classifier", lambda: ZeroShotNLIClassifier(model_path(self.config, "classifier")))

    @property
    def extractor(self):
        from query.extractor import GlinerExtractor

        hub = resolve_path(self.config, self.config["paths"]["models_dir"]) / "hub"
        return self._lazy("extractor", lambda: GlinerExtractor(model_path(self.config, "extractor"), cache_dir=str(hub)))

    @property
    def reranker(self):
        from query.reranker import CrossEncoderReranker

        return self._lazy("reranker", lambda: CrossEncoderReranker(model_path(self.config, "reranker")))

    @property
    def gate(self):
        from query.gate import EvidenceGate

        settings = self.config["gate"]
        mode = settings.get("mode", "rerank")
        return self._lazy("gate", lambda: EvidenceGate(settings["sufficient_threshold"], settings["partial_threshold"],
                                                       mode, self.classifier if mode != "rerank" else None))

    @property
    def generator(self):
        from generate.ollama_backend import OllamaGenerator

        settings = self.config["generator"]
        keys = ("model", "host", "num_ctx", "temperature", "num_predict")
        keep_alive = get(self.config, "runtime.ollama.keep_alive", "1m")
        return self._lazy("generator", lambda: OllamaGenerator(keep_alive=keep_alive,
                                                               **{key: settings[key] for key in keys if key in settings}))

    @property
    def template(self) -> str:
        return self._lazy("template", lambda: resolve_path(self.config, self.config["generator"]["template"]).read_text(encoding="utf-8"))

    @property
    def chunks(self) -> list[Chunk]:
        def load():
            path = self.index_dir / "chunks.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"No index at {self.index_dir}. Run ingest first.")
            return [_chunk_from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]

        return self._lazy("chunks", load)

    @property
    def chunk_store(self) -> dict[str, Chunk]:
        return self._lazy("chunk_store", lambda: {chunk.id: chunk for chunk in self.chunks})

    @property
    def vocab(self) -> dict:
        return self._lazy("vocab", lambda: json.loads((self.index_dir / "vocab.json").read_text(encoding="utf-8")))

    @property
    def bm25(self):
        from query.bm25_index import BM25Retriever

        return self._lazy("bm25", lambda: BM25Retriever(self.chunks))

    @property
    def dense(self):
        from query.dense_index import DenseIndex, DenseRetriever

        return self._lazy("dense", lambda: DenseRetriever(self.embedder, DenseIndex(self.index_dir / "chroma"), self.chunk_store))

    # Section: ingest

    def ingest(self, pdf_paths: Sequence[str | Path], doc_meta: dict[str, dict] | None = None,
               log: Callable[[str], None] = print, guard=None) -> dict:
        """Build the pack's index: parse, chunk, tag and embed each document, then add metadata and store.

        Each finished document is cached in <index_dir>_work, so a job stopped by the
        thermal guard's time limit resumes where it left off. Returns status "paused"
        or "complete".
        """
        import numpy as np

        from ingest.chunk_tagger import tag_with_classifier, tag_with_rules
        from ingest.chunker import chunk_document
        from ingest.parsers.pdf_text import extract_pages
        from ingest.vocab_builder import add_metadata, build_vocab
        from query.dense_index import DenseIndex

        cfg, doc_meta = self.config, doc_meta or {}
        work_dir = self.index_dir.parent / f"{self.index_dir.name}_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        profiler, count, batch = Profiler(), self.embedder.count_tokens, 64
        for number, path in enumerate(pdf_paths, start=1):
            doc_id = Path(path).stem
            if (work_dir / f"{doc_id}.npy").exists():
                continue
            if guard and not guard.checkpoint():
                return {"status": "paused", "documents_done": number - 1, "documents": len(pdf_paths)}
            with profiler.stage("parse"):
                pages = extract_pages(path, get(cfg, "parser.text_backend", "pypdfium2"))
            with profiler.stage("chunk"):
                chunks = chunk_document(doc_id, pages, count, **cfg["chunker"])
            if get(cfg, "parser.tables", False):
                with profiler.stage("tables"):
                    chunks += self._table_chunks(path, doc_id, count)
            with profiler.stage("tag"):
                if get(cfg, "tagger.backend", "rules") == "zeroshot":
                    chunks = tag_with_classifier(chunks, self.classifier, cfg["tagger"]["labels"])
                else:
                    chunks = tag_with_rules(chunks)
            vectors = []
            for start in range(0, len(chunks), batch):
                if guard and not guard.checkpoint():
                    return {"status": "paused", "documents_done": number - 1, "documents": len(pdf_paths)}
                with profiler.stage("embed"):
                    vectors += self.embedder.encode_passages([c.text for c in chunks[start : start + batch]])
            (work_dir / f"{doc_id}.jsonl").write_text("".join(json.dumps(asdict(c)) + "\n" for c in chunks), encoding="utf-8")
            np.save(work_dir / f"{doc_id}.npy", np.asarray(vectors, dtype="float32").reshape(len(chunks), -1))
            log(f"[{number}/{len(pdf_paths)}] {doc_id}: {len(pages)} pages, {len(chunks)} chunks")

        all_chunks, embeddings, pages_total = [], [], 0
        for path in pdf_paths:
            doc_id = Path(path).stem
            doc_chunks = [_chunk_from_dict(json.loads(line)) for line in
                          (work_dir / f"{doc_id}.jsonl").read_text(encoding="utf-8").splitlines()]
            all_chunks += doc_chunks
            embeddings += np.load(work_dir / f"{doc_id}.npy").tolist()
            pages_total += max((c.page_end for c in doc_chunks), default=0)
        with profiler.stage("metadata"):
            all_chunks = add_metadata(all_chunks, doc_meta, get(cfg, "vocab.section_patterns", []))
            vocab = build_vocab(all_chunks, doc_meta)
        with profiler.stage("store"):
            shutil.rmtree(self.index_dir, ignore_errors=True)
            self.index_dir.mkdir(parents=True)
            with (self.index_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
                for chunk in all_chunks:
                    handle.write(json.dumps(asdict(chunk)) + "\n")
            (self.index_dir / "vocab.json").write_text(json.dumps(vocab, indent=1), encoding="utf-8")
            DenseIndex(self.index_dir / "chroma").add(all_chunks, embeddings)
        stats = {"status": "complete", "documents": len(pdf_paths), "pages_with_chunks_max": pages_total,
                 "chunks": len(all_chunks),
                 "timings_ms": {k: round(v) for k, v in profiler.timings_ms.items()},
                 "settings": {key: cfg[key] for key in ("parser", "chunker", "tagger")},
                 "embedder": cfg["models"]["embedder"], "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        (self.index_dir / "manifest.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        self._cache = {key: value for key, value in self._cache.items()
                       if key not in ("chunks", "chunk_store", "vocab", "bm25", "dense")}
        return stats

    def _table_chunks(self, path: str | Path, doc_id: str, count: Callable[[str], int]) -> list[Chunk]:
        from ingest.parsers.pdf_tables import extract_tables, table_to_texts

        chunks = []
        for page, tables in extract_tables(path).items():
            for t, table in enumerate(tables):
                for p, text in enumerate(table_to_texts(table, self.config["chunker"]["target_tokens"], count)):
                    chunks.append(Chunk(id=f"{doc_id}:p{page}:t{t}.{p}", doc_id=doc_id, text=text,
                                        page_start=page, page_end=page, chunk_type="table"))
        return chunks

    # Section: query

    def analyze(self, query: str, flags: dict[str, bool], profiler: Profiler) -> dict[str, Any]:
        """CPU analysis: entities, metadata tags and intent. Returns a QueryTags."""
        from query.metadata_tags import QueryTags, match_company, match_year

        cfg, tags = self.config, QueryTags()
        doc_fields = self.vocab.get("doc_fields", {})

        def run_extractor():
            with profiler.stage("extract"):
                return self.extractor.extract(query, get(cfg, "tags.entity_types", ["company"]))

        def run_section():
            labels = list(get(cfg, "tags.sections", {}))
            if not labels:
                return None
            with profiler.stage("tag_section"):
                best = self.classifier.classify(query, labels, template=get(cfg, "tags.section_template", "This question is about {}."))[0]
            return best.label, round(best.score, 3)

        def run_router():
            intents = get(cfg, "router.intents", {})
            if not intents:
                return None
            type_of = {description: name for name, description in intents.items()}
            with profiler.stage("route"):
                best = self.classifier.classify(query, list(intents.values()), template=get(cfg, "router.template", "{}"))[0]
            return type_of[best.label], round(best.score, 3)

        jobs = {}
        if flags["extractor"]:
            jobs["entities"] = run_extractor
        if flags["tags"]:
            jobs["section"] = run_section
        if flags["router"]:
            jobs["intent"] = run_router
        if get(cfg, "query.parallel", False) and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                futures = {name: pool.submit(job) for name, job in jobs.items()}
                results = {name: future.result() for name, future in futures.items()}
        else:
            results = {name: job() for name, job in jobs.items()}
        tags.entities = results.get("entities") or []
        tags.section, tags.intent = results.get("section"), results.get("intent")
        if flags["tags"]:
            tags.company = match_company(query, doc_fields.get("company", []), tags.entities)
            tags.year = match_year(query, doc_fields.get("year", []))
        return tags

    def retrieve(self, query: str, variant: str | dict = "C_full", profiler: Profiler | None = None,
                 tags=None, widen: bool = False) -> Retrieval:
        from query.metadata_tags import QueryTags, boost_factors

        flags = VARIANTS[variant] if isinstance(variant, str) else variant
        cfg, profiler = self.config, profiler or Profiler()
        settings = cfg["retrieval"]
        factor = get(cfg, "gate.widen_factor", 2) if widen else 1
        if tags is None:
            tags = self.analyze(query, flags, profiler)
        use_boosts = flags["tags"] or flags["router"]
        min_conf = settings["tag_min_confidence"]

        lists: list[list[Chunk]] = []
        with profiler.stage("retrieve"):
            company_filter = None  # extra search inside the tagged filing (company, and year when known)
            if flags["tags"] and not widen and tags.company and tags.company[1] >= min_conf:
                company_filter = {"company": tags.company[0]}
                if tags.year and tags.year[1] >= min_conf:
                    company_filter["year"] = tags.year[0]
            if flags["bm25"]:
                keywords = " ".join([query] + [e.span for e in tags.entities])  # entity spans weigh a bit more
                lists.append(self.bm25.retrieve(keywords, None, settings["bm25_k"] * factor))
                if company_filter:
                    lists.append(self.bm25.retrieve(keywords, company_filter, settings["bm25_k"]))
            if flags["dense"]:
                lists.append(self.dense.retrieve(query, None, settings["dense_k"] * factor))
                if company_filter:
                    lists.append(self.dense.retrieve(query, company_filter, settings["dense_k"]))
        top_n = settings["fused_k"] * factor
        with profiler.stage("fuse"):
            candidates = fuse_chunks(lists, k=settings["rrf_k"], top_n=top_n) if len(lists) > 1 else lists[0][:top_n]
            boosts = (boost_factors(candidates, tags, settings["tag_boost"], min_conf, get(cfg, "tags.sections", {}),
                                    get(cfg, "router.chunk_types", {}))
                      if use_boosts and not widen else {c.id: 1.0 for c in candidates})

        k = cfg["reranker"]["top_k"]
        rerank_prob: dict[str, float] = {}
        if flags["reranker"] or flags["gate"]:
            with profiler.stage("rerank"):
                probs = self.reranker.score(query, [c.text for c in candidates])
                rerank_prob = {c.id: p for c, p in zip(candidates, probs)}
        if flags["reranker"]:
            ordered = sorted(candidates, key=lambda c: -rerank_prob[c.id] * boosts[c.id])
        else:
            ordered = sorted(candidates, key=lambda c: -(c.score or 0) * boosts[c.id])
        final = ordered[:k]

        result = Retrieval(chunks=final, candidates=candidates, widened=widen,
                           tags=tags.as_dict() if isinstance(tags, QueryTags) else {})
        if rerank_prob:
            # The evidence score is recorded even with the gate off, so dev runs can calibrate it.
            with profiler.stage("gate"):
                score = self.gate.evidence_score(query, [c.with_score(rerank_prob[c.id]) for c in final]) if final else 0.0
            result.confidence = round(score, 4)
            if flags["gate"]:
                result.verdict = self.gate.verdict_for(score)
        return result

    def retrieve_with_fallback(self, query: str, variant: str | dict = "C_full",
                               profiler: Profiler | None = None) -> Retrieval:
        """Retrieve; if the gate is on and says not sufficient, widen once."""
        flags = VARIANTS[variant] if isinstance(variant, str) else variant
        profiler = profiler or Profiler()
        tags = self.analyze(query, flags, profiler)
        result = self.retrieve(query, flags, profiler, tags=tags)
        if flags["gate"] and result.verdict != Verdict.SUFFICIENT:
            result = self.retrieve(query, flags, profiler, tags=tags, widen=True)
        return result

    def ask(self, query: str, variant: str | dict = "C_full") -> Answer:
        profiler = Profiler()
        result = self.retrieve_with_fallback(query, variant, profiler)
        details = {"tags": result.tags, "widened": result.widened,
                   "chunks": [{"id": c.id, "doc_id": c.doc_id, "page": c.page_start} for c in result.chunks]}
        if result.verdict is not None and result.verdict != Verdict.SUFFICIENT:
            refusal = (Path(self.config["root"]) / "generate" / "prompts" / "refusal.txt").read_text(encoding="utf-8").strip()
            return Answer(text=refusal, refused=True, verdict=result.verdict, confidence=result.confidence,
                          chunk_ids=[c.id for c in result.chunks], timings_ms=profiler.timings_ms, details=details)
        answer = self.generator.generate(query, result.chunks, self.template)
        answer.verdict, answer.confidence = result.verdict, result.confidence
        answer.timings_ms = {**profiler.timings_ms, **answer.timings_ms}
        answer.details = {**details, **answer.details}
        return answer

    def ask_slm_only(self, query: str, doc_id: str, max_tokens: int = 3000) -> Answer:
        """Baseline A: the SLM alone, given the start of the gold document cut to fit its context."""
        doc_chunks = sorted((c for c in self.chunks if c.doc_id == doc_id), key=lambda c: (c.page_start, c.id))
        picked, used = [], 0
        for chunk in doc_chunks:
            size = self.embedder.count_tokens(chunk.text)
            if used + size > max_tokens:
                break
            picked.append(chunk)
            used += size
        answer = self.generator.generate(query, picked, self.template)
        answer.details["context_tokens"] = used
        return answer
