"""Query orchestration: retrieve, apply the refusal threshold, generate a cited answer."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .config import Settings
from .embeddings import EmbeddingProvider, build_embedder
from .generation import REFUSAL_TEXT, AnswerGenerator, build_generator
from .logging_utils import kv
from .retrieval import Retriever
from .vector_store import SearchHit, VectorStore

log = logging.getLogger(__name__)


@dataclass
class QueryResult:
    question: str
    answer: str
    refused: bool
    max_similarity: float
    citations: list[str] = field(default_factory=list)
    hits: list[SearchHit] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "question": self.question, "answer": self.answer, "refused": self.refused,
            "max_similarity": round(self.max_similarity, 4), "citations": self.citations,
            "sources": [{"chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id,
                         "doc_type": h.chunk.doc_type, "section": h.chunk.section,
                         "score": round(h.score, 4)} for h in self.hits],
        }


class RagService:
    def __init__(self, settings: Settings, embedder: EmbeddingProvider | None = None,
                 generator: AnswerGenerator | None = None) -> None:
        self.settings = settings
        self.embedder = embedder or build_embedder(settings)
        self.store = VectorStore.load(settings.index_dir)
        self.retriever = Retriever(self.store, self.embedder, rerank=settings.rerank)
        self.generator = generator or build_generator(settings)

    def query(self, question: str, top_k: int | None = None,
              doc_types: list[str] | None = None) -> QueryResult:
        started = time.perf_counter()
        k = top_k or self.settings.top_k
        result = self.retriever.retrieve(question, k, doc_types)
        if not result.hits or result.max_similarity < self.settings.refusal_threshold:
            out = QueryResult(question, REFUSAL_TEXT, True, result.max_similarity, [], result.hits)
        else:
            gen = self.generator.generate(question, result.hits)
            refused = gen.text.strip() == REFUSAL_TEXT
            out = QueryResult(question, gen.text, refused, result.max_similarity,
                              gen.citations, result.hits)
        log.info(kv("query", refused=out.refused, max_sim=f"{out.max_similarity:.3f}",
                    top_doc=out.hits[0].chunk.doc_id if out.hits else "none",
                    latency_ms=f"{(time.perf_counter() - started) * 1000:.1f}"))
        return out
