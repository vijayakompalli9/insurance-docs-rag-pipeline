"""Top-k retrieval with optional lexical re-ranking."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .embeddings import EmbeddingProvider
from .exceptions import EmbeddingMismatchError
from .vector_store import SearchHit, VectorStore

_TOKEN = re.compile(r"[a-z][a-z0-9]+")


def terms(text: str) -> set[str]:
    """Lower-cased content words with a crude plural strip, used for re-rank and extraction."""
    out = set()
    for tok in _TOKEN.findall(text.lower()):
        if tok in ENGLISH_STOP_WORDS:
            continue
        out.add(tok[:-1] if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") else tok)
    return out


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[SearchHit]
    max_similarity: float  # best raw cosine, used for the refusal decision


class Retriever:
    def __init__(self, store: VectorStore, embedder: EmbeddingProvider, rerank: bool = True,
                 rerank_weight: float = 0.25) -> None:
        if store.signature != embedder.signature:
            raise EmbeddingMismatchError(
                f"index built with '{store.signature}' but embedder is '{embedder.signature}'")
        self.store, self.embedder = store, embedder
        self.rerank, self.rerank_weight = rerank, rerank_weight

    def retrieve(self, question: str, k: int = 5,
                 doc_types: list[str] | None = None) -> RetrievalResult:
        query_vec = self.embedder.embed([question])[0]
        candidates = self.store.search(query_vec, k * 3 if self.rerank else k, doc_types)
        max_sim = candidates[0].score if candidates else 0.0
        if self.rerank and candidates:
            q_terms = terms(question)
            w = self.rerank_weight

            def blended(hit: SearchHit) -> float:
                overlap = len(q_terms & terms(hit.chunk.embedding_text)) / max(len(q_terms), 1)
                return (1 - w) * hit.score + w * overlap

            candidates = sorted((SearchHit(h.chunk, blended(h)) for h in candidates),
                                key=lambda h: h.score, reverse=True)
        return RetrievalResult(candidates[:k], max_sim)
