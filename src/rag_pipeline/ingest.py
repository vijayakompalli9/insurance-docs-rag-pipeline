"""Ingestion pipeline: load -> normalize -> redact -> chunk -> incremental embed -> persist."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from .chunking import Chunk, chunk_document
from .config import Settings
from .documents import load_corpus
from .embeddings import EmbeddingProvider, build_embedder
from .logging_utils import kv
from .redaction import redact
from .vector_store import VectorStore

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    documents: int = 0
    chunks_total: int = 0
    embedded: int = 0
    reused: int = 0
    removed: int = 0
    full_rebuild: bool = False
    pii_counts: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, object]:
        return {"documents": self.documents, "chunks_total": self.chunks_total,
                "embedded": self.embedded, "reused": self.reused, "removed": self.removed,
                "full_rebuild": self.full_rebuild, "pii_redacted": dict(self.pii_counts)}


def build_chunks(settings: Settings, report: IngestReport) -> list[Chunk]:
    """Load the corpus, redact PII per document (logging counts), and chunk."""
    chunks: list[Chunk] = []
    docs = load_corpus(settings.corpus_dir)
    report.documents = len(docs)
    for doc in docs:
        result = redact(doc.text)
        if result.total:
            log.info(kv("pii_redacted", doc_id=doc.doc_id, total=result.total,
                        **{k.lower(): v for k, v in sorted(result.counts.items())}))
        report.pii_counts.update(result.counts)
        chunks.extend(chunk_document(replace(doc, text=result.text),
                                     settings.chunking.max_words, settings.chunking.overlap_words))
    return chunks


def run_ingest(settings: Settings, embedder: EmbeddingProvider | None = None) -> IngestReport:
    """Build or incrementally update the index; unchanged chunks (same id + hash) are reused."""
    embedder = embedder or build_embedder(settings)
    report = IngestReport()
    chunks = build_chunks(settings, report)

    existing = VectorStore.load_or_empty(settings.index_dir, embedder.signature)
    if existing.signature != embedder.signature:
        log.warning(kv("embedder_changed", old=existing.signature, new=embedder.signature))
        existing = VectorStore(embedder.signature)
        report.full_rebuild = True
    previous = existing.vector_by_id()

    vectors: list[np.ndarray | None] = []
    to_embed: list[int] = []
    for i, chunk in enumerate(chunks):
        prior = previous.get(chunk.chunk_id)
        if prior is not None and prior[0] == chunk.content_hash:
            vectors.append(prior[1])
        else:
            vectors.append(None)
            to_embed.append(i)
    if to_embed:
        fresh = embedder.embed([chunks[i].embedding_text for i in to_embed])
        for row, i in zip(fresh, to_embed, strict=True):
            vectors[i] = row

    current_ids = {c.chunk_id for c in chunks}
    report.chunks_total = len(chunks)
    report.embedded = len(to_embed)
    report.reused = len(chunks) - len(to_embed)
    report.removed = sum(1 for cid in previous if cid not in current_ids)

    matrix = np.vstack(vectors).astype(np.float32) if chunks else np.zeros((0, 0), np.float32)
    VectorStore(embedder.signature, chunks, matrix).save(settings.index_dir)
    log.info(kv("ingest_complete", **{k: v for k, v in report.as_dict().items()
                                      if k != "pii_redacted"}))
    return report
