"""Numpy cosine-similarity index persisted to disk with doc-type metadata filtering."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .exceptions import EmbeddingMismatchError, IndexNotFoundError
from .logging_utils import kv

log = logging.getLogger(__name__)

VECTORS_FILE, CHUNKS_FILE, MANIFEST_FILE = "vectors.npy", "chunks.jsonl", "manifest.json"


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


class VectorStore:
    """In-memory matrix of normalised vectors; cosine similarity is a dot product."""

    def __init__(self, signature: str, chunks: list[Chunk] | None = None,
                 vectors: np.ndarray | None = None) -> None:
        self.signature = signature
        self.chunks: list[Chunk] = chunks or []
        self.vectors = vectors if vectors is not None else np.zeros((0, 0), dtype=np.float32)
        if len(self.chunks) != len(self.vectors):
            raise ValueError("chunks and vectors length mismatch")

    def __len__(self) -> int:
        return len(self.chunks)

    def vector_by_id(self) -> dict[str, tuple[str, np.ndarray]]:
        """Map chunk_id -> (content_hash, vector) for incremental reuse."""
        return {c.chunk_id: (c.content_hash, self.vectors[i]) for i, c in enumerate(self.chunks)}

    def search(self, query_vec: np.ndarray, k: int,
               doc_types: list[str] | None = None) -> list[SearchHit]:
        if not self.chunks:
            return []
        query_vec = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if query_vec.shape[0] != self.vectors.shape[1]:
            raise EmbeddingMismatchError(
                f"query dim {query_vec.shape[0]} != index dim {self.vectors.shape[1]}")
        scores = self.vectors @ query_vec
        if doc_types:
            allowed = set(doc_types)
            mask = np.array([c.doc_type in allowed for c in self.chunks])
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:k]
        return [SearchHit(self.chunks[i], float(scores[i])) for i in order if np.isfinite(scores[i])]

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / VECTORS_FILE, self.vectors.astype(np.float32))
        with (index_dir / CHUNKS_FILE).open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk.to_dict()) + "\n")
        manifest = {"signature": self.signature, "chunks": len(self.chunks),
                    "dim": int(self.vectors.shape[1]) if self.vectors.size else 0,
                    "doc_ids": sorted({c.doc_id for c in self.chunks}),
                    "updated_at": datetime.now(UTC).isoformat(timespec="seconds")}
        (index_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
        log.info(kv("index_saved", index_dir=index_dir, chunks=len(self.chunks)))

    @classmethod
    def load(cls, index_dir: Path) -> VectorStore:
        manifest_path = index_dir / MANIFEST_FILE
        if not manifest_path.is_file():
            raise IndexNotFoundError(f"No index at {index_dir}; run `ingest` first")
        manifest = json.loads(manifest_path.read_text())
        with (index_dir / CHUNKS_FILE).open(encoding="utf-8") as fh:
            chunks = [Chunk(**json.loads(line)) for line in fh if line.strip()]
        vectors = np.load(index_dir / VECTORS_FILE)
        return cls(manifest["signature"], chunks, vectors)

    @classmethod
    def load_or_empty(cls, index_dir: Path, signature: str) -> VectorStore:
        try:
            return cls.load(index_dir)
        except IndexNotFoundError:
            return cls(signature)
