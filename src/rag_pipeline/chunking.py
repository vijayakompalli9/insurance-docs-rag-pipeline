"""Heading-aware chunking with word-level overlap and stable, content-hashed chunk ids."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from .documents import Document

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_type: str
    title: str
    section: str
    text: str
    content_hash: str

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedder: section context improves retrieval of short chunks."""
        return f"{self.section}\n{self.text}"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def slugify(value: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "section"


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Return ``(section_path, body)`` pairs; path joins the heading hierarchy with ' > '."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            path = " > ".join(title for _, title in stack) or "Preamble"
            sections.append((path, text))
        body.clear()

    for line in markdown.split("\n"):
        match = _HEADING.match(line)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            body.append(line)
    flush()
    return sections


def window_words(words: list[str], max_words: int, overlap: int) -> list[list[str]]:
    """Split a word list into windows of ``max_words`` that share ``overlap`` words."""
    if max_words <= 0 or not 0 <= overlap < max_words:
        raise ValueError("require max_words > 0 and 0 <= overlap < max_words")
    if len(words) <= max_words:
        return [words]
    step = max_words - overlap
    windows = []
    for start in range(0, len(words), step):
        windows.append(words[start:start + max_words])
        if start + max_words >= len(words):
            break
    return windows


def content_hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def chunk_document(doc: Document, max_words: int = 120, overlap_words: int = 25) -> list[Chunk]:
    """Chunk a (redacted) document; chunks never cross a heading boundary."""
    chunks: list[Chunk] = []
    used_ids: set[str] = set()
    for section, body in split_sections(doc.text):
        leaf = section.split(" > ")[-1]
        base = f"{doc.doc_id}#{slugify(leaf)}"
        for i, words in enumerate(window_words(body.split(), max_words, overlap_words)):
            chunk_id = f"{base}-{i}"
            suffix = 2
            while chunk_id in used_ids:  # duplicate headings within one doc
                chunk_id = f"{base}-{i}-{suffix}"
                suffix += 1
            used_ids.add(chunk_id)
            text = " ".join(words)
            chunks.append(Chunk(
                chunk_id=chunk_id, doc_id=doc.doc_id, doc_type=doc.doc_type, title=doc.title,
                section=section, text=text,
                content_hash=content_hash(doc.title, section, text),
            ))
    return chunks
