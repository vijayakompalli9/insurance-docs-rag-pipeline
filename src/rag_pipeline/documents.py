"""Corpus loading and text normalization."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml

from .exceptions import DocumentLoadError
from .logging_utils import kv

log = logging.getLogger(__name__)

REQUIRED_FIELDS = ("doc_id", "title", "doc_type")
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SMART_CHARS = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                              "\u2013": "-", "\u2014": "-", "\u00a0": " "})


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    doc_type: str
    version: str
    text: str
    source_path: str


def normalize_text(text: str) -> str:
    """NFKC-normalize, unify quotes/dashes/line endings, trim trailing space, collapse blanks."""
    text = unicodedata.normalize("NFKC", text).translate(_SMART_CHARS)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def parse_document(raw: str, source_path: str) -> Document:
    """Split YAML front matter from the markdown body and validate required fields."""
    match = _FRONT_MATTER.match(raw)
    if not match:
        raise DocumentLoadError(f"{source_path}: missing YAML front matter")
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise DocumentLoadError(f"{source_path}: invalid front matter: {exc}") from exc
    missing = [f for f in REQUIRED_FIELDS if not meta.get(f)]
    if missing:
        raise DocumentLoadError(f"{source_path}: missing front matter fields {missing}")
    return Document(
        doc_id=str(meta["doc_id"]),
        title=str(meta["title"]),
        doc_type=str(meta["doc_type"]),
        version=str(meta.get("version", "")),
        text=normalize_text(raw[match.end():]),
        source_path=source_path,
    )


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Load every ``*.md`` file in ``corpus_dir`` (sorted for determinism)."""
    if not corpus_dir.is_dir():
        raise DocumentLoadError(f"Corpus directory not found: {corpus_dir}")
    docs: list[Document] = []
    seen: set[str] = set()
    for path in sorted(corpus_dir.glob("*.md")):
        try:
            doc = parse_document(path.read_text(encoding="utf-8"), str(path))
        except UnicodeDecodeError as exc:
            raise DocumentLoadError(f"{path}: not valid UTF-8") from exc
        if doc.doc_id in seen:
            raise DocumentLoadError(f"Duplicate doc_id '{doc.doc_id}' in {path}")
        seen.add(doc.doc_id)
        docs.append(doc)
    log.info(kv("corpus_loaded", documents=len(docs), corpus_dir=corpus_dir))
    return docs
