from itertools import pairwise

import pytest

from rag_pipeline.chunking import chunk_document, split_sections, window_words
from rag_pipeline.documents import Document, normalize_text, parse_document
from rag_pipeline.exceptions import DocumentLoadError


def _doc(text: str) -> Document:
    return Document("d1", "Doc One", "policy_wording", "1", text, "mem")


def test_split_sections_tracks_heading_hierarchy():
    md = "# Top\nintro\n## A\nalpha\n### A1\ndeep\n## B\nbeta\n"
    paths = [p for p, _ in split_sections(md)]
    assert paths == ["Top", "Top > A", "Top > A > A1", "Top > B"]


def test_chunks_never_cross_heading_boundaries():
    md = "# T\n## First\n" + "apple " * 50 + "\n## Second\n" + "banana " * 50
    chunks = chunk_document(_doc(md), max_words=30, overlap_words=5)
    for c in chunks:
        assert not ("apple" in c.text and "banana" in c.text)
    assert {c.section for c in chunks} == {"T > First", "T > Second"}


def test_window_overlap_and_full_coverage():
    words = [f"w{i}" for i in range(100)]
    windows = window_words(words, max_words=30, overlap=10)
    assert all(len(w) <= 30 for w in windows)
    for prev, nxt in pairwise(windows):
        assert prev[-10:] == nxt[:10]
    assert windows[-1][-1] == "w99"
    assert {w for win in windows for w in win} == set(words)


def test_short_section_is_single_chunk_and_invalid_overlap_rejected():
    assert window_words(["a", "b"], 10, 3) == [["a", "b"]]
    with pytest.raises(ValueError):
        window_words(["a"], 5, 5)


def test_chunk_ids_stable_and_hash_changes_with_text():
    md = "# T\n## Section One\nhello world\n"
    a = chunk_document(_doc(md))[0]
    b = chunk_document(_doc(md.replace("world", "there")))[0]
    assert a.chunk_id == b.chunk_id == "d1#section-one-0"
    assert a.content_hash != b.content_hash


def test_normalize_and_front_matter_validation():
    assert normalize_text("a\u2019b  c\r\n\n\n\nd  ") == "a'b c\n\nd\n"
    with pytest.raises(DocumentLoadError):
        parse_document("# no front matter", "x.md")
    with pytest.raises(DocumentLoadError):
        parse_document("---\ntitle: x\n---\nbody", "x.md")
