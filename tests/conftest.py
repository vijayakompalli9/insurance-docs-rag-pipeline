from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rag_pipeline.config import load_settings
from rag_pipeline.ingest import run_ingest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("RAG_CONFIG", "RAG_INDEX_DIR", "RAG_CORPUS_DIR", "RAG_EMBEDDING_PROVIDER",
                "RAG_GENERATOR_PROVIDER", "RAG_REFUSAL_THRESHOLD", "RAG_TOP_K"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def corpus_copy(tmp_path: Path) -> Path:
    dest = tmp_path / "corpus"
    shutil.copytree(ROOT / "corpus", dest)
    return dest


@pytest.fixture
def settings(tmp_path: Path, corpus_copy: Path):
    return load_settings(corpus_dir=str(corpus_copy), index_dir=str(tmp_path / "index"))


@pytest.fixture
def indexed_settings(settings):
    run_ingest(settings)
    return settings
