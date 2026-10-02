import pytest

from rag_pipeline.config import load_settings
from rag_pipeline.embeddings import LocalHashingEmbedder
from rag_pipeline.exceptions import ConfigError, EmbeddingMismatchError, IndexNotFoundError
from rag_pipeline.generation import REFUSAL_TEXT, build_generator
from rag_pipeline.retrieval import Retriever
from rag_pipeline.service import RagService
from rag_pipeline.vector_store import VectorStore


@pytest.mark.parametrize(("question", "doc_id"), [
    ("Is flood or storm surge damage covered?", "ho-perils-exclusions"),
    ("What roof age makes a home ineligible for new business?", "uw-homeowners-guideline"),
    ("How many red flags require an SIU referral?", "fraud-red-flags"),
    ("Within how many hours must the adjuster contact the insured?", "claims-fnol-sop"),
])
def test_top_hit_is_expected_document(indexed_settings, question, doc_id):
    result = RagService(indexed_settings).query(question)
    assert result.hits[0].chunk.doc_id == doc_id
    assert not result.refused
    assert result.citations and all("#" in c for c in result.citations)
    assert f"[{result.citations[0]}]" in result.answer


def test_doc_type_filter_restricts_results(indexed_settings):
    result = RagService(indexed_settings).query("mold remediation limit", doc_types=["claims_sop"])
    assert result.hits and {h.chunk.doc_type for h in result.hits} == {"claims_sop"}


def test_out_of_scope_question_is_refused(indexed_settings):
    result = RagService(indexed_settings).query("What is the capital of Australia?")
    assert result.refused
    assert result.answer == REFUSAL_TEXT
    assert result.citations == []


def test_refusal_threshold_is_configurable(indexed_settings):
    indexed_settings.refusal_threshold = 0.99
    result = RagService(indexed_settings).query("What is the Coverage B limit?")
    assert result.refused and result.max_similarity < 0.99
    indexed_settings.refusal_threshold = 0.0
    assert not RagService(indexed_settings).query("What is the Coverage B limit?").refused


def test_missing_index_and_mismatched_embedder_raise(settings, indexed_settings):
    with pytest.raises(IndexNotFoundError):
        VectorStore.load(settings.index_dir.parent / "nope")
    store = VectorStore.load(indexed_settings.index_dir)
    with pytest.raises(EmbeddingMismatchError):
        Retriever(store, LocalHashingEmbedder(1024))


def test_cloud_providers_fail_fast_without_configuration(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        build_generator(load_settings(generator_provider="azure_openai"))
    with pytest.raises(ConfigError):
        build_generator(load_settings(generator_provider="bedrock"))  # no model id
    with pytest.raises(ConfigError):
        load_settings(azure_openai={"api_key": "should-not-be-in-yaml"})
