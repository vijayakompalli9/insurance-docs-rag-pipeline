"""Embedding provider interface with an offline default and lazily-imported cloud adapters."""

from __future__ import annotations

import json
import logging
import os
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from .config import Settings
from .exceptions import ConfigError, ProviderError

log = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Anything that maps texts to L2-normalised float32 vectors."""

    @property
    def signature(self) -> str:
        """Identifies model + params; the index is rebuilt when this changes."""
        ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class LocalHashingEmbedder:
    """Offline, stateless embedder: hashed unigram+bigram TF with sublinear scaling.

    Stateless matters for incremental indexing: unlike a fitted TF-IDF model, the vector
    for an unchanged chunk never changes when other documents are added, so we can safely
    skip re-embedding chunks whose content hash is unchanged.
    """

    def __init__(self, n_features: int = 8192) -> None:
        self._n_features = n_features
        self._vectorizer = HashingVectorizer(
            n_features=n_features, ngram_range=(1, 2), stop_words="english",
            alternate_sign=False, norm=None, lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b",
        )

    @property
    def signature(self) -> str:
        return f"local-hashing-v1:{self._n_features}"

    def embed(self, texts: list[str]) -> np.ndarray:
        counts = self._vectorizer.transform(texts)
        counts.data = 1.0 + np.log(counts.data)  # sublinear tf dampens repeated terms
        return l2_normalize(counts.toarray())


class BedrockTitanEmbedder:
    """Amazon Titan Text Embeddings via the Bedrock runtime ``invoke_model`` API."""

    def __init__(self, region: str, model_id: str, dimensions: int) -> None:
        try:
            import boto3  # lazy: only needed when this provider is configured
        except ImportError as exc:
            raise ConfigError("boto3 is required for the bedrock provider: pip install boto3") from exc
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._model_id, self._dimensions = model_id, dimensions

    @property
    def signature(self) -> str:
        return f"bedrock:{self._model_id}:{self._dimensions}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:  # Titan v2 embeds one input per request
            body = json.dumps({"inputText": text, "dimensions": self._dimensions, "normalize": True})
            try:
                resp = self._client.invoke_model(modelId=self._model_id, body=body)
                vectors.append(json.loads(resp["body"].read())["embedding"])
            except Exception as exc:  # boto3 raises many botocore error types
                raise ProviderError(f"Bedrock embedding failed: {exc}") from exc
        return l2_normalize(np.array(vectors))


class AzureOpenAIEmbedder:
    """Azure OpenAI embeddings deployment via the ``openai`` SDK."""

    def __init__(self, endpoint: str, api_version: str, deployment: str) -> None:
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not (endpoint and deployment and api_key):
            raise ConfigError("azure_openai embeddings need endpoint, embedding_deployment "
                              "and AZURE_OPENAI_API_KEY in the environment")
        try:
            from openai import AzureOpenAI  # lazy import
        except ImportError as exc:
            raise ConfigError("openai is required for azure_openai: pip install openai") from exc
        self._client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
        self._deployment = deployment

    @property
    def signature(self) -> str:
        return f"azure_openai:{self._deployment}"

    def embed(self, texts: list[str]) -> np.ndarray:
        try:
            resp = self._client.embeddings.create(model=self._deployment, input=texts)
        except Exception as exc:
            raise ProviderError(f"Azure OpenAI embedding failed: {exc}") from exc
        return l2_normalize(np.array([item.embedding for item in resp.data]))


def build_embedder(settings: Settings) -> EmbeddingProvider:
    """Construct the configured provider; cloud SDKs are imported only on this path."""
    name = settings.embedding_provider
    if name == "local":
        return LocalHashingEmbedder(settings.local_embedding_dim)
    if name == "bedrock":
        b = settings.bedrock
        return BedrockTitanEmbedder(b.region, b.embedding_model_id, b.embedding_dimensions)
    if name == "azure_openai":
        a = settings.azure_openai
        endpoint = a.endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        return AzureOpenAIEmbedder(endpoint, a.api_version, a.embedding_deployment)
    raise ConfigError(f"Unknown embedding_provider '{name}'")
