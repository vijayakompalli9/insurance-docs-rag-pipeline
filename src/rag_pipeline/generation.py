"""Answer generators: offline extractive default plus lazily-built Bedrock / Azure adapters."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from importlib import resources
from typing import Protocol

from .config import Settings
from .exceptions import ConfigError, ProviderError
from .retrieval import terms
from .vector_store import SearchHit

log = logging.getLogger(__name__)

REFUSAL_TEXT = "I don't know based on the indexed documents."
CITATION_RE = re.compile(r"\[([a-z0-9-]+#[a-z0-9-]+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"(])")


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    citations: list[str]


class AnswerGenerator(Protocol):
    name: str

    def generate(self, question: str, hits: list[SearchHit]) -> GeneratedAnswer: ...


def load_prompt() -> tuple[str, str]:
    """Return (system, user) templates from the packaged prompt file."""
    raw = resources.files("rag_pipeline").joinpath("prompts/answer_prompt.md").read_text()
    _, _, rest = raw.partition("<!-- system -->")
    system, _, user = rest.partition("<!-- user -->")
    return system.strip(), user.strip()


def render_context(hits: list[SearchHit]) -> str:
    return "\n\n".join(f"[{h.chunk.chunk_id}] ({h.chunk.title} / {h.chunk.section})\n{h.chunk.text}"
                       for h in hits)


def extract_citations(text: str, allowed: set[str]) -> list[str]:
    """Citations in model output, restricted to retrieved chunk ids (drops hallucinated ids)."""
    return list(dict.fromkeys(c for c in CITATION_RE.findall(text) if c in allowed))


class ExtractiveGenerator:
    """Offline generator: selects the sentences that best overlap the question, with citations."""

    name = "extractive"

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, hits: list[SearchHit]) -> GeneratedAnswer:
        q_terms = terms(question)
        # (score, overlap, rank, position, sentence, chunk_id)
        scored: list[tuple[float, int, int, int, str, str]] = []
        for rank, hit in enumerate(hits):
            for pos, sentence in enumerate(_SENTENCE_SPLIT.split(hit.chunk.text)):
                overlap = len(q_terms & terms(sentence))
                if overlap and "[REDACTED_" not in sentence:
                    scored.append((overlap + hit.score, overlap, rank, pos, sentence.strip(),
                                   hit.chunk.chunk_id))
        if not scored:
            return GeneratedAnswer(REFUSAL_TEXT, [])
        scored.sort(key=lambda s: s[0], reverse=True)
        # drop weak tail sentences: keep those with at least half the best term overlap
        floor = max(1, (scored[0][1] + 1) // 2)
        picked = [s for s in scored if s[1] >= floor][: self.max_sentences]
        picked.sort(key=lambda s: (s[2], s[3]))  # present in retrieval rank, then reading order
        text = " ".join(f"{s[4]} [{s[5]}]" for s in picked)
        return GeneratedAnswer(text, list(dict.fromkeys(s[5] for s in picked)))


class BedrockClaudeGenerator:
    """Anthropic Claude on AWS Bedrock via the Converse API."""

    name = "bedrock"

    def __init__(self, region: str, model_id: str) -> None:
        if not model_id:
            raise ConfigError("bedrock.generation_model_id must be set for the bedrock generator")
        try:
            import boto3  # lazy import
        except ImportError as exc:
            raise ConfigError("boto3 is required for the bedrock provider") from exc
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._model_id = model_id
        self._system, self._user = load_prompt()

    def generate(self, question: str, hits: list[SearchHit]) -> GeneratedAnswer:
        prompt = self._user.format(context=render_context(hits), question=question)
        try:
            resp = self._client.converse(
                modelId=self._model_id, system=[{"text": self._system}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 512, "temperature": 0.0},
            )
            text = resp["output"]["message"]["content"][0]["text"].strip()
        except Exception as exc:
            raise ProviderError(f"Bedrock Converse call failed: {exc}") from exc
        return GeneratedAnswer(text, extract_citations(text, {h.chunk.chunk_id for h in hits}))


class AzureOpenAIGenerator:
    """Chat completions against an Azure OpenAI deployment."""

    name = "azure_openai"

    def __init__(self, endpoint: str, api_version: str, deployment: str) -> None:
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not (endpoint and deployment and api_key):
            raise ConfigError("azure_openai generator needs endpoint, chat_deployment and "
                              "AZURE_OPENAI_API_KEY in the environment")
        try:
            from openai import AzureOpenAI  # lazy import
        except ImportError as exc:
            raise ConfigError("openai is required for azure_openai") from exc
        self._client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
        self._deployment = deployment
        self._system, self._user = load_prompt()

    def generate(self, question: str, hits: list[SearchHit]) -> GeneratedAnswer:
        prompt = self._user.format(context=render_context(hits), question=question)
        try:
            resp = self._client.chat.completions.create(
                model=self._deployment, temperature=0.0, max_tokens=512,
                messages=[{"role": "system", "content": self._system},
                          {"role": "user", "content": prompt}],
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            raise ProviderError(f"Azure OpenAI chat call failed: {exc}") from exc
        return GeneratedAnswer(text, extract_citations(text, {h.chunk.chunk_id for h in hits}))


def build_generator(settings: Settings) -> AnswerGenerator:
    name = settings.generator_provider
    if name == "extractive":
        return ExtractiveGenerator(settings.max_answer_sentences)
    if name == "bedrock":
        return BedrockClaudeGenerator(settings.bedrock.region, settings.bedrock.generation_model_id)
    if name == "azure_openai":
        a = settings.azure_openai
        endpoint = a.endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        return AzureOpenAIGenerator(endpoint, a.api_version, a.chat_deployment)
    raise ConfigError(f"Unknown generator_provider '{name}'")
