"""Settings loaded from defaults, an optional YAML file, then environment variables.

Secrets are never read from YAML: cloud API keys come from the environment only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigError


@dataclass
class ChunkingConfig:
    max_words: int = 120
    overlap_words: int = 25


@dataclass
class BedrockConfig:
    region: str = "us-east-1"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dimensions: int = 1024
    generation_model_id: str = ""


@dataclass
class AzureOpenAIConfig:
    endpoint: str = ""
    api_version: str = "2024-06-01"
    embedding_deployment: str = ""
    chat_deployment: str = ""


@dataclass
class Settings:
    corpus_dir: Path = Path("corpus")
    index_dir: Path = Path("data/index")
    embedding_provider: str = "local"  # local | bedrock | azure_openai
    generator_provider: str = "extractive"  # extractive | bedrock | azure_openai
    local_embedding_dim: int = 8192
    top_k: int = 5
    rerank: bool = True
    refusal_threshold: float = 0.12
    max_answer_sentences: int = 3
    log_level: str = "INFO"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    bedrock: BedrockConfig = field(default_factory=BedrockConfig)
    azure_openai: AzureOpenAIConfig = field(default_factory=AzureOpenAIConfig)


_ENV_OVERRIDES = {
    "RAG_CORPUS_DIR": "corpus_dir",
    "RAG_INDEX_DIR": "index_dir",
    "RAG_EMBEDDING_PROVIDER": "embedding_provider",
    "RAG_GENERATOR_PROVIDER": "generator_provider",
    "RAG_REFUSAL_THRESHOLD": "refusal_threshold",
    "RAG_TOP_K": "top_k",
    "RAG_LOG_LEVEL": "log_level",
}
_SECRET_KEYS = {"api_key", "aws_secret_access_key", "secret", "password", "token"}


def _apply(target: Any, data: dict[str, Any], prefix: str = "") -> None:
    valid = {f.name: f for f in fields(target)}
    for key, value in data.items():
        if key in _SECRET_KEYS:
            raise ConfigError(f"'{prefix}{key}' must be supplied via environment, not config file")
        if key not in valid:
            raise ConfigError(f"Unknown config key '{prefix}{key}'")
        current = getattr(target, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ConfigError(f"'{prefix}{key}' must be a mapping")
            _apply(current, value, prefix=f"{prefix}{key}.")
        else:
            setattr(target, key, _coerce(current, value, f"{prefix}{key}"))


def _coerce(current: Any, value: Any, name: str) -> Any:
    try:
        if isinstance(current, bool):
            return value if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes"}
        if isinstance(current, Path):
            return Path(value)
        return type(current)(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid value for '{name}': {value!r}") from exc


def load_settings(path: str | Path | None = None, **overrides: Any) -> Settings:
    """Build settings: defaults < YAML file (``path`` or $RAG_CONFIG) < env < explicit overrides."""
    settings = Settings()
    path = path or os.environ.get("RAG_CONFIG")
    if path:
        cfg_path = Path(path)
        if not cfg_path.is_file():
            raise ConfigError(f"Config file not found: {cfg_path}")
        _apply(settings, yaml.safe_load(cfg_path.read_text()) or {})
    env = {attr: os.environ[var] for var, attr in _ENV_OVERRIDES.items() if var in os.environ}
    _apply(settings, env)
    _apply(settings, overrides)
    return settings
