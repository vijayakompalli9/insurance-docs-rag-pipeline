"""Custom exception hierarchy so callers can handle pipeline failures precisely."""


class RagPipelineError(Exception):
    """Base class for all pipeline errors."""


class ConfigError(RagPipelineError):
    """Invalid or incomplete configuration (e.g. cloud provider selected without settings)."""


class DocumentLoadError(RagPipelineError):
    """A corpus document could not be read or is missing required front matter."""


class IndexNotFoundError(RagPipelineError):
    """The vector index has not been built yet."""


class EmbeddingMismatchError(RagPipelineError):
    """Query embeddings are incompatible with the persisted index."""


class ProviderError(RagPipelineError):
    """A remote embedding or generation provider failed."""
