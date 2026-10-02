"""FastAPI app exposing /health and /query."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings, load_settings
from .exceptions import IndexNotFoundError, ProviderError
from .logging_utils import configure_logging
from .service import RagService

log = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    doc_types: list[str] | None = None


class Source(BaseModel):
    chunk_id: str
    doc_id: str
    doc_type: str
    section: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    max_similarity: float
    citations: list[str]
    sources: list[Source]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    state: dict[str, RagService | None] = {"service": None}

    def get_service() -> RagService:
        if state["service"] is None:
            try:
                state["service"] = RagService(settings)
            except IndexNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return state["service"]  # type: ignore[return-value]

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        configure_logging(settings.log_level)
        yield

    app = FastAPI(title="Northwind Mutual Docs RAG (fictional)", version="0.1.0",
                  lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, object]:
        try:
            service = get_service()
        except HTTPException:
            return {"status": "degraded", "index_loaded": False, "chunks": 0}
        return {"status": "ok", "index_loaded": True, "chunks": len(service.store),
                "embedding": service.embedder.signature, "generator": service.generator.name}

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> dict[str, object]:
        try:
            return get_service().query(req.question, req.top_k, req.doc_types).as_dict()
        except ProviderError as exc:
            log.error("event=provider_error detail=%s", exc)
            raise HTTPException(status_code=502, detail="upstream model provider failed") from exc

    return app

