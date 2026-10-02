"""Structured key=value logging on top of the stdlib ``logging`` module."""

from __future__ import annotations

import logging
from typing import Any

_FORMAT = "ts=%(asctime)s level=%(levelname)s logger=%(name)s %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once; safe to call repeatedly."""
    root = logging.getLogger()
    if not any(getattr(h, "_rag_handler", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler._rag_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(level.upper())


def kv(event: str, **fields: Any) -> str:
    """Render an event plus fields as ``event=... key=value`` for grep-friendly logs."""
    parts = [f"event={event}"]
    for key, value in fields.items():
        text = str(value)
        if " " in text or not text:
            text = '"' + text.replace('"', "'") + '"'
        parts.append(f"{key}={text}")
    return " ".join(parts)
