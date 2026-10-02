"""Regex-based PII redaction applied BEFORE chunking and embedding.

Patterns are deliberately conservative and US-centric; this is a guard rail, not a
replacement for a dedicated DLP service.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

_NAME = r"[A-Z][a-z]+(?:[ '-][A-Z][a-z]+)+"

# Order matters: SSN before PHONE so 9-digit dashed IDs are not mistaken for phone numbers.
PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?1[ .-]?)?(?:\(\d{3}\)\s?|\d{3}[ .-])\d{3}[ .-]\d{4}\b")),
    # Names are only detected in labelled or honorific contexts to avoid false positives.
    ("PERSON", re.compile(
        r"(?<=\b(?:Mr|Ms|Dr)\. )" + _NAME + r"|(?<=\bMrs\. )" + _NAME
    )),
    ("PERSON", re.compile(
        r"(?:(?<=Insured: )|(?<=Claimant: )|(?<=Claimant Name: )|(?<=Policyholder: )"
        r"|(?<=Contact person: ))" + _NAME
    )),
]


@dataclass
class RedactionResult:
    text: str
    counts: Counter[str] = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact(text: str) -> RedactionResult:
    """Replace PII matches with ``[REDACTED_<TYPE>]`` tokens and count them by type."""
    counts: Counter[str] = Counter()
    for label, pattern in PII_PATTERNS:
        text, n = pattern.subn(f"[REDACTED_{label}]", text)
        if n:
            counts[label] += n
    return RedactionResult(text=text, counts=counts)
