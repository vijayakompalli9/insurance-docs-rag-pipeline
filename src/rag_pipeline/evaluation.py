"""Evaluation harness: retrieval (hit@k, MRR), citation quality, and refusal correctness."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .exceptions import ConfigError
from .generation import CITATION_RE
from .logging_utils import kv
from .service import QueryResult, RagService

log = logging.getLogger(__name__)
_ANSWER_SENTENCE = re.compile(r"(?<=[.!?\]])\s+(?=[A-Z])")


@dataclass(frozen=True)
class GoldenQuestion:
    qid: str
    question: str
    expected_doc: str | None  # None => out of scope, the system should refuse

    @property
    def out_of_scope(self) -> bool:
        return self.expected_doc is None


def load_golden(path: Path) -> list[GoldenQuestion]:
    if not path.is_file():
        raise ConfigError(f"Golden set not found: {path}")
    rows = (yaml.safe_load(path.read_text()) or {}).get("questions", [])
    try:
        return [GoldenQuestion(r["id"], r["question"], r.get("expected_doc")) for r in rows]
    except KeyError as exc:
        raise ConfigError(f"Golden question missing field {exc}") from exc


def citation_coverage(answer: str) -> float:
    """Share of answer sentences carrying at least one [doc#chunk] citation."""
    sentences = [s for s in _ANSWER_SENTENCE.split(answer.strip()) if s]
    if not sentences:
        return 0.0
    return sum(1 for s in sentences if CITATION_RE.search(s)) / len(sentences)


def first_relevant_rank(result: QueryResult, expected_doc: str) -> int | None:
    for rank, hit in enumerate(result.hits, start=1):
        if hit.chunk.doc_id == expected_doc:
            return rank
    return None


def evaluate(service: RagService, golden: list[GoldenQuestion],
             ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for gq in golden:
        res = service.query(gq.question, top_k=max(ks))
        row: dict[str, object] = {"id": gq.qid, "question": gq.question,
                                  "expected_doc": gq.expected_doc or "(out of scope)",
                                  "refused": res.refused,
                                  "max_similarity": round(res.max_similarity, 3),
                                  "top_doc": res.hits[0].chunk.doc_id if res.hits else None,
                                  "answer": res.answer}
        if gq.out_of_scope:
            row["refusal_correct"] = res.refused
        else:
            rank = first_relevant_rank(res, gq.expected_doc or "")
            cited_docs = {c.split("#", 1)[0] for c in res.citations}
            row.update(rank=rank, refusal_correct=not res.refused,
                       citation_coverage=0.0 if res.refused else citation_coverage(res.answer),
                       cited_expected=(gq.expected_doc in cited_docs))
        rows.append(row)

    in_scope = [r for r in rows if r["expected_doc"] != "(out of scope)"]
    oos = [r for r in rows if r["expected_doc"] == "(out of scope)"]
    n = max(len(in_scope), 1)
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"hit@{k}"] = sum(1 for r in in_scope if r["rank"] and r["rank"] <= k) / n
    metrics["mrr"] = sum(1 / r["rank"] for r in in_scope if r["rank"]) / n
    metrics["citation_coverage"] = sum(r["citation_coverage"] for r in in_scope) / n
    metrics["citation_accuracy"] = sum(1 for r in in_scope if r["cited_expected"]) / n
    metrics["false_refusal_rate"] = sum(1 for r in in_scope if r["refused"]) / n
    metrics["oos_refusal_accuracy"] = (sum(1 for r in oos if r["refused"]) / len(oos)) if oos else 1.0
    metrics = {k: round(v, 3) for k, v in metrics.items()}
    log.info(kv("eval_complete", questions=len(rows), **metrics))
    return {"metrics": metrics, "rows": rows, "in_scope": len(in_scope), "out_of_scope": len(oos)}


def write_report(results: dict[str, object], out_dir: Path, settings_note: str) -> Path:
    """Write ``eval_report.md`` and ``eval_results.json`` to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_results.json").write_text(json.dumps(results, indent=2, default=str))
    m = results["metrics"]
    lines = [
        "# RAG Evaluation Report", "",
        f"Generated: {datetime.now(UTC).isoformat(timespec='seconds')}  ",
        f"Configuration: {settings_note}  ",
        f"Questions: {results['in_scope']} in-scope, {results['out_of_scope']} out-of-scope", "",
        "## Metrics", "", "| Metric | Value |", "|---|---|",
        *[f"| {k} | {v:.3f} |" for k, v in m.items()],  # type: ignore[union-attr]
        "", "## Per-question results", "",
        "| ID | Expected doc | Top doc | Rank | Max sim | Refused | Correct refusal decision |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results["rows"]:  # type: ignore[union-attr]
        lines.append(f"| {r['id']} | {r['expected_doc']} | {r['top_doc']} | {r.get('rank', '-')} | "
                     f"{r['max_similarity']} | {r['refused']} | {r['refusal_correct']} |")
    lines += ["", "## Sample answers", ""]
    for r in results["rows"][:4] + results["rows"][-2:]:  # type: ignore[index]
        lines += [f"**{r['id']}: {r['question']}**", "", f"> {r['answer']}", ""]
    path = out_dir / "eval_report.md"
    path.write_text("\n".join(lines))
    return path
