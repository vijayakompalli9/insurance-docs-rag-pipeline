"""Command line entry point: ``python -m rag_pipeline.cli ingest|ask|eval|serve``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_settings
from .evaluation import evaluate, load_golden, write_report
from .exceptions import RagPipelineError
from .ingest import run_ingest
from .logging_utils import configure_logging
from .service import RagService


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag_pipeline", description=__doc__)
    p.add_argument("--config", help="YAML config path (default: $RAG_CONFIG or built-in)")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="load, redact, chunk and (incrementally) index the corpus")
    ask = sub.add_parser("ask", help="answer a question with citations")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int)
    ask.add_argument("--doc-type", action="append", dest="doc_types",
                     help="restrict to a doc type (repeatable)")
    ask.add_argument("--json", action="store_true", help="print the full JSON result")
    ev = sub.add_parser("eval", help="run the golden-question evaluation")
    ev.add_argument("--golden", default="eval/golden_questions.yaml")
    ev.add_argument("--out", default="docs/sample_output")
    serve = sub.add_parser("serve", help="run the FastAPI app with uvicorn")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = load_settings(args.config)
    configure_logging(settings.log_level)
    try:
        if args.command == "ingest":
            print(json.dumps(run_ingest(settings).as_dict(), indent=2))
        elif args.command == "ask":
            result = RagService(settings).query(args.question, args.top_k, args.doc_types)
            if args.json:
                print(json.dumps(result.as_dict(), indent=2))
            else:
                print(result.answer)
                print(f"\nrefused={result.refused} max_similarity={result.max_similarity:.3f}")
                for hit in result.hits:
                    print(f"  {hit.score:.3f}  {hit.chunk.chunk_id}  ({hit.chunk.section})")
        elif args.command == "eval":
            results = evaluate(RagService(settings), load_golden(Path(args.golden)))
            note = (f"embedding={settings.embedding_provider}, generator="
                    f"{settings.generator_provider}, top_k=5, rerank={settings.rerank}, "
                    f"refusal_threshold={settings.refusal_threshold}")
            path = write_report(results, Path(args.out), note)
            print(json.dumps(results["metrics"], indent=2))
            print(f"report written to {path}")
        elif args.command == "serve":
            import uvicorn

            from .api import create_app

            uvicorn.run(create_app(settings), host=args.host, port=args.port)
    except RagPipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
