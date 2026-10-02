import json

from conftest import ROOT
from rag_pipeline.cli import main


def test_cli_ingest_ask_eval_end_to_end(settings, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RAG_CORPUS_DIR", str(settings.corpus_dir))
    monkeypatch.setenv("RAG_INDEX_DIR", str(settings.index_dir))
    assert main(["ingest"]) == 0
    ingest_out = json.loads(capsys.readouterr().out)
    assert ingest_out["pii_redacted"]["SSN"] == 2

    assert main(["ask", "What is the Coverage D loss of use limit?", "--json"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["sources"][0]["doc_id"] == "ho-property-coverages"

    out_dir = tmp_path / "report"
    golden = ROOT / "eval" / "golden_questions.yaml"
    assert main(["eval", "--golden", str(golden), "--out", str(out_dir)]) == 0
    metrics = json.loads((out_dir / "eval_results.json").read_text())["metrics"]
    assert metrics["hit@5"] >= 0.9
    assert metrics["mrr"] >= 0.8
    assert metrics["oos_refusal_accuracy"] == 1.0
    assert metrics["false_refusal_rate"] == 0.0
    assert "| hit@1 |" in (out_dir / "eval_report.md").read_text()


def test_cli_reports_missing_index(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "missing"))
    assert main(["ask", "anything"]) == 1
    assert "run `ingest` first" in capsys.readouterr().err
