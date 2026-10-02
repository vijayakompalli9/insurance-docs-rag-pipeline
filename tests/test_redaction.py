from rag_pipeline.redaction import redact


def test_redacts_each_pii_type_and_counts():
    text = ("Insured: Jane Quillfeather, phone (555) 123-4567, alt 555.222.3333, "
            "email jane.q@example.com, SSN 900-12-3456. Contact Mr. Tobias Wren.")
    result = redact(text)
    assert result.counts == {"PHONE": 2, "EMAIL": 1, "SSN": 1, "PERSON": 2}
    for secret in ("Quillfeather", "123-4567", "jane.q@", "900-12-3456", "Tobias"):
        assert secret not in result.text
    assert "[REDACTED_SSN]" in result.text


def test_does_not_redact_policy_numbers_amounts_or_dates():
    text = ("Coverage A limit is 750,000 dollars. Form NM-HO-100 dated 2026-03-14. "
            "Coverage B applies.")
    result = redact(text)
    assert result.total == 0
    assert result.text == text


def test_ssn_not_double_counted_as_phone():
    result = redact("ID 912-48-3307")
    assert result.counts == {"SSN": 1}


def test_corpus_pii_never_reaches_index(indexed_settings):
    chunks = (indexed_settings.index_dir / "chunks.jsonl").read_text()
    for raw in ("Marisol", "912-48-3307", "example.com", "(555)", "555.778", "Ashcombe", "Priya"):
        assert raw not in chunks
    assert chunks.count("[REDACTED_") == 12
