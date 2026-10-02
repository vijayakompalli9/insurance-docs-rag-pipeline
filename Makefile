PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: install ingest ask eval run test lint clean

install:
	python3 -m venv .venv
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

ingest:
	$(PY) -m rag_pipeline.cli ingest

ask:
	$(PY) -m rag_pipeline.cli ask "$(Q)"

eval:
	$(PY) -m rag_pipeline.cli eval

run: ingest
	$(PY) -m rag_pipeline.cli serve --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

clean:
	rm -rf data .pytest_cache .ruff_cache
