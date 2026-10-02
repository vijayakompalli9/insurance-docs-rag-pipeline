# Insurance Docs RAG Pipeline

> **Portfolio project.** Independently built demonstration using synthetic data. It is not code from, or affiliated with, any current or former employer or client. Developed with AI-assisted tooling and reviewed by the author.

Underwriters and claims analysts at an insurer spend a lot of time searching long policy wordings, underwriting guidelines and claims procedures to answer narrow questions: "Is sewer backup excluded?", "What Coverage A limit can a senior underwriter bind?", "How fast must the adjuster call the insured?". This repo builds a retrieval-augmented generation (RAG) pipeline over a small fictional corpus for **Northwind Mutual Insurance (fictional)**. It is built the way a data pipeline should be. Documents are ingested, normalized and PII-redacted before anything is embedded. Chunks carry stable ids and content hashes, so re-indexing is incremental. Answers cite the exact chunk they came from, and the system refuses when retrieval confidence is low. An evaluation harness measures answer quality instead of assuming it.

## What this demonstrates

- **Data-engineering-grade ingestion:** loader with front-matter validation, Unicode/whitespace normalization, regex PII redaction with per-type counts logged, heading-aware chunking with overlap.
- **Incremental indexing:** each chunk carries `doc_id`, `section`, `chunk_id` and a SHA-256 `content_hash`. Unchanged chunks are never re-embedded, deleted chunks are pruned, and changing the embedding model forces a full rebuild.
- **Provider abstraction:** offline default embedder and answer generator, plus AWS Bedrock (Titan embeddings, Claude via the Converse API) and Azure OpenAI adapters. The adapters import their SDKs lazily and are only constructed when configured.
- **Grounded answers:** every answer sentence carries a `[doc_id#chunk]` citation, and the system refuses when the best similarity is below a threshold. The prompt template has explicit grounding and "say you don't know" rules.
- **Measured quality:** a golden question set with hit@k, MRR, citation coverage/accuracy and refusal correctness, written to a markdown report.
- **Production hygiene:** FastAPI service, CLI, structured key=value logs, custom exceptions, typed config (defaults < YAML < env), and secrets only from the environment. Also included: Docker, CI, ruff and pytest.

## Architecture

```mermaid
flowchart LR
    subgraph Ingest["Ingestion (python -m rag_pipeline.cli ingest)"]
        A[corpus/*.md<br/>front matter + markdown] --> B[Loader +<br/>normalization]
        B --> C[PII redaction<br/>EMAIL / SSN / PHONE / PERSON<br/>counts logged]
        C --> D[Heading-aware chunker<br/>overlap + content hash]
        D --> E{hash unchanged?}
        E -- yes --> F[reuse stored vector]
        E -- no --> G[EmbeddingProvider<br/>local hashing / Bedrock Titan / Azure OpenAI]
        F --> H[(Vector store<br/>vectors.npy + chunks.jsonl + manifest.json)]
        G --> H
    end
    subgraph Query["Query (CLI ask / FastAPI /query)"]
        Q[Question] --> R[Embed query]
        R --> S[Cosine top-k<br/>+ doc_type filter]
        H --> S
        S --> T[Lexical re-rank]
        T --> U{max similarity<br/>>= threshold?}
        U -- no --> V[Refuse:<br/>I don't know]
        U -- yes --> W[AnswerGenerator<br/>extractive / Bedrock Claude / Azure OpenAI]
        W --> X[Answer with<br/>doc_id#chunk citations]
    end
    subgraph Eval["Evaluation (cli eval)"]
        Y[eval/golden_questions.yaml] --> Z[hit@k, MRR, citation coverage,<br/>refusal correctness]
        Z --> AA[docs/sample_output/eval_report.md]
    end
```

## Tech stack

Python 3.11+ · NumPy (cosine index) · scikit-learn `HashingVectorizer` (offline embeddings) · FastAPI + Uvicorn · Pydantic · PyYAML · pytest · ruff · Docker · GitHub Actions. Optional: `boto3` (AWS Bedrock), `openai` (Azure OpenAI).

## Project layout

```
insurance-docs-rag-pipeline/
├── corpus/                      # 10 original fictional Northwind Mutual documents (2 contain fake PII)
├── eval/golden_questions.yaml   # 15 in-scope + 2 out-of-scope questions with expected source doc
├── src/rag_pipeline/
│   ├── config.py                # dataclass settings: defaults < YAML < env < overrides
│   ├── documents.py             # loader, front-matter validation, normalization
│   ├── redaction.py             # regex PII redaction with per-type counts
│   ├── chunking.py              # heading-aware chunking, overlap, stable ids, content hash
│   ├── ingest.py                # pipeline + incremental re-indexing by hash
│   ├── embeddings.py            # EmbeddingProvider: local hashing, Bedrock Titan, Azure OpenAI
│   ├── vector_store.py          # numpy cosine index persisted to disk, doc_type filtering
│   ├── retrieval.py             # top-k + lexical re-rank
│   ├── generation.py            # extractive default, Bedrock Converse, Azure OpenAI chat
│   ├── prompts/answer_prompt.md # grounding + refusal instructions for LLM generators
│   ├── service.py               # query orchestration + refusal threshold
│   ├── evaluation.py            # metrics + markdown/JSON report
│   ├── api.py                   # FastAPI /health, /query
│   ├── cli.py                   # ingest | ask | eval | serve
│   ├── exceptions.py, logging_utils.py
├── tests/                       # unit, API and end-to-end tests
├── docs/sample_output/          # real output from a local run
├── config.example.yaml, .env.example
├── Dockerfile, .dockerignore, Makefile, pyproject.toml
├── requirements.txt, requirements-dev.txt
└── .github/workflows/ci.yml
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .

python -m rag_pipeline.cli ingest                       # build index in data/index
python -m rag_pipeline.cli ask "Is sewer backup covered?"
python -m rag_pipeline.cli ask "Mold referral threshold?" --doc-type claims_sop
python -m rag_pipeline.cli eval                         # writes docs/sample_output/eval_report.md
python -m rag_pipeline.cli serve --port 8000            # FastAPI on http://127.0.0.1:8000/docs

pytest            # or: make test
ruff check .      # or: make lint
```

`make install`, `make ingest`, `make eval`, `make run`, `make test` and `make lint` wrap the same commands. Everything above runs fully offline, with no credentials.

## Sample output

The examples below are real output from a local run, trimmed. Full files are in [`docs/sample_output/`](docs/sample_output/).

**Ingest (first run).** PII is redacted before chunking, and the counts are logged per document:

```
event=pii_redacted doc_id=claims-fnol-sop total=6 email=1 person=2 phone=2 ssn=1
event=pii_redacted doc_id=fraud-red-flags total=6 email=2 person=2 phone=1 ssn=1
event=ingest_complete documents=10 chunks_total=51 embedded=51 reused=0 removed=0 full_rebuild=False
```

**Ingest (second run, no changes).** No chunks are re-embedded:

```
event=ingest_complete documents=10 chunks_total=51 embedded=0 reused=51 removed=0 full_rebuild=False
```

**Ask:**

```
$ python -m rag_pipeline.cli ask "What windstorm deductible is required for risks near tidal water?"
Risks within 1,000 feet of tidal water require a minimum 2 percent windstorm deductible. [uw-homeowners-guideline#coastal-and-wildfire-zones-0]

refused=False max_similarity=0.286
  0.393  uw-homeowners-guideline#coastal-and-wildfire-zones-0  (Homeowners Underwriting Guideline > Coastal and Wildfire Zones)
  0.198  ho-perils-exclusions#exclusion-flood-0  (Perils Insured Against and Exclusions > Exclusion - Flood)
  ...

$ python -m rag_pipeline.cli ask "What is the interest rate on a 30 year mortgage?"
I don't know based on the indexed documents.

refused=True max_similarity=0.065
```

**Evaluation** (`docs/sample_output/eval_report.md`), with 15 in-scope and 2 out-of-scope questions:

| Metric | Value |
|---|---|
| hit@1 | 0.933 |
| hit@3 | 1.000 |
| hit@5 | 1.000 |
| mrr | 0.967 |
| citation_coverage | 1.000 |
| citation_accuracy | 1.000 |
| false_refusal_rate | 0.000 |
| oos_refusal_accuracy | 1.000 |

The one hit@1 miss is informative. For "Is sewer or drain backup covered…" the water-damage SOP ranked above the policy exclusion. Both documents legitimately discuss the topic, and the policy doc came in at rank 2.

**API** (`docs/sample_output/api_examples.txt`):

```json
{
  "question": "What is the per person limit for medical payments to others?",
  "answer": "We pay necessary medical expenses incurred within three years ... [ho-liability#coverage-f-medical-payments-to-others-0] Coverage F applies regardless of fault, up to 5,000 dollars per person. [ho-liability#coverage-f-medical-payments-to-others-0] ...",
  "refused": false,
  "max_similarity": 0.224,
  "citations": ["ho-liability#coverage-f-medical-payments-to-others-0", "ho-liability#coverage-e-personal-liability-0"],
  "sources": [{"chunk_id": "ho-liability#coverage-f-medical-payments-to-others-0", "doc_type": "policy_wording", "score": 0.3555, "...": "..."}]
}
```

## Configuration

Settings are resolved in this order: built-in defaults, then a YAML file (`--config` or `$RAG_CONFIG`; see [`config.example.yaml`](config.example.yaml)), then environment variables, then explicit overrides. Each later source wins.

| Setting | Default | Env var |
|---|---|---|
| `corpus_dir` | `corpus` | `RAG_CORPUS_DIR` |
| `index_dir` | `data/index` | `RAG_INDEX_DIR` |
| `embedding_provider` | `local` | `RAG_EMBEDDING_PROVIDER` |
| `generator_provider` | `extractive` | `RAG_GENERATOR_PROVIDER` |
| `refusal_threshold` | `0.12` | `RAG_REFUSAL_THRESHOLD` |
| `top_k` | `5` | `RAG_TOP_K` |
| `chunking.max_words` / `overlap_words` | `120` / `25` | - |

Unknown keys are rejected with `ConfigError`. So are secret-looking keys (`api_key`, `token`, ...) in YAML.

### Swapping to Bedrock / Azure OpenAI

The cloud adapters are implemented behind the same `EmbeddingProvider` / `AnswerGenerator` interfaces. Their SDKs are imported only inside the adapter constructors, and the default install does not include them. They are never called in tests.

**AWS Bedrock.** Titan Text Embeddings v2 via `invoke_model`, and Anthropic Claude via the Converse API:

```bash
pip install boto3
export AWS_REGION=us-east-1          # credentials come from the standard boto3 chain (profile / role)
```

```yaml
# config.yaml
embedding_provider: bedrock
generator_provider: bedrock
bedrock:
  region: us-east-1
  embedding_model_id: amazon.titan-embed-text-v2:0
  embedding_dimensions: 1024
  generation_model_id: "<bedrock-claude-model-or-inference-profile-id>"
```

**Azure OpenAI:**

```bash
pip install openai
export AZURE_OPENAI_API_KEY="<your-azure-openai-key>"     # env only, never YAML
```

```yaml
# config.yaml
embedding_provider: azure_openai
generator_provider: azure_openai
azure_openai:
  endpoint: "https://<your-resource-name>.openai.azure.com/"
  api_version: "2024-06-01"
  embedding_deployment: "<your-embedding-deployment>"
  chat_deployment: "<your-chat-deployment>"
```

Then run `RAG_CONFIG=config.yaml python -m rag_pipeline.cli ingest`. The index manifest records the embedder signature. Switching embedders triggers a full rebuild automatically (`full_rebuild=True`), and the retriever refuses to query an index built with a different embedder (`EmbeddingMismatchError`). You can mix providers, for example local embeddings with a Bedrock generator. Re-tune `refusal_threshold` after switching embedders, because similarity scales differ between models.

## Design decisions

- **Stateless hashing embedder as the offline default.** A fitted TF-IDF model changes every vector whenever the vocabulary changes, which would make hash-based incremental indexing meaningless. `HashingVectorizer` (unigrams + bigrams, sublinear TF, L2 norm) is stateless: an unchanged chunk always gets the same vector. The section path is prepended to chunk text before embedding so short chunks keep their context.
- **Redact before chunking.** PII never reaches chunk text, hashes, embeddings, the on-disk index, logs or a remote provider.
- **Stable, readable chunk ids** (`doc_id#section-slug-N`). Ids come from the section heading, not the global position, so editing one section does not change ids (or force re-embeds) elsewhere. They also double as human-readable citations.
- **Numpy instead of FAISS.** At this corpus size, a brute-force dot product over normalized vectors is exact, has no extra dependency and takes well under a millisecond. The `VectorStore` class is the seam where FAISS, pgvector or OpenSearch would plug in.
- **Refusal before generation.** The threshold is checked on the raw cosine of the best hit, before any LLM is called. Out-of-scope questions cost nothing and cannot be hallucinated. In the sample run, out-of-scope questions scored about 0.05 and the weakest in-scope question scored 0.151, so the 0.12 threshold has some margin on both sides.
- **Extractive default generator.** It picks sentences by term overlap, prefers higher-ranked chunks, drops weak tail sentences and attaches a citation to each sentence. Sentences containing redaction tokens are never surfaced.
- **Citations are validated.** For LLM generators, only citations that match retrieved chunk ids are kept, so a hallucinated id is dropped.

## Security considerations

- **PII redaction before embedding.** Emails, SSN-like patterns, US phone numbers and labelled or honorific names are replaced with `[REDACTED_<TYPE>]` before chunking. Counts (never values) are logged per document. A test asserts that no raw PII from the corpus appears in the persisted index.
- **No data leaves the machine in default mode.** The local embedder and extractive generator make no network calls. Cloud SDKs are not installed by default and are only imported when explicitly configured.
- **Secrets via environment only.** `AZURE_OPENAI_API_KEY` comes from the environment. AWS credentials come from the standard boto3 chain (profile, SSO or IAM role). The config loader rejects secret-like keys in YAML. `.env` is git-ignored, and only `.env.example` / `config.example.yaml` with placeholders are committed.
- **Logs avoid content.** Query logs record refusal status, max similarity, top doc and latency, not the question text.
- **Container runs as a non-root user.**
- The regex redactor is a guard rail, not a DLP product. Names are only detected in labelled or honorific contexts, so a free-text name in prose would be missed. See Limitations.

## Testing

27 tests (`pytest`), all passing on Python 3.11 and 3.13:

- `test_redaction.py`: each PII type is detected and counted, no false positives on amounts, form numbers or dates, and no raw corpus PII appears in the index.
- `test_chunking.py`: heading hierarchy, no chunk crosses a heading, exact overlap between consecutive windows, full coverage, stable ids with hash change on edit, normalization and front-matter validation.
- `test_ingest_incremental.py`: a second run embeds 0 chunks (asserted with a counting embedder), editing one sentence re-embeds exactly 1 chunk, deleting a document prunes its chunks, and changing the embedder forces a full rebuild.
- `test_retrieval.py`: the right document ranks first for several questions, the doc_type filter works, out-of-scope questions are refused, the threshold is configurable, embedder/index mismatches raise, and cloud providers fail fast without configuration.
- `test_api.py`: FastAPI `TestClient` covers `/health`, `/query`, validation (422) and missing index (503).
- `test_end_to_end.py`: CLI `ingest`, then `ask`, then `eval` against the real golden set, with metric thresholds asserted.

## Limitations

- The corpus is small (10 short hand-written documents, 51 chunks), and the same author wrote the golden set. Treat the metrics as a regression baseline, not a benchmark.
- Bag-of-words hashing embeddings do not capture synonyms. "Is earthquake damage covered?" can rank the water-damage SOP highly because of the shared words "damage" and "covered". A dense embedding model (Titan, Azure OpenAI) fixes this.
- `citation_coverage` is 1.0 by construction for the extractive generator. It becomes informative with LLM generators.
- The extractive generator can include a loosely related third sentence. It does not synthesize or reason.
- The Bedrock and Azure adapters are written against the documented SDK APIs, but they are not exercised by tests or CI because those need credentials and network.
- The Docker image was not built here because no Docker daemon was available in the build environment.

## Future enhancements

- Hybrid retrieval (BM25 + dense) and a cross-encoder re-ranker.
- An LLM-as-judge faithfulness metric, plus answer-keyword recall in the golden set.
- An NER-based PII detector (e.g. Presidio) layered on top of the regex rules.
- S3 / ADLS document sources and an Airflow DAG for scheduled incremental ingestion.
- A FAISS or pgvector backend behind the `VectorStore` interface, with per-tenant indexes.
- Recorded-response contract tests for the cloud adapters.

## License

MIT. See [LICENSE](LICENSE). Copyright 2026 Vijaya Lakshmi Kompalli.
