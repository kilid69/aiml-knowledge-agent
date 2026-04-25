# AI/ML Knowledge Agent

A RAG-powered research assistant with multi-agent orchestration, automated
ingestion, and evaluation pipelines. Built to stay current on AI/ML tooling
by continuously ingesting documentation, blog posts, changelogs, and papers,
then answering complex questions across everything it has seen.

## Status

Phase 1 — basic ingestion pipeline (in progress).

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Project skeleton, Docker, FastAPI `/health` | done |
| 1 | Ingestion pipeline (chunker → embedder → vector store) | in progress |
| 2 | Retrieval + naive QA | planned |
| 3 | Frontend (chat UI) | planned |
| 4 | Smarter chunking strategies + comparison | planned |
| 5 | LangGraph multi-agent orchestration | planned |
| 6+ | Evaluation, scheduled ingestion, deployment | planned |

### What's working today

- **Qdrant** running locally via Docker Compose (with payload indices on
  `source`, `doc_type`, `framework`, `date_ingested`, `section_title`).
- **LM Studio client** (`LMStudioClient`) for `embed()` and `chat()`
  (streaming and non-streaming) against an OpenAI-compatible endpoint.
- **Embedding format helpers** for Qwen3-Embedding-4B
  (instruction-formatted queries, raw-passthrough documents, Matryoshka
  truncation + L2 normalization).
- **Vector store CRUD** — idempotent collection init, batch upsert,
  delete-by-source.
- **SimpleChunker** — character-based recursive splitting via LangChain's
  `RecursiveCharacterTextSplitter`, baseline strategy.

## Tech stack

- **Python 3.11+**, [`uv`](https://docs.astral.sh/uv/) for dependency management
- **FastAPI** for the HTTP API
- **Qdrant** (vector DB) — cosine distance, payload-indexed metadata
- **LM Studio** as a local OpenAI-compatible LLM/embedding server
  - Chat: configurable (currently a Qwen3 chat model)
  - Embeddings: Qwen3-Embedding-4B (native 2560-d, truncated to 1024-d)
- **LangChain** text splitters; **LangGraph** for agent orchestration (later)

## Running locally

### Prerequisites

- Docker (for Qdrant)
- LM Studio with the configured chat + embedding models loaded
- `uv` installed

### Setup

```bash
# Install Python deps into a managed venv
uv sync

# Start Qdrant
make up

# Start the FastAPI app
uv run uvicorn aiml_knowledge_agent.api.main:app --reload
```

Health check:

```bash
curl localhost:8000/health
```

### Tests

```bash
# Unit tests only — no external services required
uv run pytest tests/unit -v

# Full suite — integration tests skip gracefully if Qdrant or LM Studio
# are unreachable / models aren't loaded
uv run pytest -v
```

### Useful commands

| Command | What it does |
|---------|--------------|
| `make up` | Start Qdrant in the background |
| `make down` | Stop Qdrant |
| `make logs` | Tail Qdrant logs |
| `make lint-backend` | Ruff check + format check |

## Project layout

```
src/aiml_knowledge_agent/
├── api/              # FastAPI app + config (pydantic-settings)
├── models/           # LMStudioClient (embed, chat, stream)
└── ingestion/
    ├── embed_format.py   # Qwen3 query/document formatting + Matryoshka helper
    ├── vector_store.py   # Qdrant collection schema + CRUD
    └── chunkers/
        ├── base.py       # BaseChunker ABC + Chunk dataclass
        └── simple.py     # SimpleChunker (RecursiveCharacterTextSplitter)

tests/
├── unit/             # No external services
└── integration/      # Hits real Qdrant and LM Studio (auto-skips if down)
```

Configuration is centralized in `src/aiml_knowledge_agent/api/config.py`
(pydantic-settings). Override via env vars or a `.env` file.
