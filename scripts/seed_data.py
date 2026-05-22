"""Seed the knowledge base with real AI/ML docs — the persistent corpus.

This script POSTs each document to the running /ingest endpoint, so all the
metadata mapping + chunking + embedding logic lives in one place (the API) —
exactly the path n8n will use later. The
script stays thin: read a file, build a JSON body, POST it.

Unlike the integration tests (which use a throwaway isolated collection that
gets deleted), this writes to the REAL collection and leaves the data there.
That corpus is what retrieval (commits 11+), the demo UI (15+), and the eval
set (41) all build on.

Idempotent: each SeedDoc has a stable source_url, and the pipeline calls
delete_by_source before re-ingesting, so re-running replaces a source's
chunks instead of duplicating them.

Run:      python scripts/seed_data.py
Requires: API server running (uvicorn aiml_knowledge_agent.api.main:app)
          + LM Studio (embed model loaded) + Qdrant (make up).
"""

from dataclasses import dataclass
from pathlib import Path

import httpx


# data/seed/ sits two levels up: scripts/ -> repo root -> data/seed
SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seed"

# Where the running API lives. Change if you run uvicorn on a different port.
API_BASE_URL = "http://localhost:8000"


@dataclass
class SeedDoc:
    """One manifest entry: a file plus its provenance metadata.

    The file holds the content; this record holds where it came from
    (source_url) and how to classify it. The source_url — not the
    filename — is what lands in Qdrant and what makes re-runs idempotent.
    """

    filename: str
    source_url: str
    doc_type: str
    framework: str | None = None
    title: str | None = None


# The corpus. Files live in data/seed/; URLs recorded by hand at download time.
SEED_DOCUMENTS: list[SeedDoc] = [
    SeedDoc(
        filename="mlabonne_llm_course.md",
        source_url="https://github.com/mlabonne/llm-course",
        doc_type="markdown",
        title="LLM Course",
    ),
    SeedDoc(
        filename="httpx_changelog.md",
        source_url="https://github.com/encode/httpx/blob/master/CHANGELOG.md",
        doc_type="changelog",
        framework="httpx",
        title="httpx Changelog",
    ),
    SeedDoc(
        filename="how-we-built-langsmith-engine-our-agent-for-improving-agents.html",
        source_url="https://www.langchain.com/blog/how-we-built-langsmith-engine-our-agent-for-improving-agents",
        doc_type="html",
        framework="langsmith",
        title="How we built LangSmith Engine: our agent for improving agents",
    ),
    SeedDoc(
        filename="validating_agentic_behavior_when_correct_is_not_deterministic_GitHub_Blog.html",
        source_url="https://github.blog/ai-and-ml/generative-ai/validating-agentic-behavior-when-correct-isnt-deterministic/",
        doc_type="html",
        title="Validating agentic behavior when correct isn't deterministic",
    ),
    SeedDoc(
        filename="claude_code_effectiveness_of_html.html",
        source_url="https://claude.com/blog/using-claude-code-the-unreasonable-effectiveness-of-html",
        doc_type="html",
        framework="claude-code",
        title="Using Claude Code: The unreasonable effectiveness of HTML",
    ),
]


def main() -> None:
    """POST every document in SEED_DOCUMENTS to the /ingest endpoint."""
    # timeout=None: the big HTML docs produce hundreds of chunks, and embedding
    # them all via LM Studio can take minutes per request.
    with httpx.Client(base_url=API_BASE_URL, timeout=None) as client:
        
        for doc in SEED_DOCUMENTS:
            text = (SEED_DIR / doc.filename).read_text(encoding="utf-8")

            body = dict(
                text=text,
                source_url=doc.source_url,
                doc_type=doc.doc_type,
                framework=doc.framework,
                title=doc.title,
            )

            resp = client.post("/ingest", json=body)
            resp.raise_for_status()
            data = resp.json()  # {"chunks_ingested": int, "source": str}

            print(f"{doc.filename}: {data['chunks_ingested']} chunks (source: {data['source']})")


if __name__ == "__main__":
    main()