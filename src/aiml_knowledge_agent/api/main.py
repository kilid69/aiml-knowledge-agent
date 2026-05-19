from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient

from aiml_knowledge_agent.api.config import settings
from aiml_knowledge_agent.api.routes import ingest as ingest_route
from aiml_knowledge_agent.ingestion.chunkers.simple import SimpleChunker
from aiml_knowledge_agent.ingestion.embedder import Embedder
from aiml_knowledge_agent.ingestion.pipeline import IngestionPipeline
from aiml_knowledge_agent.models.llm_client import LMStudioClient


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build long-lived collaborators on startup, tear them down on shutdown.

    Everything before `yield` runs ONCE when the app starts. Everything
    after runs when it stops. The pipeline + its TCP connection pool
    live for the whole process — not per request.
    """

    lm_client = LMStudioClient()
    chunker   = SimpleChunker()
    embedder  = Embedder(client=lm_client)
    app.state.pipeline  = IngestionPipeline(chunker=chunker, embedder=embedder)
    app.state.lm_client = lm_client   # keep a handle so we can close it below

    yield 

    await app.state.lm_client.aclose()


app = FastAPI(
    title="AI/ML Knowledge Agent",
    description="RAG-powered research assistant for AI/ML tooling.",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount the /ingest route under the app.
app.include_router(ingest_route.router)


class HealthResponse(BaseModel):
    """Shape of the /health response."""

    api: str
    qdrant: str
    lm_studio: str


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness + dependency health check.

    Each field is a short status string: "ok", "unreachable", etc.
    Keep this endpoint fast — it's called by load balancers / uptime probes.
    """

    # The API itself is trivially "ok" — if this handler is running, the API works.
    api_status = "ok"

    # checking the Qdrant server availability by a round-trip call
    try:
        QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port).get_collections()
        qdrant_status = "ok"
    except Exception:
        qdrant_status = "unreachable"

    # LM Studio exposes an OpenAI-compatible REST API. Check if it is available.
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{settings.lm_studio_url}/models")
            lm_studio_status = "ok" if r.status_code == 200 else "unreachable"
    except Exception:
        lm_studio_status = "unreachable"

    return HealthResponse(
        api=api_status,
        qdrant=qdrant_status,
        lm_studio=lm_studio_status,
    )
