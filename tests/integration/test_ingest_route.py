"""Integration test for POST /ingest — real LM Studio + real Qdrant.

The unit tests use a FakePipeline and never touch real services. This
test starts the real FastAPI app (lifespan runs, building the real
IngestionPipeline), hits the endpoint with TestClient, and verifies the
points actually land in Qdrant.

Uses the `isolated_collection` fixture so the production collection is
never touched.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from aiml_knowledge_agent.api.main import app
from aiml_knowledge_agent.ingestion.vector_store import get_client


@pytest.fixture
def client() -> Iterator[TestClient]:
    """TestClient with lifespan — real pipeline is built on entry."""
    with TestClient(app) as c:
        yield c


async def test_post_ingest_end_to_end(
    client: TestClient,
    isolated_collection: str,
) -> None:
    """POST /ingest with real text → real chunks land in real Qdrant."""

    SAMPLE_DOC = """
    # LangGraph state

    LangGraph models the agent as a state machine. The state is a TypedDict
    that flows from node to node — each node receives the current state, may
    return updates, and the framework merges those updates into the next state.

    ## Why state machines

    Cycles are first-class in LangGraph. A critic node can route back to a
    synthesizer node, and the framework tracks how many times you've been
    through the loop so you can cap retries. Plain DAG frameworks would have
    to either flatten this into one giant node or punt the cycle to the user.

    ## Persistence

    Checkpointers persist state between invocations, which is what enables
    human-in-the-loop pauses and conversation memory across calls.
    """ * 3

    body = dict(
        text=SAMPLE_DOC,
        source_url="https://example.com/end-to-end",
        doc_type= "markdown",
        framework="langchain"
    )
    response = client.post("/ingest", json=body)
    
    assert response.status_code == 200
    assert response.json()["chunks_ingested"] > 1
    assert response.json()["source"] == "https://example.com/end-to-end"
    
    qdrant_client = get_client()

    assert qdrant_client.count(collection_name=isolated_collection).count == response.json()["chunks_ingested"]
