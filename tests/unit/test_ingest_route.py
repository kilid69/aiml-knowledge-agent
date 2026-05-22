"""Unit tests for POST /ingest — fake pipeline, no real services.

We use FastAPI's `TestClient` to send HTTP requests against the app in
the same process (no real server, no network). The `get_pipeline`
dependency is overridden to return a `FakePipeline` that records calls,
so tests assert on what the route handed to the pipeline.

Integration coverage (real pipeline → real Qdrant) lives in
`tests/integration/test_ingest_route.py`.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from aiml_knowledge_agent.api.main import app
from aiml_knowledge_agent.api.routes.ingest import get_pipeline


# --- fake pipeline --------------------------------------------------------


class FakePipeline:
    """Stand-in for IngestionPipeline. Records calls; returns a fixed count.

    Duck typing: the route only calls `.ingest_document(text, metadata)`,
    so this class doesn't need to inherit from anything.
    """

    def __init__(self, return_value: int = 3) -> None:
        self.calls = []  # list of (text, metadata) tuples
        self._return_value = return_value

    async def ingest_document(self, text: str, metadata: dict) -> int:
        self.calls.append((text, dict(metadata)))
        return self._return_value


# --- fixtures -------------------------------------------------------------


@pytest.fixture
def fake_pipeline() -> Iterator[FakePipeline]:
    """Swap `get_pipeline` for the duration of one test.

    `app.dependency_overrides[dep] = factory` tells FastAPI: "whenever
    something Depends on `dep`, call `factory` instead." We point it at
    a lambda that returns our fake. Cleared on teardown so the next test
    starts clean.
    """
    fake = FakePipeline()
    app.dependency_overrides[get_pipeline] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> Iterator[TestClient]:
    """TestClient with lifespan enabled.

    Using `with TestClient(app) as c:` runs the startup / shutdown
    lifespan hooks around the test. Construction of `LMStudioClient` is
    lazy (no network at __init__), so this is safe in unit tests.
    """
    with TestClient(app) as c:
        yield c

# -- helpers --------

def _request_body(**overrides: object) -> dict[str, object]:
    """Fresh minimally-valid request body each call. Override fields per test."""
    body = dict(
        text= "Hello World",
        source_url= "https://example.com/a",
        doc_type= "markdown")
    
    body.update(overrides)
    return body

    

# --- tests ----------------------------------------------------------------


def test_post_ingest_returns_chunk_count(client: TestClient, fake_pipeline: FakePipeline) -> None:
    """200 OK with chunks_ingested matching the pipeline's return value."""
    
    response = client.post("/ingest", json=_request_body())
    
    assert response.status_code == 200
    assert response.json()["chunks_ingested"] == 3


def test_post_ingest_echoes_source_in_response(client: TestClient, fake_pipeline: FakePipeline) -> None:
    """The response carries `source` back so the caller can confirm what was ingested."""
    
    response = client.post("/ingest", json=_request_body())

    assert response.status_code == 200
    assert response.json()["source"] == "https://example.com/a"


def test_post_ingest_passes_text_to_pipeline(client: TestClient, fake_pipeline: FakePipeline) -> None:
    """The pipeline receives the body's `text` verbatim."""
    
    _ = client.post("/ingest", json=_request_body())

    text_arg, _ = fake_pipeline.calls[0] # a tuple (text, metadata:dict)
    assert text_arg == "Hello World"


def test_post_ingest_renames_source_url_to_source(
    client: TestClient, fake_pipeline: FakePipeline
) -> None:
    """Request uses `source_url`; metadata uses `source` (pipeline's convention)."""
    
    _ = client.post("/ingest", json=_request_body(source_url="https://example.com/b"))

    _, metadata = fake_pipeline.calls[0]

    assert metadata["source"] == "https://example.com/b"
    assert "source_url" not in metadata


def test_post_ingest_passes_doc_type(client: TestClient, fake_pipeline: FakePipeline) -> None:
    """doc_type flows through to metadata unchanged."""

    _ = client.post("/ingest", json=_request_body(doc_type="html"))

    _, metadata = fake_pipeline.calls[0]

    assert metadata["doc_type"] == "html"


def test_post_ingest_passes_optional_fields_when_provided(
    client: TestClient, fake_pipeline: FakePipeline
) -> None:
    """framework + title appear in metadata when the caller provides them."""

    _ = client.post("/ingest", json=_request_body(title="State machine basics",
                                                  framework="langgraph"))
    
    _, metadata = fake_pipeline.calls[0]

    assert metadata["title"] == "State machine basics"
    assert metadata["framework"] == "langgraph"
    


def test_post_ingest_omits_optional_fields_when_not_provided(
    client: TestClient, fake_pipeline: FakePipeline
) -> None:
    """Optional fields the caller didn't send must NOT appear as None in metadata.

    Storing None in payload would pollute Qdrant and could confuse later
    filters. The handler should skip the key entirely if the value is None.
    """
    _ = client.post("/ingest", json=_request_body())

    _, metadata = fake_pipeline.calls[0]

    assert "title" not in metadata
    assert "framework" not in metadata


def test_post_ingest_rejects_missing_required_field(client: TestClient, fake_pipeline: FakePipeline) -> None:
    """Pydantic returns 422 when a required field is missing."""

    body = _request_body()
    del body["text"]
    
    response = client.post("/ingest", json=body)

    assert response.status_code == 422
    assert not fake_pipeline.calls


def test_post_ingest_returns_zero_when_pipeline_returns_zero(
    client: TestClient, fake_pipeline: FakePipeline
) -> None:
    """Edge case: empty document → pipeline returns 0 → response shows 0."""
    fake_pipeline._return_value = 0  # override the default

    response = client.post("/ingest", json=_request_body())

    assert response.status_code == 200
    assert response.json()["chunks_ingested"] == 0
