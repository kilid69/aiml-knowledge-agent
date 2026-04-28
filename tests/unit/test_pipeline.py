"""Unit tests for IngestionPipeline — fakes for chunker/embedder, spies for Qdrant.

These tests verify the *orchestration* logic without touching real services:
  - the right helpers are called, in the right order, with the right args
  - metadata stamping (`date_ingested`) behaves correctly
  - re-ingestion path (`delete_by_source`) only fires when there's a source
  - payload shape matches what the rest of the system expects
  - return value matches the chunk count

Integration coverage (real LM Studio + real Qdrant) lives in
`tests/integration/test_pipeline.py`.
"""

from datetime import datetime
from typing import Any

import pytest

from aiml_knowledge_agent.ingestion.chunkers.base import BaseChunker, Chunk
from aiml_knowledge_agent.ingestion.embedder import Embedder
from aiml_knowledge_agent.ingestion import pipeline as pipeline_module
from aiml_knowledge_agent.ingestion.pipeline import IngestionPipeline


# --- fakes ----------------------------------------------------------------


class FakeChunker(BaseChunker):
    """Returns a fixed list of chunks regardless of input.

    The pipeline tests don't care *how* chunking happens — they just need
    a known number of chunks coming out so they can assert on counts and
    payload shapes downstream.
    """

    def __init__(self, n_chunks: int = 3) -> None:
        self._n = n_chunks
        self.calls: list[tuple[str, dict[str, Any]]] = [] # records (text, metadata) per call

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        self.calls.append((text, dict(metadata)))
        return [
            Chunk(text=f"chunk-{i}-of-{text[:10]}", metadata=dict(metadata), chunk_index=i)
            for i in range(self._n)
        ]
        

class FakeLMStudioClient:
    """Same shape as the embedder unit test's fake — no network."""

    NATIVE_DIM = 2560

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        # one-hot-ish vector per text; truncate_and_normalize will L2-normalize.
        return [[1.0] + [0.0] * (self.NATIVE_DIM - 1) for _ in texts]


# --- spy fixture ----------------------------------------------------------


class _Spies:
    """Bundle of recorded calls to vector_store helpers.

    Lets each test assert on what the pipeline did to Qdrant without
    actually talking to Qdrant.
    """

    def __init__(self) -> None:
        self.init_calls: int = 0
        self.delete_calls: list[str] = []
        self.upsert_calls: list[list] = []  # list of point batches


@pytest.fixture
def spies(monkeypatch: pytest.MonkeyPatch) -> _Spies:
    """Replace vector_store helpers (as imported by pipeline.py) with spies."""
    s = _Spies()

    def fake_init() -> None:
        s.init_calls += 1

    def fake_delete(source: str) -> None:
        s.delete_calls.append(source)

    def fake_upsert(points: list) -> None:
        s.upsert_calls.append(list(points))

    # IMPORTANT: patch the *names bound inside pipeline.py*, not the source
    # module — pipeline.py did `from ... import init_collection`, so it
    # holds its own reference. Patching `vector_store.init_collection`
    # would not be seen by the pipeline.
    monkeypatch.setattr(pipeline_module, "init_collection", fake_init)
    monkeypatch.setattr(pipeline_module, "delete_by_source", fake_delete)
    monkeypatch.setattr(pipeline_module, "upsert_points", fake_upsert)

    return s


# --- helpers --------------------------------------------------------------


def _make_pipeline(n_chunks: int = 3) -> tuple[IngestionPipeline, FakeChunker, FakeLMStudioClient]:
    """Build a pipeline wired to fakes, ready for a test to drive."""
    chunker = FakeChunker(n_chunks=n_chunks)
    fake_client = FakeLMStudioClient()
    embedder = Embedder(client=fake_client)  # type: ignore[arg-type]
    return IngestionPipeline(chunker=chunker, embedder=embedder), chunker, fake_client


# --- tests ----------------------------------------------------------------


async def test_returns_chunk_count(spies: _Spies) -> None:
    """ingest_document returns len(chunks) the chunker produced."""
    pipeline, _, _ = _make_pipeline(n_chunks=5)

    n = await pipeline.ingest_document("some text", metadata={"source": "x"})

    assert n == 5


async def test_calls_init_collection_once(spies: _Spies) -> None:
    """init_collection runs exactly once per ingest (idempotency lives in the helper)."""
    pipeline, _, _ = _make_pipeline()

    await pipeline.ingest_document("some text", metadata={"source": "x"})

    assert spies.init_calls == 1


async def test_stamps_date_ingested_when_missing(spies: _Spies) -> None:
    """If caller didn't pass date_ingested, the pipeline adds it.

    Don't assert the exact value (it's a fresh UTC timestamp); assert that
    *something* is there and it parses as ISO 8601.
    """
    pipeline, chunker, _ = _make_pipeline()

    await pipeline.ingest_document("some text", metadata={"source": "x"})

    # FakeChunker recorded a copy of the metadata it was handed.
    _, recorded = chunker.calls[0]
    assert "date_ingested" in recorded
    # If this parses, it's valid ISO 8601 — good enough for a sanity check.
    datetime.fromisoformat(recorded["date_ingested"])


async def test_preserves_caller_date_ingested(spies: _Spies) -> None:
    """If caller already set date_ingested, the pipeline must not overwrite it."""
    pipeline, chunker, _ = _make_pipeline()
    fixed = "2024-01-01T00:00:00+00:00"

    await pipeline.ingest_document(
        "some text", metadata={"source": "x", "date_ingested": fixed}
    )

    _, recorded = chunker.calls[0]
    assert recorded["date_ingested"] == fixed


async def test_does_not_mutate_callers_metadata_dict(spies: _Spies) -> None:
    """Caller's dict is untouched — pipeline copies before stamping."""
    pipeline, _, _ = _make_pipeline()
    callers_dict: dict[str, Any] = {"source": "x"}

    await pipeline.ingest_document("some text", metadata=callers_dict)

    # The pipeline stamped date_ingested on its own copy, not on ours.
    assert "date_ingested" not in callers_dict
    # And the caller's dict still has exactly what they put in it.
    assert callers_dict == {"source": "x"}


async def test_delete_by_source_called_when_source_present(spies: _Spies) -> None:
    """Re-ingestion: the source's prior chunks get cleaned up first."""
    pipeline, _, _ = _make_pipeline()

    await pipeline.ingest_document(
        "some text", metadata={"source": "https://example.com/a"}
    )

    assert spies.delete_calls == ["https://example.com/a"]


async def test_delete_by_source_skipped_when_no_source(spies: _Spies) -> None:
    """No source → no delete. Avoids deleting on `source=None`."""
    pipeline, _, _ = _make_pipeline()

    await pipeline.ingest_document("some text", metadata={})

    assert spies.delete_calls == []


async def test_upserts_one_point_per_chunk(spies: _Spies) -> None:
    """Every chunk becomes exactly one PointStruct in the upsert call."""
    pipeline, _, _ = _make_pipeline(n_chunks=4)

    await pipeline.ingest_document("some text", metadata={"source": "x"})

    # Exactly one batched upsert, containing exactly four points.
    assert len(spies.upsert_calls) == 1
    assert len(spies.upsert_calls[0]) == 4


async def test_point_payload_carries_text_and_chunk_index(spies: _Spies) -> None:
    """Payload shape: chunk.text and chunk_index land in the payload."""
    pipeline, chunker, _ = _make_pipeline(n_chunks=3)

    await pipeline.ingest_document("hello world", metadata={"source": "x"})

    # Cross-reference: chunker produced these chunks; each must show up
    # in the corresponding point's payload.
    produced_chunks = chunker.chunk("hello world", metadata={"source": "x"})  # deterministic
    points = spies.upsert_calls[0]
    for i, point in enumerate(points):
        assert point.payload["text"] == produced_chunks[i].text
        assert point.payload["chunk_index"] == i


async def test_point_payload_carries_metadata(spies: _Spies) -> None:
    """User-supplied metadata fields end up in the payload too."""
    pipeline, _, _ = _make_pipeline(n_chunks=2)
    metadata = {
        "source": "https://example.com/a",
        "doc_type": "markdown",
        "framework": "langgraph",
    }

    await pipeline.ingest_document("some text", metadata=metadata)

    for point in spies.upsert_calls[0]:
        assert point.payload["source"] == "https://example.com/a"
        assert point.payload["doc_type"] == "markdown"
        assert point.payload["framework"] == "langgraph"


async def test_empty_text_returns_zero_and_skips_upsert(spies: _Spies) -> None:
    """Zero chunks in → zero points stored, no upsert call."""
    pipeline, _, _ = _make_pipeline(n_chunks=0)

    n = await pipeline.ingest_document("", metadata={"source": "x"})

    assert n == 0
    assert spies.upsert_calls == []


async def test_raises_when_chunk_has_no_vector() -> None:
    """_chunk_to_point loudly fails if vector is None — protects against silent corruption."""
    from aiml_knowledge_agent.ingestion.pipeline import _chunk_to_point

    chunk = Chunk(text="hello", chunk_index=0)
    assert chunk.vector is None  # sanity: default state

    with pytest.raises(ValueError):
        _chunk_to_point(chunk)
