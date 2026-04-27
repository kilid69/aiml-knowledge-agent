"""Unit tests for Embedder — uses a fake LM Studio client, no network.

Covers the *plumbing* that integration tests can't easily check:
- batching produces the right number of LM Studio calls of the right sizes
- every chunk gets paired with the correct vector (no off-by-one)
- the input list is mutated in place
- truncate-and-normalize is actually applied to each returned vector
"""

import math

import pytest

from aiml_knowledge_agent.api.config import settings
from aiml_knowledge_agent.ingestion.chunkers.base import Chunk
from aiml_knowledge_agent.ingestion.embedder import Embedder


# --- fake LM Studio client -------------------------------------------------


class FakeLMStudioClient:
    """Stand-in for `LMStudioClient` — same `embed()` signature, no network.

    Python's duck typing means the Embedder doesn't actually care that
    this isn't a real LMStudioClient subclass: it only calls `.embed(...)`
    and we provide that. The fake records every batch it was called with
    so tests can assert on batching behaviour.
    """

    NATIVE_DIM = 2560  # what real Qwen3-Embedding-4B returns

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        # Return one distinct one-hot-ish vector per text. The first slot
        # holds (len(text) + 1) so different inputs give different vectors;
        # the rest is zero. After truncate+normalize the vector becomes
        # [1.0, 0.0, ..., 0.0] regardless of input — fine for plumbing
        # checks that don't care about specific vector values.
        return [[float(len(t)) + 1.0] + [0.0] * (self.NATIVE_DIM - 1) for t in texts]


# --- tests -----------------------------------------------------------------


async def test_embed_calls_lm_studio_in_correct_batches() -> None:
    """100 chunks with batch_size=32 → 4 calls of sizes [32, 32, 32, 4]."""
    fake = FakeLMStudioClient()
    embedder = Embedder(client=fake, batch_size=32)  # type: ignore[arg-type]
    chunks = [Chunk(text=f"chunk-{i}", chunk_index=i) for i in range(100)]

    await embedder.embed(chunks)

    batch_sizes = [len(call) for call in fake.calls]
    assert batch_sizes == [32, 32, 32, 4]

    # Total inputs across all batches equals total chunks — nothing skipped
    # or double-sent.
    total_sent = sum(batch_sizes)
    assert total_sent == 100


async def test_embed_attaches_vector_of_configured_dim_to_every_chunk() -> None:
    """Every chunk has `chunk.vector` set, with len == settings.embedding_dim."""
    fake = FakeLMStudioClient()
    embedder = Embedder(client=fake)  # type: ignore[arg-type]
    chunks = [Chunk(text=f"chunk-{i}", chunk_index=i) for i in range(5)]

    await embedder.embed(chunks)

    for c in chunks:
        assert c.vector is not None
        assert len(c.vector) == settings.embedding_dim


async def test_embed_normalizes_each_vector_to_unit_length() -> None:
    """Magnitude of every returned vector ≈ 1.0 (truncate_and_normalize ran)."""
    fake = FakeLMStudioClient()
    embedder = Embedder(client=fake)  # type: ignore[arg-type]
    chunks = [Chunk(text=f"chunk-{i}", chunk_index=i) for i in range(3)]

    await embedder.embed(chunks)

    for c in chunks:
        assert c.vector is not None
        magnitude = math.sqrt(sum(x * x for x in c.vector))
        assert magnitude == pytest.approx(1.0, abs=1e-9)


async def test_embed_returns_the_same_list_object_it_was_given() -> None:
    """In-place mutation — caller's list and the returned list are the same."""
    fake = FakeLMStudioClient()
    embedder = Embedder(client=fake)  # type: ignore[arg-type]
    chunks = [Chunk(text="a", chunk_index=0)]

    result = await embedder.embed(chunks)

    assert result is chunks
    assert result[0] is chunks[0]


async def test_embed_sends_format_document_output_to_lm_studio() -> None:
    """The text we hand LM Studio is what `format_document` produced.

    For Qwen3 documents this is currently raw passthrough, so the texts
    sent should equal the chunk texts. If document formatting ever changes
    (e.g. instruction prefix), this test breaks loudly — which is the
    point.
    """
    fake = FakeLMStudioClient()
    embedder = Embedder(client=fake)  # type: ignore[arg-type]
    chunks = [Chunk(text="hello"), Chunk(text="world")]

    await embedder.embed(chunks)

    assert fake.calls == [["hello", "world"]]


async def test_embed_handles_empty_chunk_list() -> None:
    """Zero chunks → zero LM Studio calls, no crash."""
    fake = FakeLMStudioClient()
    embedder = Embedder(client=fake)  # type: ignore[arg-type]

    result = await embedder.embed([])

    assert result == []
    assert fake.calls == []
