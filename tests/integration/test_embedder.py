"""Integration test for Embedder — hits a real LM Studio instance.

End-to-end check: real chunks → real embeddings → vectors attached.
The unit test (`tests/unit/test_embedder.py`) covers batching/plumbing
with mocks; this one verifies the full stack actually works against
the live model.
"""

import math

import pytest

from aiml_knowledge_agent.api.config import settings
from aiml_knowledge_agent.ingestion.chunkers.base import Chunk
from aiml_knowledge_agent.ingestion.embedder import Embedder
from aiml_knowledge_agent.models.llm_client import LMStudioClient


async def test_embed_attaches_vectors_of_configured_dim(
    lm_client: LMStudioClient,
) -> None:
    """10 chunks → each gets a vector of settings.embedding_dim, L2-normalized.

    This is the build plan's stated verification step (adapted: 1024-d for
    Qwen3-Embedding-4B Matryoshka-truncated, not 768-d).
    """
    embedder = Embedder(client=lm_client)
    chunks = [
        Chunk(text=f"Sentence number {i}: the sky is blue today.", chunk_index=i)
        for i in range(10)
    ]

    result = await embedder.embed(chunks)

    # Mutated in place — same list object comes back.
    assert result is chunks

    # Every chunk has a vector now, of the configured (truncated) dimension.
    for c in result:
        assert c.vector is not None
        assert len(c.vector) == settings.embedding_dim

        # L2-normalized → magnitude ≈ 1. Tolerance accounts for floating-point.
        magnitude = math.sqrt(sum(x * x for x in c.vector))
        assert magnitude == pytest.approx(1.0, abs=1e-5)


async def test_embed_handles_more_chunks_than_one_batch(
    lm_client: LMStudioClient,
) -> None:
    """50 chunks with default batch_size=32 forces 2 LM Studio round-trips.

    Catches bugs where batching off-by-ones would either skip chunks or
    re-embed them — the kind of failure that wouldn't show up in a
    single-batch test.
    """
    embedder = Embedder(client=lm_client)
    chunks = [Chunk(text=f"Test sentence number {i}.", chunk_index=i) for i in range(50)]

    result = await embedder.embed(chunks)

    assert len(result) == 50
    assert all(c.vector is not None for c in result)
    assert all(len(c.vector) == settings.embedding_dim for c in result)
