"""Integration tests for LMStudioClient — hits a real LM Studio instance."""

from aiml_knowledge_agent.models.llm_client import LMStudioClient
from aiml_knowledge_agent.ingestion.embed_format import format_document


async def test_embed_returns_vectors_of_expected_dim(lm_client: LMStudioClient) -> None:
    """embed() should return one 2560-d vector per input string.

    2560 is the Qwen3-Embedding-4B native output dimension. Truncation to a
    smaller dim (Matryoshka) is the embedder's job, not the client's — so we
    assert the raw client output here.
    """
    raw_texts = ["The sky is blue.", "Today I am going to the supermarket!"]
    formatted_texts = [format_document(text) for text in raw_texts]

    vectors = await lm_client.embed(formatted_texts)

    assert len(vectors) == len(raw_texts)                    # one vector per input
    assert all(len(v) == 2560 for v in vectors)              # native Qwen3 dim
    assert all(isinstance(x, float) for x in vectors[0])     # element type sanity


async def test_chat_returns_non_empty_string(lm_client: LMStudioClient) -> None:
    """chat() in non-stream mode returns the full assistant message as a string."""
    messages = [{"role": "user", "content": "Reply with the single word: pong"}]

    response = await lm_client.chat(messages)

    assert isinstance(response, str)    # shape: it's a string
    assert response                     # truthy: non-empty string


async def test_chat_stream_yields_chunks(lm_client: LMStudioClient) -> None:
    """chat(stream=True) returns an async iterator that yields string chunks."""
    messages = [{"role": "user", "content": "Count from 1 to 5."}]

    chunks: list[str] = []
    stream = await lm_client.chat(messages, stream=True)
    async for chunk in stream:
        assert isinstance(chunk, str)   # shape: each chunk is a string
        chunks.append(chunk)

    assert chunks                       # at least one chunk was yielded
    assert "".join(chunks)              # joining produces a non-empty string
