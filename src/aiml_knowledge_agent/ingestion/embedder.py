"""Embedding service — turns Chunks into Chunks-with-vectors.

Takes the chunks produced by a `BaseChunker`, runs each chunk's text
through LM Studio's embedding model, applies our Matryoshka
truncate-and-normalize step, and returns the same chunks with
`chunk.vector` populated. Batches requests to LM Studio so we don't pay
HTTP overhead per chunk and so the embedding model can process texts in
parallel.
"""

from aiml_knowledge_agent.api.config import settings
from aiml_knowledge_agent.ingestion.chunkers.base import Chunk
from aiml_knowledge_agent.ingestion.embed_format import (
    format_document,
    truncate_and_normalize,
)
from aiml_knowledge_agent.models.llm_client import LMStudioClient


class Embedder:
    """Wraps an LMStudioClient and embeds Chunks in batches.

    Holds the client by reference rather than constructing one — so the
    same shared client can be reused across the pipeline (one TCP
    connection pool for the whole app).
    """

    def __init__(self, client: LMStudioClient, batch_size: int = 32) -> None:
        self._client = client
        self._batch_size = batch_size

    async def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed every chunk's text and attach the vector in-place.

        Returns the same list (with `chunk.vector` now set) so callers can
        use the return value naturally without thinking about whether
        chunks were mutated.
        """
        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i: i + self._batch_size]
            formatted_texts = [format_document(chunk.text) for chunk in batch]
            vectors = await self._client.embed(formatted_texts)

            for chunk, vec in zip(batch, vectors):
                # Mutation works through chunk.vector = ...
                chunk.vector = truncate_and_normalize(vec, settings.embedding_dim)

        return chunks