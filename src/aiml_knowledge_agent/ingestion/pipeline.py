"""Ingestion pipeline — chunk → embed → upsert.

The glue that ties the three ingestion stages together. One call per
document: hand it the raw text and a metadata dict, and the pipeline
chunks the text, embeds each chunk, and writes the points into Qdrant.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from qdrant_client.models import PointStruct

from aiml_knowledge_agent.ingestion.chunkers.base import BaseChunker, Chunk
from aiml_knowledge_agent.ingestion.embedder import Embedder
from aiml_knowledge_agent.ingestion.vector_store import (
    delete_by_source,
    init_collection,
    upsert_points,
)


class IngestionPipeline:
    """Runs one document through chunker → embedder → vector store.

    Holds chunker and embedder by reference so a single pipeline instance
    can be built once at startup and reused for every ingest call.
    """

    def __init__(self, chunker: BaseChunker, embedder: Embedder) -> None:
        # Get the chunker that inherits from the BaseChunker 
        self._chunker = chunker
        self._embedder = embedder

    async def ingest_document(self, text: str, metadata: dict[str, Any]) -> int:
        """Ingest one document end-to-end. Returns the number of chunks stored."""
        # make sure the collection exists.
        init_collection()

        # real defensive copy — local variable, not stored on self.
        # dict(metadata) builds a NEW dict; the caller's dict is untouched.
        metadata = dict(metadata)
        # stamp date_ingested only if the caller didn't already provide one
        metadata.setdefault("date_ingested", datetime.now(timezone.utc).isoformat())

        # clear any prior version if source is set.
        # .get() returns None when the key is missing, instead of raising KeyError.
        source = metadata.get("source")
        if source:
            delete_by_source(source)

        # chunk, then embed.
        chunks = self._chunker.chunk(text, metadata)
        if not chunks:
            return 0
        chunks = await self._embedder.embed(chunks)

        # one Qdrant write call for the whole document
        upsert_points([_chunk_to_point(c) for c in chunks])

        return len(chunks)


def _chunk_to_point(chunk: Chunk) -> PointStruct:
    """Translate one Chunk into the Qdrant point shape."""
    
    # guard first — refuse to build a point with no vector
    if chunk.vector is None:
        raise ValueError(f"chunk {chunk.chunk_index} has no vector")

    # copy metadata so we don't mutate the chunk's own dict
    payload = dict(chunk.metadata)
    payload["text"] = chunk.text
    payload["chunk_index"] = chunk.chunk_index

    return PointStruct(id=str(uuid4()), vector=chunk.vector, payload=payload)