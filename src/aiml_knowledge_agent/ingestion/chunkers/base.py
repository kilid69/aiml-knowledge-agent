"""Chunker interface — every chunking strategy implements this.

A "chunker" splits one document into many smaller pieces (chunks) that fit
inside the embedder's context and represent one retrievable unit each.
Different strategies (naive character splitting, semantic splitting, header-
aware splitting, ...) all conform to the same `BaseChunker` interface so the
ingestion pipeline doesn't care which one is in use.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """One piece of a document, ready to be embedded and stored.

    `metadata` carries everything we'll later filter on in Qdrant
    (source URL, doc_type, framework, section_title, ...). `chunk_index` is
    the chunk's position within its parent document — useful for debugging
    and for stitching neighbours back together when displaying results.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict) # docs/
    chunk_index: int = 0
    vector: list[float] | None = None


class BaseChunker(ABC):
    """Abstract base — concrete chunkers implement `chunk()`."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        """Split `text` into chunks, attaching `metadata` to each one.

        The same metadata dict is shared across every chunk produced from
        one document — only `chunk_index` differs between siblings.
        """
        ...
