"""Naive character-based chunker — our baseline strategy.

Uses LangChain's `RecursiveCharacterTextSplitter`, which tries to split on
the most "natural" boundary it can find (paragraphs first, then sentences,
then words, then individual characters as a last resort) while keeping
chunk size near the target. Good enough as a baseline; later phases will
add smarter, structure-aware chunkers and compare against this one.
"""

from typing import Any

from aiml_knowledge_agent.ingestion.chunkers.base import BaseChunker, Chunk


class SimpleChunker(BaseChunker):
    """Splits text into ~512-char chunks with 50-char overlap.

    Overlap matters: a sentence sitting on the boundary between two chunks
    would otherwise be split in half and likely retrievable from neither.
    A small overlap means the boundary sentence appears (whole) in at least
    one chunk.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        pieces: list[str] = self._splitter.split_text(text)
        return [
            Chunk(text=piece, metadata=dict(metadata), chunk_index=i)
            for i, piece in enumerate(pieces)
        ]

