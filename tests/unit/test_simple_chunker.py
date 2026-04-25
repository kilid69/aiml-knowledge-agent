"""Unit tests for SimpleChunker — pure logic, no external services."""

from aiml_knowledge_agent.ingestion.chunkers.base import Chunk
from aiml_knowledge_agent.ingestion.chunkers.simple import SimpleChunker


def test_splits_long_text_into_multiple_chunks() -> None:
    """A 2000-char string with chunk_size=512 produces several chunks.

    We use a string with no natural separators ("a" * 2000) so the splitter
    is forced into pure character-level splitting — that gives deterministic
    behaviour we can reason about, instead of the splitter trying to break
    on paragraphs/sentences/words.
    """
    chunker = SimpleChunker(chunk_size=512, chunk_overlap=50)

    text = "a" * 2000
    chunks = chunker.chunk(text, metadata={})

    # More than one chunk (otherwise the splitter isn't doing anything),
    # and every chunk fits inside the configured size.
    assert len(chunks) > 1
    assert all(len(c.text) <= 512 for c in chunks)


def test_chunks_have_configured_overlap() -> None:
    """Adjacent chunks share `chunk_overlap` characters.

    The whole point of overlap is that information sitting on a chunk
    boundary appears (intact) in at least one chunk. We verify that by
    checking that the *end* of chunk N equals the *start* of chunk N+1
    for `chunk_overlap` characters.
    """
    chunker = SimpleChunker(chunk_size=100, chunk_overlap=20)

    # Distinct characters per position so we can detect overlap precisely.
    text = "".join(chr(ord("a") + (i % 26)) for i in range(500))
    chunks = chunker.chunk(text, metadata={})

    for prev, curr in zip(chunks, chunks[1:]):
        assert prev.text[-20:] == curr.text[:20], (
            f"expected 20-char overlap between adjacent chunks, "
            f"got prev tail={prev.text[-20:]!r}, curr head={curr.text[:20]!r}"
        )


def test_metadata_is_attached_to_every_chunk() -> None:
    """Every produced Chunk carries the caller's metadata.

    The pipeline relies on this: payload fields like `source` and
    `doc_type` come from the document, get passed in once, and need to
    end up on every chunk for filtered retrieval to work.
    """
    chunker = SimpleChunker(chunk_size=100, chunk_overlap=10)
    metadata = {"source": "https://example.com/doc", "doc_type": "markdown"}

    chunks = chunker.chunk("a" * 500, metadata=metadata)

    assert chunks  # at least one chunk produced
    for c in chunks:
        assert c.metadata == metadata


def test_chunk_index_is_sequential_starting_from_zero() -> None:
    """chunk_index reflects position in the original document."""
    chunker = SimpleChunker(chunk_size=100, chunk_overlap=10)

    chunks = chunker.chunk("a" * 500, metadata={})

    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_metadata_per_chunk_is_independent() -> None:
    """Mutating one chunk's metadata must NOT affect siblings.

    We pay the cost of copying metadata per chunk (`dict(metadata)` in
    SimpleChunker.chunk) precisely so this holds. Without the copy, every
    chunk would share one dict and mutating one would leak everywhere.
    """
    chunker = SimpleChunker(chunk_size=100, chunk_overlap=10)
    metadata = {"source": "shared"}

    chunks = chunker.chunk("a" * 300, metadata=metadata)
    assert len(chunks) >= 2  # need at least two to compare

    chunks[0].metadata["new_field"] = "only_on_first"

    assert "new_field" not in chunks[1].metadata
    assert "new_field" not in metadata  # caller's dict also untouched


def test_returns_list_of_chunks() -> None:
    """Sanity: the return type is `list[Chunk]`, not `list[str]`."""
    chunker = SimpleChunker()

    chunks = chunker.chunk("hello world", metadata={"k": "v"})

    assert all(isinstance(c, Chunk) for c in chunks)
