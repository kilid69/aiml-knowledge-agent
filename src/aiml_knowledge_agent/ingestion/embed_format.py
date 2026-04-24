"""Each Embedding model has its own structure and prefix.

use this file as the central module to implement the change.
"""
import math

def format_query(
    query: str, task: str = "Given a question, retrieve relevant passages that answer it"
) -> str:
    """This format is for text-embedding-qwen3-embedding-4b"""
    return f"Instruct: {task}\nQuery: {query}"


def format_document(document: str) -> str:
    """This format is for text-embedding-qwen3-embedding-4b"""
    return document


def truncate_and_normalize(vec: list[float], target_dim: int) -> list[float]:
    """Cut a Qwen3 embedding down to `target_dim` and re-normalize to unit length.

    Qwen3 Embedding is Matryoshka-trained, so `vec[:target_dim]` is a valid
    shorter embedding. Cosine similarity assumes unit-length vectors — so after
    truncating we must divide by the new Euclidean length.
    """
    truncated = vec[:target_dim]
    norm = math.sqrt(sum(x * x for x in truncated))
    if norm == 0:
        return truncated
    return [x / norm for x in truncated]
