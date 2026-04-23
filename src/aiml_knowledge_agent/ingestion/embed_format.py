"""Each Embedding model has its own structure and prefix.

use this file as the central module to implement the change.
"""


def format_query(
    query: str, task: str = "Given a question, retrieve relevant passages that answer it"
) -> str:
    """This format is for text-embedding-qwen3-embedding-4b"""
    return f"Instruct: {task}\nQuery: {query}"


def format_document(document: str) -> str:
    """This format is for text-embedding-qwen3-embedding-4b"""
    return document
