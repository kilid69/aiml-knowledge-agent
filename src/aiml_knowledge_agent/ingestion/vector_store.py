"""Qdrant collection management and basic CRUD helpers.

This module is the single place that knows about our Qdrant schema: vector
dimension, distance metric, and the set of payload indices used for filtering.
Everything else in the ingestion and retrieval paths goes through the helpers
defined here.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from aiml_knowledge_agent.api.config import settings


# The payload fields we filter on. Indexing them lets Qdrant answer filtered
# queries (e.g. "only langgraph changelogs") without scanning every point.
# Unindexed fields are still stored, just not fast to filter by.
_INDEXED_PAYLOAD_FIELDS: dict[str, PayloadSchemaType] = {
    "source": PayloadSchemaType.KEYWORD,         # exact-match URL or identifier
    "doc_type": PayloadSchemaType.KEYWORD,       # "markdown" | "html" | "changelog" | ...
    "framework": PayloadSchemaType.KEYWORD,      # "langgraph" | "qdrant" | ...
    "date_ingested": PayloadSchemaType.DATETIME, # ISO timestamp, supports range filters
    "section_title": PayloadSchemaType.KEYWORD,  # heading the chunk came from
}


# --- client singleton ------------------------------------------------------

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Return a shared QdrantClient, creating it on first call.

    Holding one client saves on TCP setup cost and keeps connection state in
    one place. In a larger app we'd manage this via FastAPI lifespan hooks;
    module-level is fine for now.
    """
    global _client
    if _client is None:
        _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return _client


# --- schema ----------------------------------------------------------------


def init_collection() -> None:
    """Create the collection and its payload indices if they don't exist.

    Idempotent — safe to call on every startup. If the collection is already
    there with the right shape, this is a no-op.
    """
    client = get_client()
    name = settings.collection_name

    if client.collection_exists(name):
        return

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=settings.embedding_dim,
            distance=Distance.COSINE,
        ),
    )
    for field_name, schema in _INDEXED_PAYLOAD_FIELDS.items():
        client.create_payload_index(
            collection_name=name,
            field_name=field_name,
            field_schema=schema,
        )


# --- CRUD ------------------------------------------------------------------


def upsert_points(points: list[PointStruct]) -> None:
    """Insert-or-update a batch of points in one call.

    "Upsert" = "update if exists, insert otherwise" — matched by point id.
    Qdrant handles both cases atomically; no need to check existence first.
    """
    client = get_client()
    client.upsert(collection_name=settings.collection_name, points=points)


def delete_by_source(source: str) -> None:
    """Delete every point whose payload.source matches the given value.

    Used when re-ingesting a document: clear the old chunks first, then
    upsert fresh ones. Alternative would be to upsert with stable ids and
    rely on the 'update' half of upsert — but stable ids require stable
    chunking, which isn't guaranteed when chunking strategy changes.
    """
    client = get_client()

    selector = Filter(
        must=[FieldCondition(key="source", match=MatchValue(value=source))],
    )
    client.delete(
        collection_name=settings.collection_name,
        points_selector=selector,
    )
