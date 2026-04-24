"""Integration test for vector_store — hits a real Qdrant instance.

One end-to-end flow test exercises every function in the module plus the
interactions between them (upsert something → retrieve it → delete it →
verify it's gone). Running against a throwaway collection keeps the real
production collection untouched.
"""

from qdrant_client.models import PointStruct

from aiml_knowledge_agent.api.config import settings
from aiml_knowledge_agent.ingestion.vector_store import (
    delete_by_source,
    get_client,
    init_collection,
    upsert_points,
)


def _make_point(point_id: int, source: str, text: str) -> PointStruct:
    """Build a PointStruct with a simple but valid vector for testing.

    The vector content doesn't matter here — we're testing CRUD plumbing, not
    similarity — but the *dimension* must match the collection's configured
    size or Qdrant will reject the upsert.
    """
    vector = [0.01 * point_id] * settings.embedding_dim
    return PointStruct(
        id=point_id,
        vector=vector,
        payload={"source": source, "text": text},
    )


def test_vector_store_lifecycle(isolated_collection: str) -> None:
    """init → upsert → retrieve → delete_by_source → verify removal.

    Single flow test that covers every public function in vector_store.py.
    The `isolated_collection` fixture points settings.collection_name at a
    unique throwaway name and deletes that collection on teardown.
    """
    client = get_client()
    name = settings.collection_name

    # create the collection
    init_collection()
    assert client.collection_exists(name)

    # ...and is idempotent — calling it again with the collection already
    # present must not raise.
    init_collection()

    # Upsert three points: two from source A, one from source B.
    points = [
        _make_point(1, "source://a", "chunk 1 from A"),
        _make_point(2, "source://a", "chunk 2 from A"),
        _make_point(3, "source://b", "chunk 1 from B"),
    ]
    upsert_points(points)

    retrieved = client.retrieve(collection_name=name, ids=[1, 2, 3])
    assert len(retrieved) == 3

    # Delete everything from source A — points 1 and 2 should be gone,
    #    point 3 should remain.
    delete_by_source("source://a")

    remaining = client.retrieve(collection_name=name, ids=[1, 2, 3])
    assert len(remaining) == 1
    assert remaining[0].id == 3
