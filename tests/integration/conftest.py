"""Shared fixtures for integration tests.

Integration tests hit real services (LM Studio, Qdrant). They should skip
gracefully — not fail — when those services are unreachable, so `pytest`
works locally whether you have your stack running or not.

Being inside `tests/integration/` (not `tests/conftest.py`) scopes the autouse
skip fixtures to integration tests only — unit tests never touch these services
and must never skip because a service is down.
"""

from collections.abc import AsyncIterator, Iterator
from functools import lru_cache

import httpx
import pytest

from aiml_knowledge_agent.api.config import settings
from aiml_knowledge_agent.models.llm_client import LMStudioClient


# --- reachability probes (cached per session) -----------------------------


@lru_cache(maxsize=1)
def _lm_studio_reachable() -> bool:
    try:
        r = httpx.get(f"{settings.lm_studio_url}/models", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


@lru_cache(maxsize=1)
def _qdrant_reachable() -> bool:
    try:
        r = httpx.get(f"http://{settings.qdrant_host}:{settings.qdrant_port}/", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


# --- autouse skip fixtures -------------------------------------------------


@pytest.fixture(autouse=True)
def require_lm_studio() -> None:
    """Skip every integration test with a clear reason when LM Studio is down."""
    if not _lm_studio_reachable():
        pytest.skip(f"LM Studio not reachable at {settings.lm_studio_url}")


@pytest.fixture(autouse=True)
def require_qdrant() -> None:
    """Skip every integration test with a clear reason when Qdrant is down."""
    if not _qdrant_reachable():
        pytest.skip(f"Qdrant not reachable at {settings.qdrant_host}:{settings.qdrant_port}")


# --- reusable fixtures -----------------------------------------------------


@pytest.fixture
async def lm_client() -> AsyncIterator[LMStudioClient]:
    """Provides a ready-to-use LMStudioClient, closed after the test."""
    client = LMStudioClient()
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def isolated_collection(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Point vector_store functions at a throwaway Qdrant collection.

    Replaces `settings.collection_name` for the test's duration, then drops
    the temp collection afterwards. The real production collection is never
    touched, and nothing leaks between tests.
    """
    from aiml_knowledge_agent.ingestion.vector_store import get_client

    test_name = f"test_{__import__('uuid').uuid4().hex[:8]}"
    monkeypatch.setattr(settings, "collection_name", test_name)

    yield test_name

    # Teardown — runs even if the test raised.
    client = get_client()
    if client.collection_exists(test_name):
        client.delete_collection(test_name)
