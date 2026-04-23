"""Shared fixtures for integration tests.

Integration tests hit real services (LM Studio, Qdrant). They should skip
gracefully — not fail — when those services are unreachable, so `pytest`
works locally whether you have your stack running or not.
"""

from collections.abc import AsyncIterator
from functools import lru_cache

import httpx
import pytest

from aiml_knowledge_agent.api.config import settings
from aiml_knowledge_agent.models.llm_client import LMStudioClient


@lru_cache(maxsize=1)
def _lm_studio_reachable() -> bool:
    """Check once per test session whether LM Studio is up.

    lru_cache makes repeat calls free — the actual HTTP probe runs once.
    """
    try:
        r = httpx.get(f"{settings.lm_studio_url}/models", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(autouse=True)
def require_lm_studio() -> None:
    """Auto-applied to every test in this directory.

    If LM Studio is not reachable, skip the test with a clear reason instead
    of letting it fail with a cryptic ConnectionError deep inside httpx.
    """
    if not _lm_studio_reachable():
        pytest.skip(f"LM Studio not reachable at {settings.lm_studio_url}")


@pytest.fixture
async def lm_client() -> AsyncIterator[LMStudioClient]:
    """Provides a ready-to-use LMStudioClient, closed after the test."""

    client = LMStudioClient()
    try:
        yield client
    finally:
        await client.aclose()
