"""Unit tests for the embed_format helpers — pure logic, no external services."""

import math

from aiml_knowledge_agent.ingestion.embed_format import truncate_and_normalize


def test_truncate_and_normalize_shrinks_to_target_dim() -> None:
    """Slicing behaviour — output length must equal target_dim."""
    vec = [1.0, 2.0, 3.0, 4.0]
    result = truncate_and_normalize(vec, target_dim=2)
    assert len(result) == 2


def test_truncate_and_normalize_produces_unit_length() -> None:
    """After truncation the vector must be re-normalized to Euclidean length 1.

    Truncating [3, 4, 99, 99] to 2 dims gives [3, 4] — a classic 3-4-5 triangle
    with length 5. After normalization each element is divided by 5, giving
    [0.6, 0.8], which has length sqrt(0.36 + 0.64) == 1.
    """
    vec = [3.0, 4.0, 99.0, 99.0]
    result = truncate_and_normalize(vec, target_dim=2)
    length = math.sqrt(sum(x * x for x in result))
    assert math.isclose(length, 1.0, abs_tol=1e-9)


def test_truncate_and_normalize_handles_zero_vector() -> None:
    """A zero vector has no direction — return it as-is, don't divide by zero."""
    vec = [0.0, 0.0, 0.0, 0.0]
    result = truncate_and_normalize(vec, target_dim=2)
    assert result == [0.0, 0.0]
