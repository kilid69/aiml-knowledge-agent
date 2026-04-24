"""Top-level conftest — currently intentionally empty.

Service-specific fixtures (LM Studio, Qdrant skip guards, shared clients) live
in `tests/integration/conftest.py` so they only apply to integration tests.
Unit tests have no external dependencies and should never be skipped due to
service availability.
"""
