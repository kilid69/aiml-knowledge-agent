from collections.abc import AsyncIterator

import httpx
import json

from aiml_knowledge_agent.api.config import settings


class LMStudioClient:
    """Async client for LM Studio's OpenAI-compatible HTTP API.

    LM Studio exposes three relevant endpoints (all under the configured base URL):
      POST /chat/completions   — chat / generation (supports SSE streaming)
      POST /embeddings         — embeddings (Nomic in our setup)
      GET  /models             — list loaded models (used by /health)

    Design notes:
      - One shared httpx.AsyncClient per LMStudioClient instance. Reusing the
        client enables HTTP connection pooling instead of opening a new TCP
        connection per request.
      - Lifecycle: create the client once (e.g. at app startup), call aclose()
        at shutdown. Don't instantiate per request.
    """

    def __init__(
        self,
        base_url: str | None = None,
        chat_model: str | None = None,
        embed_model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        # Fall back to settings when the caller doesn't override. Lets tests
        # inject their own values without monkeypatching the settings module.
        self.base_url = (base_url or settings.lm_studio_url).rstrip("/")
        self.chat_model = chat_model or settings.lm_studio_chat_model
        self.embed_model = embed_model or settings.lm_studio_embed_model

        # Long default timeout — chat completions can take tens of seconds on
        # a local model. The caller overrides for short probes (e.g. /health).
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def aclose(self) -> None:
        """Close the underlying HTTP client. Call once at shutdown."""
        await self._client.aclose()

    # --- public API ------------------------------------------------------

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings. Returns one vector per input.

        Nomic expects a task prefix on each input ("search_document: ..." for
        ingestion, "search_query: ..." for queries). Add the prefix at the
        call site, not here - this method stays generic.
        """

        response = await self._client.post(
            "/embeddings", json={"model": self.embed_model, "input": texts}
        )

        response.raise_for_status()
        data = response.json()
        #  response shape: {"data": [{"embedding": [0.013, ...]}, {"embedding": [0.019, ...]}]}
        #  extract each item's "embedding" in order and return list[list[float]]
        return [item["embedding"] for item in data["data"]]

    async def chat(
        self,
        messages: list[dict],
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """Run a chat completion.

        Returns a full string when stream=False, or an async iterator yielding
        token chunks when stream=True. The two cases have different shapes
        because you consume them differently ("await" vs "async for").
        """
        if stream:
            return self._chat_stream(messages)
        return await self._chat_once(messages)

    # --- internals -------------------------------------------------------

    async def _chat_once(self, messages: list[dict]) -> str:
        """Non-streaming path — one request, one response, one string."""

        response = await self._client.post(
            "/chat/completions", json={"model": self.chat_model, "messages": messages}
        )
        #  response shape: {"choices": [{"message": {"content": "..."}, ...}], ...}
        response.raise_for_status()
        # first call .json() and then subscript the dict.
        # Because the httpx used directly and not through OpenAI SDK.
        return response.json()["choices"][0]["message"]["content"]

    async def _chat_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Streaming path — yields content deltas as they arrive.

        LM Studio streams responses as Server-Sent Events (SSE). Each line is:
          data: {"choices":[{"delta":{"content":"..."}}], ...}
          ...
          data: [DONE]

          - Each data line after the prefix is JSON; parse with json.loads.
          - Yield choices[0].delta.content when present (some frames have no content).
        """
        payload = {"model": self.chat_model, "messages": messages, "stream": True}

        async with self._client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: ") :]
                if data_str == "[DONE]":
                    break
                # parse to json and use .get() to return None instead of raising.
                # stream sometimes only have role: data: {"choices":[{"delta":{"role":"assistant"}}]}
                content = json.loads(data_str)["choices"][0]["delta"].get("content")
                if content:
                    yield content
