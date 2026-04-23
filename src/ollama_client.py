"""Thin async client for the Ollama /api/chat endpoint."""

import httpx


class OllamaClient:
    def __init__(self, host: str, model: str, timeout: float = 120.0):
        self._host = host.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(timeout=timeout)

    async def chat(self, messages: list[dict]) -> str:
        resp = await self._client.post(
            f"{self._host}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    async def close(self) -> None:
        await self._client.aclose()
