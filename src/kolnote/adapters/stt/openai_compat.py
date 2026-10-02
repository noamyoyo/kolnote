"""STT via any server speaking OpenAI's /v1/audio/transcriptions (OpenAI, Groq, Speaches, whisper.cpp server...)."""

from __future__ import annotations

import os
import time
from typing import Any, Self

from ...models import Transcript


class OpenAICompatSTT:
    name = "openai_compat"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key_env: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        import httpx

        self.name = f"openai_compat:{model}"
        self._model = model
        headers = {}
        if api_key_env:
            key = os.environ.get(api_key_env)
            if not key:
                raise ValueError(f"environment variable {api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_s)

    @classmethod
    def from_config(cls, options: dict[str, Any]) -> Self:
        return cls(**options)

    async def transcribe(self, audio: bytes, *, mime_type: str, language: str | None) -> Transcript:
        data = {"model": self._model, "response_format": "json"}
        if language:
            data["language"] = language
        started = time.perf_counter()
        resp = await self._client.post(
            "/audio/transcriptions", data=data, files={"file": ("audio", audio, mime_type)}
        )
        resp.raise_for_status()
        return Transcript(
            text=resp.json().get("text", "").strip(),
            engine=self.name,
            processing_s=time.perf_counter() - started,
            language=language,
        )

    async def close(self) -> None:
        await self._client.aclose()
