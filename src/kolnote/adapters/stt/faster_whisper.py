"""Local STT via faster-whisper. Model can be any CTranslate2 model id or path,
e.g. `large-v3-turbo` or `ivrit-ai/whisper-large-v3-turbo-ct2`.
"""

from __future__ import annotations

import asyncio
import io
import time
from typing import Any, Self

from ...models import Transcript


class FasterWhisperSTT:
    name = "faster_whisper"

    def __init__(
        self,
        model: str,
        *,
        device: str = "auto",
        compute_type: str = "default",
        beam_size: int = 5,
        vad_filter: bool = True,
        initial_prompt: str | None = None,
        cpu_threads: int = 0,
        download_root: str | None = None,
    ) -> None:
        self.model_id = model
        self.name = f"faster_whisper:{model}"
        self._opts = dict(device=device, compute_type=compute_type, cpu_threads=cpu_threads, download_root=download_root)
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._initial_prompt = initial_prompt
        self._model = None
        self._lock = asyncio.Lock()

    @classmethod
    def from_config(cls, options: dict[str, Any]) -> Self:
        return cls(**options)

    def _load(self):
        from faster_whisper import WhisperModel

        return WhisperModel(self.model_id, **self._opts)

    def _run(self, audio: bytes, language: str | None) -> tuple[str, str | None, float]:
        segments, info = self._model.transcribe(
            io.BytesIO(audio),
            language=language,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
            initial_prompt=self._initial_prompt,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, info.language, info.duration

    async def transcribe(self, audio: bytes, *, mime_type: str, language: str | None) -> Transcript:
        async with self._lock:  # one inference at a time; model is not re-entrant
            if self._model is None:
                self._model = await asyncio.to_thread(self._load)
            started = time.perf_counter()
            text, lang, duration = await asyncio.to_thread(self._run, audio, language)
        return Transcript(
            text=text,
            engine=self.name,
            processing_s=time.perf_counter() - started,
            language=lang,
            duration_s=duration,
        )

    async def close(self) -> None:
        self._model = None
