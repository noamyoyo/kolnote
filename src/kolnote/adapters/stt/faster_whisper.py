"""Local STT via faster-whisper. Model can be any CTranslate2 model id or path,
e.g. `large-v3-turbo` or `ivrit-ai/whisper-large-v3-turbo-ct2`.

`idle_unload_s` frees the model (and its GPU memory) after that many seconds without a request;
the next request loads it again. Use it when the GPU is shared with something else.
"""

from __future__ import annotations

import asyncio
import gc
import io
import logging
import time
from typing import Any, Self

from ...models import Transcript

log = logging.getLogger("kolnote.faster_whisper")


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
        idle_unload_s: float = 0,
    ) -> None:
        self.model_id = model
        self.name = f"faster_whisper:{model}"
        self._opts = dict(device=device, compute_type=compute_type, cpu_threads=cpu_threads, download_root=download_root)
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._initial_prompt = initial_prompt
        self._idle_unload_s = idle_unload_s
        self._model = None
        self._lock = asyncio.Lock()
        self._unload_task: asyncio.Task | None = None

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

    def _arm_unload(self) -> None:
        if not self._idle_unload_s:
            return
        if self._unload_task:
            self._unload_task.cancel()
        self._unload_task = asyncio.get_running_loop().create_task(self._unload_when_idle())

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def needs_load(self, language: str | None = None) -> bool:
        return not self.loaded

    async def unload(self) -> None:
        async with self._lock:
            if self._model is not None:
                self._model = None
                gc.collect()

    async def _unload_when_idle(self) -> None:
        await asyncio.sleep(self._idle_unload_s)
        if self.loaded:
            await self.unload()
            log.info("model unloaded after %.0fs idle", self._idle_unload_s)

    async def preload(self) -> None:
        async with self._lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load)
        self._arm_unload()

    async def transcribe(self, audio: bytes, *, mime_type: str, language: str | None) -> Transcript:
        async with self._lock:  # one inference at a time; model is not re-entrant
            if self._model is None:
                self._model = await asyncio.to_thread(self._load)
            started = time.perf_counter()
            text, lang, duration = await asyncio.to_thread(self._run, audio, language)
        self._arm_unload()
        return Transcript(
            text=text,
            engine=self.name,
            processing_s=time.perf_counter() - started,
            language=lang,
            duration_s=duration,
        )

    async def close(self) -> None:
        if self._unload_task:
            self._unload_task.cancel()
        self._model = None
