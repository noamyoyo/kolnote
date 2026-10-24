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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Self

from ...audio import decode_16k_mono, probe_duration_s
from ...dialogue import MergeOptions, build_dialogue
from ...diarize import SherpaDiarizer, Word
from ...models import Transcript

log = logging.getLogger("kolnote.faster_whisper")


def join_segments(segments, paragraph_gap_s: float = 0) -> str:
    """Join segment texts with a space, or a blank line where the pause before a segment is at least
    `paragraph_gap_s` (0 disables paragraphs)."""
    parts: list[str] = []
    previous_end = None
    for s in segments:
        text = s.text.strip()
        if not text:
            continue
        if parts:
            gap = s.start - previous_end if paragraph_gap_s and previous_end is not None else 0
            parts.append("\n\n" if paragraph_gap_s and gap >= paragraph_gap_s else " ")
        parts.append(text)
        previous_end = s.end if paragraph_gap_s else None
    return "".join(parts)


def _speaker_labels(value) -> list[str]:
    """`speaker_labels` must be a list of distinct, non-empty names (a bare string would be split into letters)."""
    if not isinstance(value, (list, tuple)) or not all(isinstance(x, str) and x.strip() for x in value):
        raise ValueError("diarize.speaker_labels must be a list of non-empty strings")
    labels = [x.strip() for x in value]
    if len(set(labels)) != len(labels):
        raise ValueError("diarize.speaker_labels must not repeat a name")
    return labels


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
        condition_on_previous_text: bool | None = None,
        paragraph_gap_s: float = 0,
        diarize: dict[str, Any] | None = None,
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
        self._condition_on_previous_text = condition_on_previous_text
        self._paragraph_gap_s = paragraph_gap_s
        # `diarize` = {segmentation, embedding, num_speakers, threshold, threads, min_duration_s,
        # max_duration_s, min_speaker_share, speaker_labels, merge}: label speakers (A, B) in recordings between min and max
        # duration, when the audio really holds a conversation. Speakers owning less than
        # min_speaker_share of the words are noise and dropped; fewer than two speakers left means plain
        # text (a voice note by one person). Needs the `diarize` extra.
        self._diarizer: SherpaDiarizer | None = None
        self._diarize_min_s = 120.0
        self._diarize_max_s = 3600.0
        self._min_speaker_share = 0.05
        self._speaker_labels: list[str] = []
        self._merge = MergeOptions()
        if diarize:
            options = dict(diarize)
            self._diarize_min_s = float(options.pop("min_duration_s", 120))
            self._diarize_max_s = float(options.pop("max_duration_s", 3600))
            self._min_speaker_share = float(options.pop("min_speaker_share", 0.05))
            self._speaker_labels = _speaker_labels(options.pop("speaker_labels", []))
            try:
                self._merge = MergeOptions(**options.pop("merge", {}))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"diarize.merge: {exc}") from exc
            self._diarizer = SherpaDiarizer(**options)
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

    def _diarize(self, audio: bytes):
        return self._diarizer.diarize(decode_16k_mono(audio))

    def _run(self, audio: bytes, language: str | None) -> tuple[str, str | None, float]:
        extra = {}
        if self._condition_on_previous_text is not None:
            extra["condition_on_previous_text"] = self._condition_on_previous_text
        labelled = False
        if self._diarizer is not None:
            labelled = self._diarize_min_s <= (probe_duration_s(audio) or 0) <= self._diarize_max_s
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            # the CPU diarization runs while the GPU transcribes
            turns_future = pool.submit(self._diarize, audio) if labelled else None
            segments, info = self._model.transcribe(
                io.BytesIO(audio),
                language=language,
                beam_size=self._beam_size,
                vad_filter=self._vad_filter,
                initial_prompt=self._initial_prompt,
                word_timestamps=labelled,
                **extra,
            )
            segments = list(segments)
            turns = None
            if turns_future is not None:
                try:
                    turns = turns_future.result()
                except Exception:
                    log.exception("diarization failed, returning the transcript without speaker labels")
        except BaseException:
            # do not hold the engine lock (and the error reply) until the CPU diarization is done
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown()
        if turns:
            words = [Word(w.start, w.end, w.word) for s in segments for w in (s.words or [])]
            dialogue = None
            try:
                dialogue = build_dialogue(
                    words,
                    turns,
                    min_share=self._min_speaker_share,
                    paragraph_gap_s=self._paragraph_gap_s,
                    labels=self._speaker_labels,
                    options=self._merge,
                )
            except Exception:
                log.exception("speaker merge failed, returning the transcript without speaker labels")
            if dialogue:
                return dialogue, info.language, info.duration
        return join_segments(segments, self._paragraph_gap_s), info.language, info.duration

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
