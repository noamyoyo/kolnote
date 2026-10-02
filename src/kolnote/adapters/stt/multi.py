"""Several STT models behind one engine, routed by language.

    [stt]
    type = "multi"
    default = "en"          # model used when no language matches
    exclusive = true        # optional: keep at most one model in memory

    [stt.models.en]
    type = "faster_whisper"
    model = "large-v3-turbo"
    languages = ["en"]
    keep_warm = true        # loaded at start; in exclusive mode reloaded after another model idles out

    [stt.models.he]
    type = "faster_whisper"
    model = "ivrit-ai/whisper-large-v3-turbo-ct2"
    languages = ["he"]
    idle_unload_s = 300     # loaded on demand, unloaded after 5 idle minutes

Requests are served one at a time. The engines are built by the normal registry, so any engine
type works; models whose engine has no `preload`/`unload` simply stay as they are.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Self

from ...models import Transcript
from ...ports import STTEngine

log = logging.getLogger("kolnote.multi")

MODEL_KEYS = ("languages", "keep_warm", "idle_unload_s")


def _lang(code: str | None) -> str | None:
    return code.lower().replace("_", "-").split("-")[0] if code else None


class MultiSTT:
    name = "multi"

    def __init__(
        self,
        models: dict[str, STTEngine],
        *,
        default: str,
        languages: dict[str, list[str]] | None = None,
        keep_warm: set[str] | None = None,
        idle_unload_s: dict[str, float] | None = None,
        exclusive: bool = False,
    ) -> None:
        if default not in models:
            raise ValueError(f"default model {default!r} is not one of: {', '.join(models)}")
        self._models = models
        self._default = default
        self._languages = {name: [_lang(c) for c in (languages or {}).get(name, [])] for name in models}
        self._keep_warm = keep_warm or set()
        self._idle = idle_unload_s or {}
        self._exclusive = exclusive
        for name in self._keep_warm & {n for n, s in self._idle.items() if s}:
            raise ValueError(f"model {name!r}: keep_warm and idle_unload_s cannot be combined")
        if exclusive and self._keep_warm:
            loose = [n for n in models if n not in self._keep_warm and not self._idle.get(n)]
            if loose:
                raise ValueError(
                    f"exclusive mode: set idle_unload_s on {', '.join(loose)} so the warm model can come back"
                )
        by_language: dict[str, str] = {}
        for name, codes in self._languages.items():
            for code in codes:
                if by_language.setdefault(code, name) != name:
                    raise ValueError(f"language {code!r} is claimed by both {by_language[code]!r} and {name!r}")
        self._by_language = by_language
        self._lock = asyncio.Lock()
        self._timers: dict[str, asyncio.Task] = {}

    @classmethod
    def from_config(cls, options: dict[str, Any]) -> Self:
        from ...registry import build_stt

        options = dict(options)
        specs = options.pop("models", None) or {}
        if not specs:
            raise ValueError("multi stt needs at least one [stt.models.<name>] table")
        default = options.pop("default", None) or next(iter(specs))
        exclusive = bool(options.pop("exclusive", False))
        if options:
            raise ValueError(f"unknown multi stt options: {', '.join(options)}")
        models, languages, keep_warm, idle = {}, {}, set(), {}
        for name, spec in specs.items():
            spec = dict(spec)
            languages[name] = list(spec.pop("languages", []))
            if spec.pop("keep_warm", False):
                keep_warm.add(name)
            idle[name] = float(spec.pop("idle_unload_s", 0))
            models[name] = build_stt(spec)
        return cls(
            models, default=default, languages=languages, keep_warm=keep_warm, idle_unload_s=idle, exclusive=exclusive
        )

    @property
    def models(self) -> dict[str, list[str]]:
        """Model name -> languages it serves (for advertising to clients)."""
        return {name: list(codes) for name, codes in self._languages.items()}

    def route(self, language: str | None) -> str:
        return self._by_language.get(_lang(language) or "", self._default)

    def needs_load(self, language: str | None = None) -> bool:
        engine = self._models[self.route(language)]
        return hasattr(engine, "loaded") and not engine.loaded

    async def warm_up(self) -> None:
        async with self._lock:
            for name in self._models:
                if name in self._keep_warm:
                    await self._load(name)

    async def _load(self, name: str) -> None:
        engine = self._models[name]
        if hasattr(engine, "preload") and not getattr(engine, "loaded", False):
            log.info("loading model %s", name)
            await engine.preload()

    async def _unload(self, name: str) -> None:
        engine = self._models[name]
        if hasattr(engine, "unload") and getattr(engine, "loaded", False):
            await engine.unload()
            log.info("model %s unloaded", name)

    def _arm(self, name: str) -> None:
        self._cancel(name)
        if self._idle.get(name):
            self._timers[name] = asyncio.get_running_loop().create_task(self._idle_unload(name, self._idle[name]))

    def _cancel(self, name: str) -> None:
        timer = self._timers.pop(name, None)
        if timer:
            timer.cancel()

    async def _idle_unload(self, name: str, delay: float) -> None:
        await asyncio.sleep(delay)
        async with self._lock:
            self._timers.pop(name, None)
            await self._unload(name)
            if self._exclusive:
                for warm in self._keep_warm:
                    await self._load(warm)

    async def transcribe(self, audio: bytes, *, mime_type: str, language: str | None) -> Transcript:
        name = self.route(language)
        langs = self._languages[name]
        language = _lang(language) or (langs[0] if len(langs) == 1 else None)
        async with self._lock:
            if self._exclusive:
                for other in self._models:
                    if other != name:
                        self._cancel(other)
                        await self._unload(other)
            transcript = await self._models[name].transcribe(audio, mime_type=mime_type, language=language)
            self._arm(name)
        return transcript

    async def close(self) -> None:
        for name in list(self._timers):
            self._cancel(name)
        for engine in self._models.values():
            await engine.close()
