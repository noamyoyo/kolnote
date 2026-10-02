"""Name -> adapter class lookup. Adapters import lazily so optional deps stay optional.

A `type` that is not a built-in name is treated as a `package.module:Class` path,
so third-party adapters need no registration.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .ports import Channel, STTEngine

CHANNELS: dict[str, str] = {
    "folder": "kolnote.adapters.channels.folder:FolderChannel",
    "openwa": "kolnote.adapters.channels.openwa:OpenWAChannel",
}

STT_ENGINES: dict[str, str] = {
    "faster_whisper": "kolnote.adapters.stt.faster_whisper:FasterWhisperSTT",
    "openai_compat": "kolnote.adapters.stt.openai_compat:OpenAICompatSTT",
}


def _load(table: dict[str, str], spec: dict[str, Any], kind: str) -> Any:
    options = dict(spec)
    try:
        type_name = options.pop("type")
    except KeyError:
        raise ValueError(f"{kind} config is missing 'type'") from None
    target = table.get(type_name, type_name)
    module_name, sep, attr = target.partition(":")
    if not sep:
        known = ", ".join(sorted(table))
        raise ValueError(f"unknown {kind} type {type_name!r} (built-in: {known})")
    cls = getattr(import_module(module_name), attr)
    return cls.from_config(options)


def build_channel(spec: dict[str, Any]) -> Channel:
    return _load(CHANNELS, spec, "channel")


def build_stt(spec: dict[str, Any]) -> STTEngine:
    return _load(STT_ENGINES, spec, "stt")
