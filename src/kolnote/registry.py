"""Name -> adapter class lookup. Adapters import lazily so optional deps stay optional.

Lookup order for `type`: built-in name, then an installed package's entry point
(groups `kolnote.channels` / `kolnote.stt`), then a `package.module:Class` path.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points
from typing import Any

from .ports import Channel, STTEngine

CHANNELS: dict[str, str] = {
    "folder": "kolnote.adapters.channels.folder:FolderChannel",
    "openwa": "kolnote.adapters.channels.openwa:OpenWAChannel",
    "telegram": "kolnote.adapters.channels.telegram:TelegramChannel",
}

STT_ENGINES: dict[str, str] = {
    "faster_whisper": "kolnote.adapters.stt.faster_whisper:FasterWhisperSTT",
    "openai_compat": "kolnote.adapters.stt.openai_compat:OpenAICompatSTT",
}


def _plugins(group: str) -> dict[str, str]:
    return {ep.name: ep.value for ep in entry_points(group=group)}


def _load(table: dict[str, str], spec: dict[str, Any], kind: str, group: str) -> Any:
    options = dict(spec)
    try:
        type_name = options.pop("type")
    except KeyError:
        raise ValueError(f"{kind} config is missing 'type'") from None
    target = table.get(type_name) or _plugins(group).get(type_name) or type_name
    module_name, sep, attr = target.partition(":")
    if not sep:
        known = ", ".join(sorted({**table, **_plugins(group)}))
        raise ValueError(f"unknown {kind} type {type_name!r} (available: {known})")
    cls = getattr(import_module(module_name), attr)
    return cls.from_config(options)


def build_channel(spec: dict[str, Any]) -> Channel:
    return _load(CHANNELS, spec, "channel", "kolnote.channels")


def build_stt(spec: dict[str, Any]) -> STTEngine:
    return _load(STT_ENGINES, spec, "stt", "kolnote.stt")
