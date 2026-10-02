from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .pipeline import LOADING, UNSUPPORTED

ENV_PREFIX = "KOLNOTE_"
LIST_KEYS = {"allow_chats", "allow_senders", "languages"}


@dataclass(frozen=True)
class Settings:
    channel: dict[str, Any] = field(default_factory=dict)
    channels: dict[str, dict[str, Any]] = field(default_factory=dict)
    stt: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    server: dict[str, Any] = field(default_factory=dict)
    wyoming: dict[str, Any] | None = None
    language: str | None = None
    concurrency: int = 1
    unsupported_text: str = UNSUPPORTED
    loading_text: str = LOADING

    def channel_specs(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """(channel spec, policy options) for each configured channel.

        `[channel]` configures one channel. `[channels.<name>]` configures several; `type`
        defaults to the name, and a `policy` sub-table replaces the top-level `[policy]`.
        """
        if self.channel and self.channels:
            raise ValueError("use either [channel] or [channels.<name>], not both")
        if self.channels:
            specs = []
            for name, table in self.channels.items():
                spec = {k: v for k, v in table.items() if k != "policy"}
                spec.setdefault("type", name)
                specs.append((spec, table.get("policy", self.policy)))
            return specs
        return [(self.channel, self.policy)] if self.channel else []


def _parse(key: str, raw: str, existing: Any) -> Any:
    if key in LIST_KEYS:
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(existing, str):
        return raw
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    if raw[:1] in "[{":
        return json.loads(raw)
    return raw


def apply_env(data: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """Override config keys from env: KOLNOTE_STT__MODEL=x sets data['stt']['model'] = 'x'."""
    for name, raw in env.items():
        if not name.startswith(ENV_PREFIX):
            continue
        path = name[len(ENV_PREFIX) :].lower().split("__")
        node = data
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = _parse(path[-1], raw, node.get(path[-1]))
    return data


def load_settings(path: Path | None = None, env: Mapping[str, str] | None = None) -> Settings:
    data = tomllib.loads(path.read_text(encoding="utf-8")) if path else {}
    apply_env(data, os.environ if env is None else env)
    return Settings(
        channel=data.get("channel", {}),
        channels=data.get("channels", {}),
        stt=data.get("stt", {}),
        policy=data.get("policy", {}),
        server=data.get("server", {}),
        wyoming=data.get("wyoming"),
        language=data.get("language"),
        concurrency=int(data.get("concurrency", 1)),
        unsupported_text=str(data.get("unsupported_text", UNSUPPORTED)),
        loading_text=str(data.get("loading_text", LOADING)),
    )
