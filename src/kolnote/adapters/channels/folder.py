"""Channel that reads audio files from a folder and writes `<name>.txt` next to each.

Useful for demos, batch jobs, and as the reference implementation of the Channel port.
"""

from __future__ import annotations

import asyncio
import mimetypes
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Self

from ...models import AUDIO_EXTENSIONS, IncomingAudio


class FolderChannel:
    name = "folder"

    def __init__(self, inbox: Path, *, poll_s: float = 2.0, once: bool = False) -> None:
        self.inbox = inbox
        self.poll_s = poll_s
        self.once = once

    @classmethod
    def from_config(cls, options: dict[str, Any]) -> Self:
        return cls(Path(options["inbox"]), poll_s=float(options.get("poll_s", 2.0)), once=bool(options.get("once", False)))

    def _pending(self) -> list[Path]:
        return [
            p
            for p in sorted(self.inbox.iterdir())
            if p.suffix.lower() in AUDIO_EXTENSIONS and not p.with_suffix(".txt").exists()
        ]

    async def messages(self) -> AsyncIterator[IncomingAudio]:
        seen: set[Path] = set()
        while True:
            for path in self._pending():
                if path in seen:
                    continue
                seen.add(path)
                yield IncomingAudio(
                    channel=self.name,
                    chat_id=str(self.inbox),
                    message_id=path.name,
                    sender_id="local",
                    audio=path.read_bytes(),
                    mime_type=mimetypes.guess_type(path.name)[0] or "audio/ogg",
                )
            if self.once:
                return
            await asyncio.sleep(self.poll_s)

    async def reply(self, to: IncomingAudio, text: str) -> None:
        (self.inbox / to.message_id).with_suffix(".txt").write_text(text + "\n", encoding="utf-8")

    async def close(self) -> None:
        return None
