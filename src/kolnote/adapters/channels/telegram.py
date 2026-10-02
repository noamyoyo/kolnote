"""Telegram channel via the Bot API: long polling in, REST out. No inbound port needed.

Create a bot with @BotFather and pass its token (env `KOLNOTE_CHANNEL__TOKEN`). In groups the
bot only sees voice notes if privacy mode is off (BotFather /setprivacy) or it is an admin.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, Self

from ...models import Incoming, IncomingAudio, UnsupportedMessage

log = logging.getLogger(__name__)

VOICE_KEYS = ("voice", "audio")
USER_CONTENT_KEYS = (
    "text", "photo", "video", "video_note", "document", "sticker", "animation", "location", "contact", "poll",
)
MAX_TEXT = 4096


class TelegramChannel:
    name = "telegram"

    def __init__(
        self,
        token: str,
        *,
        api_base: str = "https://api.telegram.org",
        poll_timeout_s: int = 30,
        typing: bool = True,
    ) -> None:
        import httpx

        if not token:
            raise ValueError("telegram channel requires a bot token")
        self._api = f"{api_base.rstrip('/')}/bot{token}"
        self._files = f"{api_base.rstrip('/')}/file/bot{token}"
        self._poll_timeout_s = poll_timeout_s
        self._typing = typing
        self._client = httpx.AsyncClient(timeout=poll_timeout_s + 15)
        self._offset = 0

    @classmethod
    def from_config(cls, options: dict[str, Any]) -> Self:
        return cls(**options)

    async def _call(self, method: str, **payload: Any) -> Any:
        resp = await self._client.post(f"{self._api}/{method}", json=payload)
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"telegram {method} failed: {data.get('description', resp.status_code)}")
        return data["result"]

    async def _download(self, file_id: str) -> bytes:
        info = await self._call("getFile", file_id=file_id)
        resp = await self._client.get(f"{self._files}/{info['file_path']}")
        resp.raise_for_status()
        return resp.content

    async def _to_incoming(self, update: dict[str, Any]) -> Incoming | None:
        message = update.get("message")
        if not message or (message.get("from") or {}).get("is_bot"):
            return None
        chat_id, message_id = str(message["chat"]["id"]), str(message["message_id"])
        sender = str((message.get("from") or {}).get("id", ""))
        for key in VOICE_KEYS:
            if key in message:
                media = message[key]
                audio = await self._download(media["file_id"])
                if self._typing:
                    try:
                        await self._call("sendChatAction", chat_id=message["chat"]["id"], action="typing")
                    except Exception as exc:
                        log.warning("typing indicator failed: %s", type(exc).__name__)
                return IncomingAudio(
                    channel=self.name,
                    chat_id=chat_id,
                    message_id=message_id,
                    sender_id=sender,
                    audio=audio,
                    mime_type=media.get("mime_type", "audio/ogg"),
                    duration_s=media.get("duration"),
                    meta={"username": (message.get("from") or {}).get("username")},
                )
        for key in USER_CONTENT_KEYS:
            if key in message:
                return UnsupportedMessage(self.name, chat_id, message_id, sender, key)
        return None

    async def messages(self) -> AsyncIterator[Incoming]:
        while True:
            try:
                updates = await self._call(
                    "getUpdates", offset=self._offset, timeout=self._poll_timeout_s, allowed_updates=["message"]
                )
            except Exception as exc:
                # never log the exception text: httpx errors can contain the bot token in the URL
                log.warning("getUpdates failed (%s), retrying in 5s", type(exc).__name__)
                await asyncio.sleep(5)
                continue
            for update in updates:
                self._offset = update["update_id"] + 1
                try:
                    msg = await self._to_incoming(update)
                except Exception as exc:
                    log.warning("could not read update %s (%s)", update.get("update_id"), type(exc).__name__)
                    continue
                if msg:
                    yield msg
            await asyncio.sleep(0)

    async def reply(self, to: Incoming, text: str) -> None:
        for start in range(0, len(text) or 1, MAX_TEXT):
            await self._call(
                "sendMessage",
                chat_id=int(to.chat_id),
                text=text[start : start + MAX_TEXT],
                reply_parameters={"message_id": int(to.message_id), "allow_sending_without_reply": True},
            )

    async def close(self) -> None:
        await self._client.aclose()
