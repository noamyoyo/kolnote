"""WhatsApp channel via an OpenWA gateway: webhook in, REST out.

OpenWA POSTs `message.received` events to this adapter's webhook (HMAC-SHA256 signed).
Voice notes are taken from the inline base64 media, or downloaded from the gateway
when it was too large to inline. Transcripts are sent back as a plain text reply in
the chat (a chat-restricted key cannot quote).
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from hashlib import sha256
from typing import Any, Self
from urllib.parse import quote

from ...models import Incoming, IncomingAudio, UnsupportedMessage

log = logging.getLogger(__name__)

VOICE_TYPES = {"voice", "audio", "ptt"}
USER_CONTENT_TYPES = {"text", "chat", "image", "video", "document", "sticker", "location", "contact", "poll"}
MAX_BODY_BYTES = 32 * 1024 * 1024
SEEN_LIMIT = 1024


class OpenWAChannel:
    name = "openwa"

    def __init__(
        self,
        base_url: str,
        session_id: str,
        api_key: str,
        webhook_secret: str,
        *,
        host: str = "0.0.0.0",
        port: int = 8765,
        path: str = "/webhook",
        timeout_s: float = 30.0,
        mark_read: bool = True,
    ) -> None:
        import httpx

        if not webhook_secret:
            raise ValueError("openwa channel requires webhook_secret (unsigned webhooks are rejected)")
        self._base = f"{base_url.rstrip('/')}/api/sessions/{session_id}"
        self._secret = webhook_secret.encode()
        self._host, self._port, self._path = host, port, path
        self._client = httpx.AsyncClient(headers={"X-API-Key": api_key}, timeout=timeout_s)
        self._queue: asyncio.Queue[tuple[dict[str, Any], float]] = asyncio.Queue()
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._server: asyncio.AbstractServer | None = None
        self._mark_read = mark_read
        self._tasks: set[asyncio.Task[None]] = set()

    @classmethod
    def from_config(cls, options: dict[str, Any]) -> Self:
        return cls(**options)

    def _sign(self, body: bytes) -> str:
        return "sha256=" + hmac.new(self._secret, body, sha256).hexdigest()

    def _first_time(self, key: str) -> bool:
        if key in self._seen:
            return False
        self._seen[key] = None
        if len(self._seen) > SEEN_LIMIT:
            self._seen.popitem(last=False)
        return True

    async def _handle_http(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        status = 400
        try:
            request_line = (await asyncio.wait_for(reader.readline(), 10)).decode("latin-1").split()
            headers: dict[str, str] = {}
            while (line := (await asyncio.wait_for(reader.readline(), 10)).decode("latin-1").strip()):
                key, _, value = line.partition(":")
                headers[key.strip().lower()] = value.strip()
            length = int(headers.get("content-length", "0"))
            if len(request_line) < 2 or request_line[0] != "POST" or request_line[1] != self._path:
                status = 404
            elif not 0 < length <= MAX_BODY_BYTES:
                status = 413 if length > MAX_BODY_BYTES else 400
            else:
                body = await asyncio.wait_for(reader.readexactly(length), 30)
                if not hmac.compare_digest(self._sign(body), headers.get("x-openwa-signature", "")):
                    status = 401
                else:
                    status = self._accept(body, headers)
        except (TimeoutError, ValueError, asyncio.IncompleteReadError, ConnectionError):
            status = 400
        except Exception:
            log.exception("webhook handler failed")
            status = 500
        finally:
            try:
                writer.write(f"HTTP/1.1 {status} X\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode())
                await writer.drain()
            except ConnectionError:
                pass
            writer.close()

    def _accept(self, body: bytes, headers: dict[str, str]) -> int:
        try:
            event = json.loads(body)
        except json.JSONDecodeError:
            return 400
        if event.get("event") != "message.received":
            return 200
        data = event.get("data") or {}
        kind = data.get("type")
        if data.get("fromMe") or (kind not in VOICE_TYPES and kind not in USER_CONTENT_TYPES):
            return 200
        key = headers.get("x-openwa-idempotency-key") or str(data.get("id"))
        if self._first_time(key):
            self._queue.put_nowait((data, time.monotonic()))
        return 200

    async def _audio_bytes(self, data: dict[str, Any]) -> tuple[bytes, str]:
        media = data.get("media") or {}
        mime = (media.get("mimetype") or "audio/ogg").split(";")[0]
        if isinstance(media.get("data"), str) and not media.get("omitted"):
            return base64.b64decode(media["data"]), mime
        url = f"{self._base}/messages/{quote(data['chatId'], safe='')}/{quote(data['id'], safe='')}/media"
        resp = await self._client.get(url)
        resp.raise_for_status()
        return resp.content, resp.headers.get("content-type", mime).split(";")[0]

    async def messages(self) -> AsyncIterator[Incoming]:
        self._server = await asyncio.start_server(self._handle_http, self._host, self._port)
        log.info("openwa webhook listening on %s:%s%s", self._host, self._port, self._path)
        while True:
            data, received_at = await self._queue.get()
            sender = data.get("senderPhone") or data.get("author") or data.get("from") or ""
            if data.get("type") not in VOICE_TYPES:
                yield UnsupportedMessage(
                    channel=self.name,
                    chat_id=data["chatId"],
                    message_id=data["id"],
                    sender_id=str(sender),
                    kind=str(data.get("type")),
                )
                continue
            try:
                audio, mime = await self._audio_bytes(data)
            except Exception:
                log.exception("could not get audio for message %s", data.get("id"))
                continue
            if self._mark_read:
                task = asyncio.create_task(self._send_read_receipt(data))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            yield IncomingAudio(
                channel=self.name,
                chat_id=data["chatId"],
                message_id=data["id"],
                sender_id=str(sender),
                audio=audio,
                mime_type=mime,
                duration_s=(data.get("media") or {}).get("duration"),
                received_at=received_at,
                meta={"author": data.get("author"), "from": data.get("from"), "type": data.get("type")},
            )

    async def _send_read_receipt(self, data: dict[str, Any]) -> None:
        try:
            resp = await self._client.post(
                f"{self._base}/chats/read", json={"chatId": data["chatId"], "messageIds": [data["id"]]}
            )
            resp.raise_for_status()
        except Exception as exc:
            log.warning("read receipt failed for %s: %s", data.get("id"), exc)

    async def reply(self, to: Incoming, text: str) -> None:
        resp = await self._client.post(
            f"{self._base}/messages/send-text",
            json={"chatId": to.chat_id, "text": text},
        )
        resp.raise_for_status()

    async def close(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        await self._client.aclose()
