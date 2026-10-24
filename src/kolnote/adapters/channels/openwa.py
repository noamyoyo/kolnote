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
import mimetypes
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from hashlib import sha256
from pathlib import PurePath
from typing import Any, Self
from urllib.parse import quote

from ...models import AUDIO_EXTENSIONS, Incoming, IncomingAudio, UnsupportedMessage

log = logging.getLogger(__name__)

VOICE_TYPES = {"voice", "audio", "ptt"}
USER_CONTENT_TYPES = {"text", "chat", "image", "video", "document", "sticker", "location", "contact", "poll"}
MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_TEXT = 4000  # OpenWA rejects send-text above 4096 characters
SEEN_LIMIT = 1024


MEDIA_KEYS = ("mimetype", "filename", "data", "omitted", "duration")
# .mp4 and .webm are usually video: only trust them when the mimetype says audio
# Generic or non-audio types that phones attach to audio files; the extension then decides.
UNTYPED_MIMES = ("", "application/octet-stream", "application/ogg", "application/x-ogg")
DOCUMENT_AUDIO_EXTENSIONS = AUDIO_EXTENSIONS - {".mp4", ".webm"}
EXTENSION_MIMES = {".m4a": "audio/mp4"}


def _media_info(data: dict[str, Any]) -> dict[str, Any]:
    """Media fields merged from every place OpenWA may put them; later sources win."""
    metadata = data.get("metadata")
    sources = [data, metadata, metadata.get("media") if isinstance(metadata, dict) else None, data.get("media")]
    info: dict[str, Any] = {}
    for source in sources:
        if isinstance(source, dict):
            info.update({k: source[k] for k in MEDIA_KEYS if source.get(k) not in (None, "")})
    return info


def _duration(value: Any) -> float | None:
    """Seconds as a float. Gateways send numbers or numeric strings; anything else is unknown."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if 0 < seconds < float("inf") else None


def _split_text(text: str, limit: int = MAX_TEXT) -> list[str]:
    """Split at a line break or space near the limit, so a long transcript is not cut mid-word."""
    parts: list[str] = []
    while len(text) > limit:
        # prefer a blank line (a paragraph or speaker block), then a line break, then a space
        cut = next(
            (c for c in (text.rfind("\n\n", 0, limit), text.rfind("\n", 0, limit), text.rfind(" ", 0, limit)) if c >= limit // 2),
            limit,
        )
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    return [p for p in (*parts, text) if p.strip()] or [text]


def _clean_mime(mime: Any) -> str:
    return str(mime or "").split(";")[0].strip().lower()


def _resolve_mime(mime: Any, filename: Any) -> str:
    """Keep a real audio/video type; otherwise derive it from the filename, else assume ogg (voice notes)."""
    mime = _clean_mime(mime)
    if mime.startswith(("audio/", "video/")):
        return mime
    suffix = PurePath(str(filename or "")).suffix.lower()
    return mimetypes.guess_type(f"x{suffix}")[0] or EXTENSION_MIMES.get(suffix) or "audio/ogg"


def _is_audio(data: dict[str, Any]) -> bool:
    """Voice notes, plus audio files attached as documents (gallery or file picker, e.g. call recordings)."""
    kind = data.get("type")
    if kind in VOICE_TYPES:
        return True
    if kind != "document":
        return False
    media = _media_info(data)
    mime = _clean_mime(media.get("mimetype"))
    if mime.startswith("audio/"):
        return True
    suffix = PurePath(str(media.get("filename") or "")).suffix.lower()
    return mime in UNTYPED_MIMES and suffix in DOCUMENT_AUDIO_EXTENSIONS


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
        part_text: str = "Part {n} of {total}",
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
        try:
            header = part_text.format(n=99, total=99)  # fail at start, not after a long transcription
        except (KeyError, IndexError, ValueError, AttributeError) as exc:
            raise ValueError(f"channel option part_text is not a valid template ({exc!r}); use {{n}} and {{total}}") from exc
        if len(header) > 64:
            raise ValueError("channel option part_text is too long (64 characters at most)")
        self._part_text = part_text
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
                if status == 413:
                    log.warning("webhook body too large: %d bytes (limit %d), message dropped", length, MAX_BODY_BYTES)
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
        media = _media_info(data)
        mime = _resolve_mime(media.get("mimetype"), media.get("filename"))
        if isinstance(media.get("data"), str) and not media.get("omitted"):
            return base64.b64decode(media["data"]), mime
        url = f"{self._base}/messages/{quote(data['chatId'], safe='')}/{quote(data['id'], safe='')}/media"
        resp = await self._client.get(url)
        resp.raise_for_status()
        return resp.content, _resolve_mime(resp.headers.get("content-type") or mime, media.get("filename"))

    async def messages(self) -> AsyncIterator[Incoming]:
        self._server = await asyncio.start_server(self._handle_http, self._host, self._port)
        log.info("openwa webhook listening on %s:%s%s", self._host, self._port, self._path)
        while True:
            data, received_at = await self._queue.get()
            sender = data.get("senderPhone") or data.get("author") or data.get("from") or ""
            if not _is_audio(data):
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
                duration_s=_duration(_media_info(data).get("duration")),
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
        parts = _split_text(text)
        for n, part in enumerate(parts, 1):
            if len(parts) > 1 and self._part_text:
                # said in the message itself, so several messages read as one transcript
                part = f"{self._part_text.format(n=n, total=len(parts))}\n\n{part}"
            resp = await self._client.post(
                f"{self._base}/messages/send-text",
                json={"chatId": to.chat_id, "text": part},
            )
            resp.raise_for_status()

    async def close(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        await self._client.aclose()
