import asyncio
import base64
import hashlib
import hmac
import json
import socket

import httpx

from kolnote.adapters.channels.openwa import OpenWAChannel
from kolnote.models import UnsupportedMessage
from kolnote.registry import build_channel

SECRET = "s3cret"
CHAT = "120363000000000000@g.us"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_channel(handler=None) -> OpenWAChannel:
    ch = OpenWAChannel("http://openwa", "sid", "key", SECRET, host="127.0.0.1", port=free_port())
    handler = handler or (lambda request: httpx.Response(200, json={}))
    ch._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), headers={"X-API-Key": "key"})
    return ch


async def post(ch: OpenWAChannel, payload: dict, *, sign: bool = True, key: str = "k1") -> int:
    body = json.dumps(payload).encode()
    sig = "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest() if sign else "sha256=bad"
    reader, writer = await asyncio.open_connection("127.0.0.1", ch._port)
    writer.write(
        f"POST /webhook HTTP/1.1\r\nContent-Length: {len(body)}\r\nX-OpenWA-Signature: {sig}\r\n"
        f"X-OpenWA-Idempotency-Key: {key}\r\n\r\n".encode() + body
    )
    await writer.drain()
    status = int((await reader.readline()).split()[1])
    writer.close()
    return status


def voice(msg_id="MSG1", **over) -> dict:
    data = {
        "id": msg_id,
        "chatId": CHAT,
        "author": "972500000000@c.us",
        "type": "voice",
        "fromMe": False,
        "media": {"mimetype": "audio/ogg; codecs=opus", "data": base64.b64encode(b"OGGDATA").decode()},
    }
    data.update(over)
    return {"event": "message.received", "sessionId": "sid", "data": data}


async def first(ch: OpenWAChannel):
    return await asyncio.wait_for(anext(ch.messages()), 5)


def run_with_server(ch, scenario):
    async def main():
        gen = ch.messages()
        task = asyncio.ensure_future(anext(gen))
        await asyncio.sleep(0.1)
        try:
            return await scenario(task)
        finally:
            task.cancel()
            await ch.close()

    return asyncio.run(main())


def test_voice_note_inline_media_is_yielded():
    ch = make_channel()

    async def scenario(task):
        assert await post(ch, voice()) == 200
        msg = await asyncio.wait_for(task, 5)
        assert msg.audio == b"OGGDATA"
        assert msg.mime_type == "audio/ogg"
        assert (msg.chat_id, msg.message_id, msg.sender_id) == (CHAT, "MSG1", "972500000000@c.us")

    run_with_server(ch, scenario)


def test_bad_signature_is_rejected_and_not_queued():
    ch = make_channel()

    async def scenario(task):
        assert await post(ch, voice(), sign=False) == 401
        assert ch._queue.empty()

    run_with_server(ch, scenario)


def test_ignores_own_text_and_other_events_and_dedupes():
    ch = make_channel()

    async def scenario(task):
        assert await post(ch, voice(fromMe=True), key="a") == 200
        assert await post(ch, voice(type="reaction"), key="b") == 200
        assert await post(ch, {"event": "session.status", "data": {}}, key="c") == 200
        assert ch._queue.empty()
        assert await post(ch, voice(), key="same") == 200
        assert await post(ch, voice(), key="same") == 200
        await asyncio.wait_for(task, 5)
        assert ch._queue.empty()

    run_with_server(ch, scenario)


def test_omitted_media_is_downloaded_from_gateway():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/api/sessions/sid/messages/{CHAT.replace('@', '%40')}/MSG2/media" or "MSG2" in request.url.path
        assert request.headers["x-api-key"] == "key"
        return httpx.Response(200, content=b"BIGOGG", headers={"content-type": "audio/ogg"})

    ch = make_channel(handler)

    async def scenario(task):
        await post(ch, voice("MSG2", media={"mimetype": "audio/ogg", "omitted": True, "sizeBytes": 9}))
        msg = await asyncio.wait_for(task, 5)
        assert msg.audio == b"BIGOGG"

    run_with_server(ch, scenario)


def test_reply_sends_plain_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"], seen["body"] = request.url.path, json.loads(request.content)
        return httpx.Response(201, json={})

    ch = make_channel(handler)

    async def scenario(task):
        await post(ch, voice())
        msg = await asyncio.wait_for(task, 5)
        await ch.reply(msg, "שלום")

    run_with_server(ch, scenario)
    assert seen["path"] == "/api/sessions/sid/messages/send-text"
    assert seen["body"] == {"chatId": CHAT, "text": "שלום"}


def test_registry_builds_openwa_and_requires_secret():
    ch = build_channel(
        {"type": "openwa", "base_url": "http://x", "session_id": "s", "api_key": "k", "webhook_secret": "w"}
    )
    assert ch.name == "openwa"
    try:
        build_channel({"type": "openwa", "base_url": "http://x", "session_id": "s", "api_key": "k", "webhook_secret": ""})
    except ValueError as exc:
        assert "webhook_secret" in str(exc)
    else:
        raise AssertionError("empty secret must be rejected")


def test_voice_note_is_marked_read_when_enabled():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"success": True})

    ch = make_channel(handler)

    async def scenario(task):
        await post(ch, voice("MSG9"))
        await asyncio.wait_for(task, 5)
        await asyncio.sleep(0.1)

    run_with_server(ch, scenario)
    assert seen == [("/api/sessions/sid/chats/read", {"chatId": CHAT, "messageIds": ["MSG9"]})]


def test_text_message_is_yielded_as_unsupported():
    ch = make_channel()

    async def scenario(task):
        assert await post(ch, voice(type="chat", media=None), key="t") == 200
        return await asyncio.wait_for(task, 5)

    msg = run_with_server(ch, scenario)
    assert isinstance(msg, UnsupportedMessage)
    assert (msg.chat_id, msg.kind) == (CHAT, "chat")
