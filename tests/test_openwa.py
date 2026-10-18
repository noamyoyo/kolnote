import asyncio
import base64
import hashlib
import hmac
import json
import socket

import httpx
import pytest

from kolnote.adapters.channels.openwa import OpenWAChannel, _is_audio, _split_text
from kolnote.models import UnsupportedMessage
from kolnote.registry import build_channel

SECRET = "s3cret"
CHAT = "120363000000000000@g.us"
CRLF = bytes([13, 10])


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


def document(msg_id="DOC1", **media) -> dict:
    payload = voice(msg_id, type="document", media={"data": base64.b64encode(b"M4ADATA").decode(), **media})
    return payload


def test_audio_document_is_transcribed_by_mimetype():
    ch = make_channel()

    async def scenario(task):
        await post(ch, document(mimetype="audio/mp4", filename="Call.m4a"))
        msg = await asyncio.wait_for(task, 5)
        assert msg.audio == b"M4ADATA"
        assert msg.mime_type == "audio/mp4"

    run_with_server(ch, scenario)


def test_audio_document_is_transcribed_by_extension_when_mimetype_missing():
    ch = make_channel()

    async def scenario(task):
        await post(ch, document(filename="note.M4A"))
        msg = await asyncio.wait_for(task, 5)
        assert msg.audio == b"M4ADATA"
        assert msg.mime_type == "audio/mp4"

    run_with_server(ch, scenario)


def test_audio_document_media_under_metadata_is_read():
    ch = make_channel()
    payload = voice("DOC2", type="document", media=None, metadata={"media": {
        "mimetype": "audio/mp4", "filename": "Call.m4a", "data": base64.b64encode(b"M4ADATA").decode()}})

    async def scenario(task):
        await post(ch, payload)
        msg = await asyncio.wait_for(task, 5)
        assert msg.audio == b"M4ADATA"

    run_with_server(ch, scenario)


def test_non_audio_document_stays_unsupported():
    ch = make_channel()

    async def scenario(task):
        await post(ch, document(mimetype="application/pdf", filename="bill.pdf"))
        msg = await asyncio.wait_for(task, 5)
        assert isinstance(msg, UnsupportedMessage)
        assert msg.kind == "document"

    run_with_server(ch, scenario)


def test_video_document_with_mp4_extension_stays_unsupported():
    ch = make_channel()

    async def scenario(task):
        await post(ch, document(mimetype="video/mp4", filename="clip.mp4"))
        assert isinstance(await asyncio.wait_for(task, 5), UnsupportedMessage)

    run_with_server(ch, scenario)


def test_audio_document_without_inline_data_is_downloaded_with_resolved_mime():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "DOC3" in request.url.path
        return httpx.Response(200, content=b"BIGM4A", headers={"content-type": "application/octet-stream"})

    ch = make_channel(handler)

    async def scenario(task):
        await post(ch, voice("DOC3", type="document", media={"mimetype": "audio/mp4", "filename": "Call.m4a", "omitted": True}))
        msg = await asyncio.wait_for(task, 5)
        assert msg.audio == b"BIGM4A"
        assert msg.mime_type == "audio/mp4"

    run_with_server(ch, scenario)


def test_audio_document_duration_is_read_from_metadata():
    ch = make_channel()
    payload = voice("DOC4", type="document", media=None, metadata={"media": {
        "mimetype": "audio/mp4", "duration": 630, "data": base64.b64encode(b"M4ADATA").decode()}})

    async def scenario(task):
        await post(ch, payload)
        assert (await asyncio.wait_for(task, 5)).duration_s == 630

    run_with_server(ch, scenario)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"type": "voice"}, True),
        ({"type": "document", "media": {"mimetype": "Audio/MP4; codecs=mp4a.40.2"}}, True),
        ({"type": "document", "media": {"mimetype": "application/octet-stream", "filename": "a.m4a"}}, True),
        ({"type": "document", "media": {"mimetype": "application/octet-stream", "filename": "a.pdf"}}, False),
        ({"type": "document", "media": {"mimetype": "application/ogg", "filename": "a.opus"}}, True),
        ({"type": "document", "media": {"mimetype": "application/pdf", "filename": "a.m4a"}}, False),
        ({"type": "document", "media": {"data": "x"}, "metadata": {"media": {"mimetype": "audio/mp4"}}}, True),
        ({"type": "document", "metadata": {"filename": "a.mp3"}}, True),
        ({"type": "document", "media": {"mimetype": "application/octet-stream", "filename": "clip.mp4"}}, False),
        ({"type": "document", "media": {"filename": "clip.webm"}}, False),
        ({"type": "document", "media": {"mimetype": "audio/webm", "filename": "clip.webm"}}, True),
        ({"type": "document", "media": "oops"}, False),
        ({"type": "document", "metadata": ["x"]}, False),
        ({"type": "document", "metadata": {"media": "oops"}}, False),
        ({"type": "document"}, False),
        ({"type": "image"}, False),
    ],
)
def test_is_audio_table(data, expected):
    assert _is_audio(data) is expected


def test_oversize_body_is_logged_and_rejected(caplog):
    ch = make_channel()

    async def scenario(task):
        reader, writer = await asyncio.open_connection("127.0.0.1", ch._port)
        writer.write(b"POST /webhook HTTP/1.1" + CRLF + f"Content-Length: {64 * 1024 * 1024}".encode() + CRLF + CRLF)
        await writer.drain()
        assert int((await reader.readline()).split()[1]) == 413
        writer.close()

    with caplog.at_level("WARNING"):
        run_with_server(ch, scenario)
    assert "webhook body too large" in caplog.text


def test_split_text_keeps_short_text_whole_and_splits_long_text_at_spaces():
    assert _split_text("שלום") == ["שלום"]
    text = " ".join(["מילה"] * 3000)
    parts = _split_text(text)
    assert len(parts) > 1
    assert all(len(p) <= 4000 for p in parts)
    assert " ".join(parts) == text
    assert _split_text("x" * 9000) == ["x" * 4000, "x" * 4000, "x" * 1000]


def test_reply_sends_long_text_in_order_in_several_messages():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["text"])
        return httpx.Response(201, json={})

    ch = make_channel(handler)
    text = " ".join(f"w{i}" for i in range(3000))

    async def scenario(task):
        msg = UnsupportedMessage("openwa", CHAT, "M", "u", "text")
        await ch.reply(msg, text)

    run_with_server(ch, scenario)
    assert len(sent) > 1 and all(len(p) <= 4000 for p in sent)
    assert " ".join(sent) == text


def test_split_text_never_returns_an_empty_part():
    from kolnote.adapters.channels.openwa import _split_text

    parts = _split_text("a" * 3999 + " " * 10)
    assert parts == ["a" * 3999]
    assert _split_text("") == [""]


def test_duration_accepts_numbers_and_numeric_strings_only():
    from kolnote.adapters.channels.openwa import _duration

    assert _duration(630) == 630.0
    assert _duration("630.5") == 630.5
    for bad in (None, "", "abc", 0, -3, "inf", float("nan"), [1], {"s": 1}):
        assert _duration(bad) is None


def test_audio_document_with_string_duration_gives_a_number():
    ch = make_channel()
    payload = voice("DOC5", type="document", media=None, metadata={"media": {
        "mimetype": "audio/mp4", "duration": "630", "data": base64.b64encode(b"M4ADATA").decode()}})

    async def scenario(task):
        await post(ch, payload)
        assert (await asyncio.wait_for(task, 5)).duration_s == 630.0

    run_with_server(ch, scenario)
