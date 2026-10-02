import asyncio
import socket

import httpx
import pytest

from kolnote import server as server_module
from kolnote.adapters.stt.openai_compat import OpenAICompatSTT
from kolnote.config import Settings
from kolnote.models import Transcript
from kolnote.server import TranscriptionServer, parse_multipart

TEXT = "שלום עולם"


class StubSTT:
    name = "stub"

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self.fail = fail
        self.preloaded = False
        self.closed = False

    async def preload(self) -> None:
        self.preloaded = True

    async def transcribe(self, audio: bytes, *, mime_type: str, language: str | None) -> Transcript:
        if self.fail:
            raise RuntimeError("secret internal detail")
        self.calls.append({"audio": audio, "mime_type": mime_type, "language": language})
        return Transcript(text=TEXT, engine=self.name, processing_s=0.01, language=language)

    async def close(self) -> None:
        self.closed = True


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def with_server(scenario, engine=None, **options):
    engine = engine or StubSTT()

    async def main():
        srv = TranscriptionServer(engine, host="127.0.0.1", port=0, **options)
        await srv.start()
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{srv.port}") as client:
                return await scenario(client, srv)
        finally:
            await srv.close()

    return asyncio.run(main())


def transcribe(client, *, data=None, files=None, headers=None):
    files = {"file": ("audio", b"OGGDATA", "audio/ogg")} if files is None else files
    return client.post("/v1/audio/transcriptions", data=data or {}, files=files, headers=headers)


def test_transcribes_and_returns_json():
    engine = StubSTT()

    async def scenario(client, srv):
        resp = await transcribe(client, data={"model": "ignored", "language": "he"})
        assert resp.status_code == 200
        assert resp.json() == {"text": TEXT}

    with_server(scenario, engine)
    assert engine.calls == [{"audio": b"OGGDATA", "mime_type": "audio/ogg", "language": "he"}]


def test_language_falls_back_to_server_default():
    engine = StubSTT()

    async def scenario(client, srv):
        await transcribe(client)
        await transcribe(client, data={"language": "en"})

    with_server(scenario, engine, language="he")
    assert [c["language"] for c in engine.calls] == ["he", "en"]


def test_text_response_format():
    async def scenario(client, srv):
        resp = await transcribe(client, data={"response_format": "text"})
        assert resp.status_code == 200
        assert resp.text == TEXT
        assert (await transcribe(client, data={"response_format": "srt"})).status_code == 400

    with_server(scenario)


def test_api_key_is_enforced_but_health_is_open():
    async def scenario(client, srv):
        assert (await transcribe(client)).status_code == 401
        assert (await transcribe(client, headers={"Authorization": "Bearer wrong"})).status_code == 401
        assert (await transcribe(client, headers={"Authorization": "Basic k3y"})).status_code == 401
        assert (await transcribe(client, headers={"Authorization": "Bearer k3y"})).status_code == 200
        assert (await client.get("/healthz")).status_code == 200

    with_server(scenario, api_key="k3y")


def test_no_api_key_means_open_access():
    async def scenario(client, srv):
        assert (await transcribe(client)).status_code == 200

    with_server(scenario)


def test_bad_requests():
    async def scenario(client, srv):
        assert (await client.get("/nope")).status_code == 404
        assert (await client.get("/v1/audio/transcriptions")).status_code == 405
        assert (await client.post("/v1/audio/transcriptions", json={"a": 1})).status_code == 400
        assert (await transcribe(client, files={"other": ("x", b"1", "audio/ogg")})).status_code == 400
        assert (await transcribe(client, files={"file": ("audio", b"", "audio/ogg")})).status_code == 400

    with_server(scenario)


def test_oversized_body_is_rejected():
    async def scenario(client, srv):
        resp = await transcribe(client, files={"file": ("audio", b"x" * 4096, "audio/ogg")})
        assert resp.status_code == 413

    with_server(scenario, max_body_mb=0.001)


def test_engine_failure_is_500_without_leaking_details():
    async def scenario(client, srv):
        resp = await transcribe(client)
        assert resp.status_code == 500
        assert "secret internal detail" not in resp.text

    with_server(scenario, StubSTT(fail=True))


def test_openai_compat_engine_round_trips_through_the_server():
    engine = StubSTT()

    async def scenario(client, srv):
        stt = OpenAICompatSTT(f"http://127.0.0.1:{srv.port}/v1", "any-model")
        try:
            result = await stt.transcribe(b"OGGDATA", mime_type="audio/ogg", language="he")
        finally:
            await stt.close()
        assert result.text == TEXT

    with_server(scenario, engine, api_key=None)
    assert engine.calls[0]["language"] == "he"


def test_openai_compat_engine_sends_the_api_key(monkeypatch):
    monkeypatch.setenv("STT_KEY", "k3y")

    async def scenario(client, srv):
        stt = OpenAICompatSTT(f"http://127.0.0.1:{srv.port}/v1", "m", api_key_env="STT_KEY")
        try:
            assert (await stt.transcribe(b"OGGDATA", mime_type="audio/ogg", language=None)).text == TEXT
        finally:
            await stt.close()

    with_server(scenario, api_key="k3y")


def test_parse_multipart_handles_binary_and_unicode():
    boundary = "XBOUNDARY"
    audio = bytes(range(256)) * 4 + b"\r\n--not-a-boundary\r\n"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="language"\r\n\r\n'.encode()
        + "עברית".encode()
        + f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.ogg"\r\n'
        f"Content-Type: audio/ogg\r\n\r\n".encode()
        + audio
        + f"\r\n--{boundary}--\r\n".encode()
    )
    fields, file = parse_multipart(f"multipart/form-data; boundary={boundary}", body)
    assert fields == {"language": "עברית"}
    assert file == (audio, "audio/ogg")


def test_serve_preloads_the_engine_and_requires_a_set_key_variable(monkeypatch):
    engine = StubSTT()
    monkeypatch.setattr(server_module, "build_stt", lambda spec: engine)
    port = free_port()

    async def main():
        task = asyncio.ensure_future(server_module.serve(Settings(stt={"type": "x"}, server={"host": "127.0.0.1", "port": port})))
        async with httpx.AsyncClient() as client:
            for _ in range(50):
                try:
                    assert (await client.get(f"http://127.0.0.1:{port}/healthz")).status_code == 200
                    break
                except httpx.ConnectError:
                    await asyncio.sleep(0.1)
            else:
                pytest.fail("server did not start")
        assert engine.preloaded
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert engine.closed

    asyncio.run(main())

    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(ValueError, match="MISSING_KEY"):
        asyncio.run(server_module.serve(Settings(stt={"type": "x"}, server={"api_key_env": "MISSING_KEY"})))
