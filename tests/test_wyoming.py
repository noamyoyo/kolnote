import asyncio
import io
import wave

import pytest

pytest.importorskip("wyoming")

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info

from kolnote.models import Transcript as KTranscript
from kolnote.wyoming_server import WyomingServer


class Engine:
    name = "fake"

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    async def transcribe(self, audio, *, mime_type, language):
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append((audio, mime_type, language))
        return KTranscript(text="hello", engine="fake", processing_s=0.01)

    async def close(self):
        pass


class Multi(Engine):
    models = {"en": ["en"], "he": ["he"]}


async def started(engine, **kw):
    server = WyomingServer(engine, port=0, **kw)
    await server.start()
    return server


def test_describe_advertises_each_models_languages():
    async def go():
        server = await started(Multi())
        async with AsyncTcpClient("127.0.0.1", server.port) as client:
            await client.write_event(Describe().event())
            info = Info.from_event(await client.read_event())
        models = {m.name: m.languages for m in info.asr[0].models}
        assert models == {"en": ["en"], "he": ["he"]}
        await server.close()

    asyncio.run(go())


def test_single_engine_needs_a_language_to_advertise():
    with pytest.raises(ValueError, match="languages"):
        WyomingServer(Engine())
    assert WyomingServer(Engine(), language="en")


def test_transcribe_request_becomes_a_wav_and_a_transcript():
    async def go():
        engine = Engine()
        server = await started(engine, language="en")
        async with AsyncTcpClient("127.0.0.1", server.port) as client:
            await client.write_event(Transcribe(language="he").event())
            await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
            for _ in range(3):
                await client.write_event(AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00\x01" * 800).event())
            await client.write_event(AudioStop().event())
            reply = Transcript.from_event(await client.read_event())
        assert reply.text == "hello"
        audio, mime, language = engine.calls[0]
        assert mime == "audio/wav" and language == "he"
        with wave.open(io.BytesIO(audio)) as wav:
            assert (wav.getframerate(), wav.getsampwidth(), wav.getnchannels(), wav.getnframes()) == (16000, 2, 1, 2400)
        await server.close()

    asyncio.run(go())


def test_language_falls_back_to_the_configured_one():
    async def go():
        engine = Engine()
        server = await started(engine, language="en")
        async with AsyncTcpClient("127.0.0.1", server.port) as client:
            await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
            await client.write_event(AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00\x00" * 10).event())
            await client.write_event(AudioStop().event())
            await client.read_event()
        assert engine.calls[0][2] == "en"
        await server.close()

    asyncio.run(go())


def test_engine_failure_returns_an_empty_transcript_instead_of_hanging():
    async def go():
        server = await started(Engine(fail=True), language="en")
        async with AsyncTcpClient("127.0.0.1", server.port) as client:
            await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
            await client.write_event(AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00\x00" * 10).event())
            await client.write_event(AudioStop().event())
            assert Transcript.from_event(await client.read_event()).text == ""
        await server.close()

    asyncio.run(go())
