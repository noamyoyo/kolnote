import asyncio
from types import SimpleNamespace

from kolnote import pipeline
from kolnote.adapters.stt.faster_whisper import join_segments
from kolnote.models import IncomingAudio
from kolnote.policy import Policy


def seg(text, start, end):
    return SimpleNamespace(text=text, start=start, end=end)


def test_join_segments_uses_spaces_by_default_and_skips_empty_segments():
    segs = [seg(" a ", 0, 1), seg("", 1, 2), seg("b", 5, 6)]
    assert join_segments(segs) == "a b"


def test_join_segments_breaks_paragraph_on_long_pause_only():
    segs = [seg("a", 0, 1), seg("b", 1.2, 2), seg("c", 4.0, 5)]
    assert join_segments(segs, 1.5) == "a b\n\nc"


class FakeChannel:
    def __init__(self):
        self.replies = []

    async def reply(self, to, text):
        self.replies.append(text)


class FakeSTT:
    name = "fake"

    async def transcribe(self, audio, *, mime_type, language):
        from kolnote.models import Transcript

        return Transcript(text="hello", engine="fake", processing_s=0.0)


def audio(duration):
    return IncomingAudio("c", "chat", "m", "u", b"x", duration_s=duration)


def run_handle(msg, **kw):
    ch = FakeChannel()
    asyncio.run(pipeline.handle(msg, channel=ch, engine=FakeSTT(), policy=Policy(open=True), language=None,
                                error_text=None, **kw))
    return ch.replies


def test_long_recording_gets_notice_before_transcript():
    assert run_handle(audio(600), long_text="long") == ["long", "hello"]


def test_short_recording_gets_no_notice():
    assert run_handle(audio(30), long_text="long") == ["hello"]


def test_long_notice_disabled_when_text_empty():
    assert run_handle(audio(600), long_text=None) == ["hello"]


def test_unknown_duration_is_probed_from_the_audio(monkeypatch):
    monkeypatch.setattr(pipeline, "_probe_duration_s", lambda data: 300.0)
    assert run_handle(audio(None), long_text="long") == ["long", "hello"]
    monkeypatch.setattr(pipeline, "_probe_duration_s", lambda data: None)
    assert run_handle(audio(None), long_text="long") == ["hello"]
