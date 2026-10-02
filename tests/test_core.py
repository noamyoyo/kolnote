import asyncio
from collections.abc import AsyncIterator

import pytest

from kolnote import pipeline
from kolnote.metrics import normalize, rate, word_errors
from kolnote.models import IncomingAudio, Transcript, UnsupportedMessage
from kolnote.policy import Policy
from kolnote.ports import Channel, STTEngine
from kolnote.registry import build_channel, build_stt


def audio(chat="c1", sender="s1", duration=None) -> IncomingAudio:
    return IncomingAudio("fake", chat, "m1", sender, b"x", duration_s=duration)


class FakeChannel:
    name = "fake"

    def __init__(self, msgs):
        self.msgs, self.replies, self.closed = msgs, [], False

    @classmethod
    def from_config(cls, options):
        return cls([])

    async def messages(self) -> AsyncIterator[IncomingAudio]:
        for m in self.msgs:
            yield m

    async def reply(self, to, text):
        self.replies.append(text)

    async def close(self):
        self.closed = True


class FakeSTT:
    name = "fake-stt"

    def __init__(self, text="שלום עולם", fail=False):
        self.text, self.fail = text, fail

    @classmethod
    def from_config(cls, options):
        return cls(**options)

    async def transcribe(self, audio, *, mime_type, language):
        if self.fail:
            raise RuntimeError("boom")
        return Transcript(self.text, self.name, 0.01, language)

    async def close(self):
        pass


def run(channel, engine, policy, **kw):
    asyncio.run(pipeline.run(channel, engine, policy, **kw))


def test_fakes_satisfy_ports():
    assert isinstance(FakeChannel([]), Channel)
    assert isinstance(FakeSTT(), STTEngine)


def test_pipeline_replies_with_transcript_and_closes():
    ch = FakeChannel([audio()])
    run(ch, FakeSTT(), Policy(open=True), language="he")
    assert ch.replies == ["שלום עולם"]
    assert ch.closed


def test_empty_transcript_gets_placeholder():
    ch = FakeChannel([audio()])
    run(ch, FakeSTT(text=""), Policy(open=True))
    assert ch.replies == [pipeline.NO_SPEECH]


def test_policy_denies_by_default():
    ch = FakeChannel([audio()])
    run(ch, FakeSTT(), Policy())
    assert ch.replies == []


def test_policy_allowlist_and_duration():
    p = Policy(allow_chats=frozenset({"c1"}), max_duration_s=60)
    assert p.check(audio(chat="c1")) is None
    assert p.check(audio(chat="other", sender="x")) == "not allowlisted"
    assert "longer" in p.check(audio(chat="c1", duration=61))


def test_engine_failure_sends_error_text_and_keeps_running():
    ch = FakeChannel([audio(), audio()])
    run(ch, FakeSTT(fail=True), Policy(open=True), error_text="nope")
    assert ch.replies == ["nope", "nope"]


def test_unsupported_message_gets_reply_but_only_when_allowed():
    ch = FakeChannel([UnsupportedMessage("fake", "c1", "m1", "s1", "chat"), UnsupportedMessage("fake", "x", "m2", "s2", "chat")])
    run(ch, FakeSTT(), Policy(allow_chats=frozenset({"c1"})), unsupported_text="voice only")
    assert ch.replies == ["voice only"]


class ColdSTT(FakeSTT):
    def __init__(self):
        super().__init__()
        self.cold = True

    def needs_load(self, language=None):
        return self.cold

    async def transcribe(self, audio, *, mime_type, language):
        self.cold = False
        return await super().transcribe(audio, mime_type=mime_type, language=language)


def test_loading_notice_is_sent_once_before_a_cold_transcript():
    ch = FakeChannel([audio(), audio()])
    run(ch, ColdSTT(), Policy(open=True), loading_text="loading")
    assert ch.replies == ["loading", "שלום עולם", "שלום עולם"]


def test_no_loading_notice_when_disabled_or_engine_cannot_say():
    ch = FakeChannel([audio()])
    run(ch, ColdSTT(), Policy(open=True), loading_text=None)
    assert ch.replies == ["שלום עולם"]
    ch = FakeChannel([audio()])
    run(ch, FakeSTT(), Policy(open=True), loading_text="loading")
    assert ch.replies == ["שלום עולם"]


def test_unsupported_reply_can_be_disabled():
    ch = FakeChannel([UnsupportedMessage("fake", "c1", "m1", "s1", "image")])
    run(ch, FakeSTT(), Policy(open=True), unsupported_text=None)
    assert ch.replies == []


def test_registry_loads_dotted_path_and_rejects_unknown():
    assert isinstance(build_channel({"type": "tests.test_core:FakeChannel"}), FakeChannel)
    assert isinstance(build_stt({"type": "tests.test_core:FakeSTT", "text": "x"}), FakeSTT)
    with pytest.raises(ValueError, match="unknown"):
        build_stt({"type": "nope"})
    with pytest.raises(ValueError, match="missing 'type'"):
        build_stt({})


def test_normalize_strips_niqqud_punctuation_case():
    assert normalize("שָׁלוֹם, עולם!") == "שלום עולם"
    assert normalize("Hello,  WORLD.") == "hello world"
    assert normalize("בית־ספר") == "בית ספר"


def test_word_errors():
    assert word_errors("שלום עולם", "שלום עולם!") == (0, 2)
    assert word_errors("a b c", "a x c") == (1, 3)
    assert word_errors("a b c", "a c") == (1, 3)
    assert rate(*word_errors("", "")) == 0.0


def test_entry_point_plugins_are_found(monkeypatch):
    from importlib.metadata import EntryPoint

    from kolnote import registry

    eps = [EntryPoint("mine", "tests.test_core:FakeChannel", "kolnote.channels")]
    monkeypatch.setattr(registry, "entry_points", lambda group: eps if group == "kolnote.channels" else [])
    assert isinstance(build_channel({"type": "mine"}), FakeChannel)
    with pytest.raises(ValueError, match="mine"):
        build_channel({"type": "nope"})


def test_run_channels_shares_engine_with_per_channel_policy():
    a = FakeChannel([audio(chat="c1")])
    b = FakeChannel([audio(chat="c2"), audio(chat="c1")])
    engine = FakeSTT()
    pols = [(a, Policy(allow_chats=frozenset({"c1"}))), (b, Policy(allow_chats=frozenset({"c2"})))]
    asyncio.run(pipeline.run_channels(pols, engine))
    assert a.replies == ["שלום עולם"]
    assert b.replies == ["שלום עולם"]  # c1 on channel b is denied by b's own policy
    assert a.closed and b.closed


def test_run_channels_stops_everything_when_one_channel_fails():
    class Broken(FakeChannel):
        async def messages(self):
            raise ConnectionError("gateway down")
            yield

    ok, bad = FakeChannel([]), Broken([])
    with pytest.raises(ExceptionGroup):
        asyncio.run(pipeline.run_channels([(ok, Policy(open=True)), (bad, Policy(open=True))], FakeSTT()))
    assert ok.closed and bad.closed


def test_run_channels_runs_services_and_closes_them():
    class Service:
        closed = False
        started = False

        async def serve_forever(self):
            self.started = True
            raise ConnectionError("listener died")

        async def close(self):
            self.closed = True

    service, channel = Service(), FakeChannel([])
    with pytest.raises(ExceptionGroup):
        asyncio.run(pipeline.run_channels([(channel, Policy(open=True))], FakeSTT(), services=[service]))
    assert service.started and service.closed and channel.closed
