"""Every channel adapter must pass the same contract, and the unchanged core must work with each."""

import asyncio
import json

import httpx
import pytest

from kolnote import pipeline
from kolnote.adapters.channels.folder import FolderChannel
from kolnote.adapters.channels.telegram import TelegramChannel
from kolnote.models import Transcript, UnsupportedMessage
from kolnote.policy import Policy
from kolnote.registry import build_channel
from kolnote.testing import check_channel
from tests.test_openwa import CHAT, make_channel as make_openwa, post, voice


class FakeTelegram:
    """Just enough of the Bot API: queued updates, getFile, file download, sendMessage."""

    def __init__(self):
        self.updates: list[dict] = []
        self.sent: list[dict] = []
        self.actions: list[dict] = []

    def push(self, **message) -> None:
        n = len(self.updates) + 1
        message.setdefault("chat", {"id": -100123, "type": "supergroup"})
        message.setdefault("from", {"id": 42, "is_bot": False, "username": "dana"})
        message.setdefault("message_id", n)
        self.updates.append({"update_id": 1000 + n, "message": message})

    async def handler(self, request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.01)
        path = request.url.path
        if path.startswith("/file/bot"):
            return httpx.Response(200, content=b"OGGDATA")
        method = path.rsplit("/", 1)[-1]
        body = json.loads(request.content or b"{}")
        if method == "getUpdates":
            ready = [u for u in self.updates if u["update_id"] >= body["offset"]]
            return httpx.Response(200, json={"ok": True, "result": ready})
        if method == "getFile":
            return httpx.Response(200, json={"ok": True, "result": {"file_path": f"voice/{body['file_id']}.oga"}})
        if method == "sendMessage":
            self.sent.append(body)
        if method == "sendChatAction":
            self.actions.append(body)
        return httpx.Response(200, json={"ok": True, "result": True})


def make_telegram(fake: FakeTelegram) -> TelegramChannel:
    ch = TelegramChannel("123:TOKEN", poll_timeout_s=1)
    ch._client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    return ch


VOICE_NOTE = {"voice": {"file_id": "F1", "duration": 4, "mime_type": "audio/ogg"}}


def test_folder_channel_passes_contract(tmp_path):
    ch = FolderChannel(tmp_path, poll_s=0.01)

    async def deliver():
        (tmp_path / "note.ogg").write_bytes(b"OGG")

    msg = asyncio.run(check_channel(ch, deliver))
    assert msg.message_id == "note.ogg"
    assert (tmp_path / "note.txt").read_text().strip() == "kolnote contract check"


def test_openwa_channel_passes_contract():
    ch = make_openwa()

    async def deliver():
        assert await post(ch, voice()) == 200

    msg = asyncio.run(check_channel(ch, deliver))
    assert msg.chat_id == CHAT


def test_telegram_channel_passes_contract():
    fake = FakeTelegram()
    ch = make_telegram(fake)

    async def deliver():
        fake.push(**VOICE_NOTE)

    msg = asyncio.run(check_channel(ch, deliver))
    assert (msg.chat_id, msg.sender_id, msg.audio, msg.duration_s) == ("-100123", "42", b"OGGDATA", 4)
    assert fake.sent[0]["chat_id"] == -100123
    assert fake.sent[0]["reply_parameters"]["message_id"] == 1
    assert fake.actions and fake.actions[0]["action"] == "typing"


def test_telegram_text_photo_and_bot_messages():
    fake = FakeTelegram()
    ch = make_telegram(fake)
    fake.push(text="hello")
    fake.push(photo=[{"file_id": "P"}])
    fake.push(text="from a bot", **{"from": {"id": 7, "is_bot": True}})
    fake.push(new_chat_members=[{"id": 9}])

    async def collect():
        gen = ch.messages()
        got = [await asyncio.wait_for(anext(gen), 5) for _ in range(2)]
        await ch.close()
        return got

    got = asyncio.run(collect())
    assert [(type(m), m.kind) for m in got] == [(UnsupportedMessage, "text"), (UnsupportedMessage, "photo")]


def test_telegram_long_reply_is_split():
    fake = FakeTelegram()
    ch = make_telegram(fake)
    to = UnsupportedMessage("telegram", "5", "6", "7", "text")
    asyncio.run(ch.reply(to, "x" * 5000))
    assert [len(s["text"]) for s in fake.sent] == [4096, 904]


def test_telegram_requires_token_and_registry_builds_it():
    with pytest.raises(ValueError, match="token"):
        TelegramChannel("")
    assert isinstance(build_channel({"type": "telegram", "token": "1:x"}), TelegramChannel)


def test_core_pipeline_serves_telegram_unchanged():
    """Same pipeline.run that serves WhatsApp: voice note gets a transcript, text gets the notice."""
    fake = FakeTelegram()
    fake.push(**VOICE_NOTE)
    fake.push(text="hi")
    ch = make_telegram(fake)

    class Engine:
        name = "fake-stt"

        async def transcribe(self, audio, *, mime_type, language):
            return Transcript("שלום עולם", self.name, 0.01, language)

        async def close(self):
            pass

    async def main():
        task = asyncio.ensure_future(
            pipeline.run(ch, Engine(), Policy(allow_chats=frozenset({"-100123"})), unsupported_text="voice only")
        )
        for _ in range(100):
            if len(fake.sent) >= 2:
                break
            await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(main())
    assert sorted(s["text"] for s in fake.sent) == sorted(["שלום עולם", "voice only"])
