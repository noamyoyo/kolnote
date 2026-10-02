# Adding a channel

A channel is the only part of kolnote that knows about a messenger. The core
(`pipeline.py`) is the source of truth: it applies the allowlist, runs the speech engine and
sends the answer back. Your adapter only translates between the messenger and four calls.

```
messenger  --(your adapter)-->  IncomingAudio / UnsupportedMessage  -->  core
messenger  <--(your adapter)--  reply(msg, text)                    <--  core
```

The same recipe covers a new messenger (Telegram, Signal, Matrix, ...) and a second way of
reaching an existing one (for example another WhatsApp gateway next to OpenWA).

## The contract

Implement this class shape (see `src/kolnote/ports.py`):

| Member | What it does |
|--------|--------------|
| `name` | Short lowercase id, e.g. `"telegram"`. Appears in logs. |
| `from_config(options)` | Classmethod. `options` is the `[channel]` table minus `type`. |
| `messages()` | Async generator. Yields `IncomingAudio` for each voice note and `UnsupportedMessage` for anything else a person sent (text, image, ...). |
| `reply(to, text)` | Send `text` back to the chat `to` came from. |
| `close()` | Release connections. Must be safe to call twice. |

Rules the core relies on:

- Yield only messages from other people. Never yield the bot's own messages, or it will
  answer itself.
- `chat_id`, `message_id`, `sender_id` are strings. They are what users put in
  `allow_chats` / `allow_senders`, so use the messenger's own ids and say so in your docs.
  Use `""` for `sender_id` if unknown.
- `audio` is the raw file bytes. Set `mime_type` and, if the messenger tells you, `duration_s`.
- Do not apply the allowlist yourself and do not transcribe. Both belong to the core.
- Ignore non-message events and system messages instead of yielding them.
- Deduplicate redelivered messages if the messenger can send them twice (webhooks can).
- Never log secrets. HTTP client errors often contain the request URL, so log the exception
  type, not its text, when the URL holds a token.

## Skeleton

```python
from kolnote.models import IncomingAudio, UnsupportedMessage


class MyChannel:
    name = "mychat"

    def __init__(self, token: str, *, poll_s: float = 2.0) -> None:
        ...

    @classmethod
    def from_config(cls, options):
        return cls(**options)

    async def messages(self):
        while True:
            for event in await self._fetch_new_events():
                if event.is_voice:
                    yield IncomingAudio(
                        channel=self.name, chat_id=event.chat, message_id=event.id,
                        sender_id=event.sender, audio=await self._download(event),
                        mime_type="audio/ogg", duration_s=event.seconds,
                    )
                elif event.is_user_content:
                    yield UnsupportedMessage(self.name, event.chat, event.id, event.sender, event.kind)

    async def reply(self, to, text):
        await self._send(to.chat_id, text, in_reply_to=to.message_id)

    async def close(self):
        ...
```

Two reference implementations: `adapters/channels/telegram.py` (polling, no inbound port) and
`adapters/channels/openwa.py` (signed webhook in, REST out). `folder.py` is the smallest.

## Make it selectable

Pick one:

1. **Inside this repo.** Add one line to `CHANNELS` in `src/kolnote/registry.py`, then use
   `type = "mychat"`.
2. **Dotted path.** No registration at all: `type = "mypackage.module:MyChannel"`.
3. **Installable plugin.** In your package's `pyproject.toml`:

   ```toml
   [project.entry-points."kolnote.channels"]
   mychat = "mypackage.module:MyChannel"
   ```

   After `pip install mypackage`, `type = "mychat"` works. Speech engines use the group
   `kolnote.stt`.

## Verify it

`kolnote.testing.check_channel` feeds one voice note through your adapter, replies to it and
closes it twice, and fails with a clear message if the contract is broken. Fake the messenger
with `httpx.MockTransport`, as `tests/test_channels.py` does for Telegram:

```python
import asyncio
from kolnote.testing import check_channel

def test_my_channel_passes_contract():
    channel = MyChannel(...)

    async def deliver():          # make exactly one voice note arrive
        fake_upstream.push_voice()

    asyncio.run(check_channel(channel, deliver))
```

Then also run it through the real core, as `test_core_pipeline_serves_telegram_unchanged` does:
that proves the pipeline needs no change for your channel. A mock only proves your reading of
the messenger's API, so finish with one live test: send a real voice note and a real text.
