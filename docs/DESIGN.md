# Design

kolnote turns voice notes into text. The core is small and fixed. Chat services and speech
engines plug in as adapters.

```
 chat service            core (pipeline.py)                speech engine
┌───────────────┐   ┌────────────────────────────┐   ┌──────────────────┐
│ Channel       │──▶│ policy check               │──▶│ STTEngine        │
│  openwa       │   │ unsupported? reply + stop  │   │  faster_whisper  │
│  telegram     │◀──│ transcribe, reply, log     │   │  openai_compat   │
│  folder       │   └────────────────────────────┘   └──────────────────┘
└───────────────┘
```

The core is the source of truth for behavior: who is allowed, what happens to non-voice
messages, how replies are logged, how errors are handled. A channel adapter only does the
chat-specific parts: receive messages, fetch the audio, send text back.

## The two ports (`ports.py`)

```python
class Channel(Protocol):
    name: str
    @classmethod
    def from_config(cls, options: dict) -> Self: ...
    def messages(self) -> AsyncIterator[Incoming]: ...
    async def reply(self, to: Incoming, text: str) -> None: ...
    async def close(self) -> None: ...        # must be safe to call twice

class STTEngine(Protocol):
    name: str
    @classmethod
    def from_config(cls, options: dict) -> Self: ...
    async def transcribe(self, audio: bytes, *, mime_type: str, language: str | None) -> Transcript: ...
    async def close(self) -> None: ...
```

`messages()` yields `Incoming`, which is one of:

- `IncomingAudio` (channel, chat id, message id, sender id, audio bytes, MIME type, duration if
  known, free-form `meta`, time received).
- `UnsupportedMessage` (same ids plus a `kind` such as `text` or `image`). The adapter reports
  these so the core can answer "I can only transcribe voice notes". The adapter never decides
  that itself.

## What the core does for every message (`pipeline.handle`)

1. Check the policy. A denied message is dropped and logged.
2. `UnsupportedMessage`: send the configured reply (`unsupported_text`, empty disables) and stop.
3. `IncomingAudio`: log `received`, transcribe, send the transcript, log `replied` with engine,
   STT time, total time and character count.
4. Any exception is logged for that one message. The loop keeps running.

Transcript text is never logged.

## Registry and plugins (`registry.py`)

`type = "..."` in `[channel]` or `[stt]` is resolved in this order:

1. A built-in name (`openwa`, `telegram`, `folder`, `faster_whisper`, `openai_compat`).
2. An installed Python entry point in group `kolnote.channels` or `kolnote.stt`.
3. A dotted path `package.module:ClassName`.

So a third party can add a channel without touching this repository. See
[ADDING-A-CHANNEL.md](ADDING-A-CHANNEL.md). `kolnote.testing.check_channel` is a conformance
helper: it drives any adapter through receive, reply and double close.

## Access control (`policy.py`)

Deny by default. A message passes if its chat is in `allow_chats` or its sender is in
`allow_senders`, or `open = true`. `max_duration_s` rejects long audio when the channel reports
a duration. The duration check is skipped for messages that have none.

## Configuration (`config.py`)

TOML file plus environment variables. Environment wins. Every key maps to
`KOLNOTE_<SECTION>__<KEY>`; list values are comma separated; values are coerced to the type in
the file. Secrets are read from the environment only and are never accepted on the command
line.

## Security decisions

- OpenWA webhooks must carry a valid `X-OpenWA-Signature` (HMAC-SHA256 of the body). kolnote
  refuses to start without a webhook secret and rejects unsigned requests. The delivery's
  idempotency key is remembered (last 1024) so repeated deliveries are dropped.
- The bot ignores its own messages, so it cannot answer itself in a loop.
- Telegram puts the bot token in every request URL. kolnote sets the `httpx` logger to WARNING
  and logs only the exception type for Telegram errors, never the message text, because the text
  can contain the URL.
- Audio is processed in memory. The OpenWA and Telegram adapters write nothing to disk (the
  `folder` channel writes the `.txt` next to the input, by design). Transcripts are not logged.
- Chat-scoped gateway keys are supported and recommended. A scoped OpenWA key cannot send quoted
  replies, so the OpenWA adapter sends plain replies.

## Why this shape

The first version served only WhatsApp. When Telegram was added, `pipeline.py` did not change:
the integration test `test_core_pipeline_serves_telegram_unchanged` runs the same `pipeline.run` against
a mocked Telegram API. That is the property to keep. If a new channel needs a core change, the
port is missing something, and the port should be fixed instead of special-casing the channel.
