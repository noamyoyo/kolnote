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

## Several channels, one model (`pipeline.run_channels`)

`kolnote run` can host several channels in one process. `run_channels` takes a list of
`(Channel, Policy)` pairs and one `STTEngine`. Each channel gets its own consumer task and its
own policy; all of them share the engine and a semaphore, so the model sees one note at a time.

```
 telegram ─┐
 openwa   ─┼─▶ run_channels ─▶ faster_whisper (one copy)
 folder   ─┘
```

Config is `[channels.<name>]` tables (`type` defaults to the name, an optional `policy`
sub-table). The tables map cleanly to `KOLNOTE_CHANNELS__<NAME>__<KEY>` env overrides, which a
TOML array of tables would not. One channel failing ends the process: a half-working bot hides
errors, and a restart policy is the simplest recovery. `pipeline.run` is the one-channel
wrapper, so existing callers did not change.

For a GPU shared with another model, `FasterWhisperSTT(idle_unload_s=...)` drops the model after
that many idle seconds and loads it again on the next request.

## Several models, one engine (`adapters/stt/multi.py`)

`MultiSTT` is an `STTEngine` that wraps several other engines, so the core and the channels do not
know there is more than one. It routes each request by language: the language is normalised to its
first subtag (`en-US` becomes `en`), looked up in a table built from each model's `languages`, and
falls back to `default`. A model with exactly one language receives it when the request names none,
so a Hebrew-only model is never left to auto-detect.

Lifecycle is the router's job, not the sub-engines'. It builds each sub-engine without
`idle_unload_s` and runs its own timers, because "unload" alone is not enough in `exclusive` mode:
when an on-demand model times out the router must also load the `keep_warm` models back. Sub-engines
only need the optional `preload`, `unload` and `loaded` members; one without them is simply left alone.

- `keep_warm` models load in `warm_up()`, which the CLI calls at start.
- On-demand models load on their first request. `idle_unload_s` re-arms a timer after every request.
- `exclusive = true` unloads every other model before serving one, which is what makes two models
  fit on a card that holds one of them plus a second program.
- A single lock serializes requests and lifecycle changes, so a timer firing mid-request cannot pull
  the model out from under a running transcription.

Combinations that cannot work are rejected when the config is read: an unknown `default`, a
language claimed by two models, `keep_warm` with `idle_unload_s`, and, in exclusive mode with a warm
model, an on-demand model without `idle_unload_s` (the warm model would never return).

There is deliberately no CPU fallback. A model that does not fit is a configuration problem to
solve with `exclusive` and on-demand loading, not a reason to silently run ten times slower.

## Wyoming service (`wyoming_server.py`)

Home Assistant's voice pipeline speaks the Wyoming protocol. `WyomingServer` is a small `Service`
(see `ports.py`): an object with `serve_forever()` and `close()` that `run_channels` runs next to the
channels in the same task group and closes in the same `finally`. A failure in any of them stops the
process, as for channels. It shares the engine, so voice notes and Home Assistant use the same models.

```
 telegram ─┐
 whatsapp ─┼─▶ run_channels ─▶ MultiSTT ─▶ en (warm) / he (on demand)
 wyoming  ─┘
```

Per connection: `Describe` is answered with one ASR model per `multi` model (name and languages);
`Transcribe` sets the language; `AudioChunk` PCM is buffered; `AudioStop` wraps the PCM as a WAV,
calls `engine.transcribe`, replies with `Transcript` and ends the connection. An engine error becomes
an empty transcript so the client does not wait for a timeout. The transcript text is never logged,
only its length.

The server uses `AsyncServer.start()` and waits on an event instead of `AsyncServer.run()`, because
`run()` installs its own SIGTERM handler that stops only the Wyoming listener and leaves the rest of
the process running.

`wyoming` is an optional dependency (the `wyoming` extra), imported only when a `[wyoming]` section is
present. A config with `[wyoming]` and no channels is valid.

## Sharing the model with other programs (`server.py`)

`kolnote serve` is a small HTTP server that exposes any `STTEngine` as OpenAI's
`/v1/audio/transcriptions`. It is for programs outside the process (a Wyoming bridge, another
kolnote with the `openai_compat` engine), not needed for several channels.

```
 other program ─▶ HTTP (OpenAI API) ─▶ kolnote serve ─▶ faster_whisper
```

The server is dependency-free, in the same style as the OpenWA webhook receiver: an asyncio
socket server, the standard library's `email` parser for multipart bodies, and an optional
bearer key.

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
- `kolnote serve` listens on `127.0.0.1` by default and compares the bearer key in constant
  time. It logs a warning when it listens on another address without a key. Error responses
  never include exception text, and audio and transcripts are neither stored nor logged.

## Why this shape

The first version served only WhatsApp. When Telegram was added, `pipeline.py` did not change:
the integration test `test_core_pipeline_serves_telegram_unchanged` runs the same `pipeline.run` against
a mocked Telegram API. That is the property to keep. If a new channel needs a core change, the
port is missing something, and the port should be fixed instead of special-casing the channel.
