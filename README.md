# kolnote

Voice note in, text out. A self-hosted speech-to-text bridge: someone sends a voice message,
kolnote transcribes it locally and replies in the same chat. The chat medium and the speech
engine are both swappable adapters, so nothing is tied to one messenger or one vendor.

Built and benchmarked for Hebrew (using the [ivrit.ai](https://huggingface.co/ivrit-ai) Whisper
models), but it works for any language Whisper supports. Audio never leaves your machine.

```
 Channel adapter  ->  Policy  ->  STT adapter  ->  Channel adapter (reply)
 WhatsApp, folder,    allowlist   faster-whisper,   same chat, as plain text
 your own ...                     HTTP API, ...
```

The core (`pipeline.py`) only knows the two contracts in `ports.py`. Adding a messenger or an
engine means writing one small class.

## Status

Proof of concept, used daily on one setup. Working today:

- Channels: `openwa` (WhatsApp through an [OpenWA](#whatsapp-with-openwa) gateway, tested live), `telegram` (Bot API, [see below](#telegram), tested live), `folder` (drop files in, get `.txt` out).
- STT engines: `faster_whisper` (local, GPU or CPU) and `openai_compat` (any `/v1/audio/transcriptions` server).
- Several channels in one process sharing one model, with optional idle unload to free GPU memory, see [Several channels, one model](#several-channels-one-model).
- Several models in one process (`type = "multi"`), chosen by language, with always-warm and on-demand models, see [Several models, one GPU](#several-models-one-gpu).
- A Wyoming speech-to-text server, so Home Assistant can use the same process, see [Home Assistant (Wyoming)](#home-assistant-wyoming).
- `kolnote serve`: expose the model to other programs over the OpenAI transcription API, see [Sharing the model with other programs](#sharing-the-model-with-other-programs).
- Deny-by-default allowlist, benchmark tool (WER/CER/speed), Docker image.

Not done yet: measuring audio length when the gateway does not report it. How it is built is in
[docs/DESIGN.md](docs/DESIGN.md).

## Quick start (no chat account needed)

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[faster-whisper,dev]'
cp config.example.toml config.toml
.venv/bin/kolnote run -c config.toml      # drop audio files into ./inbox, get <name>.txt back
```

The first run downloads the model from Hugging Face. Run the tests with `.venv/bin/pytest`.

## Activating a channel

A config with a `[channel]` section runs one channel. To run several (say WhatsApp and Telegram)
in one process, with one copy of the model, see [Several channels, one model](#several-channels-one-model).

Every channel needs an allowlist before it answers anyone (see [Access control](#access-control)).
With an empty `[policy]` the bot ignores everything.

### WhatsApp with OpenWA

kolnote talks to WhatsApp through an OpenWA gateway (a self-hosted REST/webhook layer over the
Baileys WhatsApp Web engine). kolnote receives webhooks and replies over REST.

Install the HTTP extra first: `pip install -e '.[faster-whisper,http]'`.

1. Run OpenWA and pair a WhatsApp number with a session. Use a separate number or account for
   the bot, not your personal one.
2. Create a webhook in OpenWA for the `message.received` event, pointing at
   `http://<kolnote-host>:8765/webhook`, with a secret of your choice. Optionally filter it to
   the chats you want. OpenWA signs each request with `X-OpenWA-Signature: sha256=<hmac>`;
   kolnote rejects anything unsigned or wrongly signed, and refuses to start without a secret.
3. Create an API key for kolnote. Scope it to the one session and the chats it should serve
   (`allowedSessions`, `allowedChats`), not your master key.
4. Copy `configs/openwa.example.toml`, fill in the gateway URL, session id and the chat ids in
   `allow_chats` (group ids look like `1203...@g.us`), and set the two secrets as environment
   variables (never in the file):

```bash
export KOLNOTE_CHANNEL__API_KEY=...
export KOLNOTE_CHANNEL__WEBHOOK_SECRET=...
kolnote run -c configs/openwa.toml
```

Behavior in an allowed chat:

- A voice note gets a read receipt, then the transcript as a text reply.
- Any other message (text, image, sticker, a file sent as a document, ...) gets a short "I can
  only transcribe voice notes" reply. Change the wording with `unsupported_text`, or set it to
  `""` to stay silent.
- The bot's own messages and non-message events are ignored, so it cannot reply to itself.
- Repeated deliveries of the same webhook are deduplicated.

Caveats:

- **Docker networking.** If kolnote runs in Docker, the gateway's public hostname often does
  not resolve from inside the container. Put both on one Docker network and set
  `KOLNOTE_CHANNEL__BASE_URL=http://<gateway-container-name>:<port>`. A wrong URL shows up as
  `ConnectError: Name or service not known` when the bot tries to reply.
- **Unofficial connection.** Baileys-based gateways use an unofficial WhatsApp Web connection.
  Use a dedicated number; WhatsApp's terms do not endorse this.
- **No duration limit.** The gateway does not report audio length, so `max_duration_s` is not
  enforced on WhatsApp. Log lines show `audio=0.0s` for this reason.
- **Plain replies.** The reply is sent as plain text, not as a quoted reply, because a
  chat-restricted OpenWA key is not allowed to quote.
- **Webhook port.** Publish port 8765 only to a network the gateway can reach, never the open
  internet. The signature check is the only thing between that port and the bot.

### Telegram

1. Open @BotFather in Telegram, send `/newbot`, and choose a name and a username ending in
   `bot`. BotFather replies with a token.
2. Find the ids you want to allow. For a private chat use your own numeric user id (ask
   @userinfobot). For a group, use the group id, which is negative (for example
   `-100123456789`).
3. Copy `configs/telegram.example.toml` and put those ids in `allow_senders` (people) or
   `allow_chats` (chats; a group here admits everyone in it).
4. Pass the token as an environment variable (never in the file) and run:

```bash
export KOLNOTE_CHANNEL__TOKEN=123456:ABC...
kolnote run -c configs/telegram.toml
```

Then open a chat with the bot, press Start, and send a voice note. Telegram bots cannot message
you first, so the chat has to exist before the bot can answer.

The bot long-polls Telegram, so it needs no public URL or open port. It replies to the voice
note it is answering, shows a "typing" indicator while it works, and splits transcripts longer
than 4096 characters. Text, photos, stickers and other non-voice messages get the "I can only
transcribe voice notes" reply. `max_duration_s` works on Telegram because it reports the length.

Caveats:

- **Group privacy.** In groups the bot only sees voice notes if privacy mode is off
  (BotFather `/setprivacy`, then Disable) or the bot is a group admin. Remove and re-add the bot
  to the group after changing it.
- **One poller per token.** Two processes polling the same bot, or a webhook set on the bot,
  make Telegram answer with a conflict error and nobody gets updates reliably.
- **The token is a password.** It is part of every Telegram API URL. kolnote turns off HTTP
  request logging and never logs exception text for Telegram calls, but do not enable debug
  logging of HTTP traffic. If a token leaks, send `/revoke` to BotFather and use the new one.
- **Not end-to-end encrypted.** Bot chats are ordinary cloud chats, so Telegram can read the
  audio. Transcription still runs locally; the audio only leaves your server as the user's own
  message to Telegram.
- **Size limit.** Bots can download files up to 20 MB.

## Several channels, one model

Name each channel under `[channels.<name>]` instead of `[channel]`. One `kolnote run` process
serves all of them with a single model:

```
 telegram ──┐
 whatsapp ──┼──▶ kolnote run ──▶ faster-whisper (one copy in memory)
 folder   ──┘
```

```toml
language = "he"

[channels.telegram]                     # type defaults to the name
[channels.telegram.policy]
allow_senders = ["123456789"]

[channels.whatsapp]
type = "openwa"
base_url = "http://openwa:2785"
session_id = "..."
[channels.whatsapp.policy]
allow_chats = ["1203...@g.us"]

[stt]
type = "faster_whisper"
model = "ivrit-ai/whisper-large-v3-turbo-ct2"
compute_type = "int8_float16"
idle_unload_s = 300
```

See [`configs/multi.example.toml`](configs/multi.example.toml). Secrets stay in the environment,
for example `KOLNOTE_CHANNELS__TELEGRAM__TOKEN` and `KOLNOTE_CHANNELS__WHATSAPP__API_KEY`. Each
channel has its own `policy` (a channel without one uses the top-level `[policy]`). Do not mix
`[channel]` and `[channels.*]` in one config.

- **One request at a time.** The model transcribes one note at a time, so notes from different
  channels queue. A short note takes about a second once the model is loaded.
- **One failure stops all.** If one channel's connection dies for good, the process exits and
  Docker's restart policy brings it back. Run separate containers if you want them isolated.
- **Sharing a GPU.** `idle_unload_s` frees the model and its VRAM after that many seconds without
  a note. The next note reloads it, which takes 5 to 30 s depending on the disk. Use it when
  something else (another model, a voice assistant) needs the card most of the time. The default
  is 0: keep the model loaded.

[`docker-compose.example.yml`](docker-compose.example.yml) runs channels, the Wyoming server and
several models as one container (see [Docker](#docker)).

## Several models, one GPU

One model rarely fits everything. A Hebrew fine-tune is good at Hebrew and poor at English; the
stock model is the other way round. `type = "multi"` holds several models and picks one per
request by language. It also decides which models stay in memory, so a small GPU can serve both.

```toml
[stt]
type = "multi"
default = "en"          # used when the language matches no model
exclusive = true        # keep at most one model in memory (see below)

[stt.models.en]
type = "faster_whisper"
model = "large-v3-turbo"
compute_type = "int8_float16"
languages = ["en"]
keep_warm = true        # loaded at start, reloaded whenever it is evicted

[stt.models.he]
type = "faster_whisper"
model = "ivrit-ai/whisper-large-v3-turbo-ct2"
compute_type = "int8_float16"
languages = ["he"]
idle_unload_s = 300     # loaded on the first Hebrew note, dropped after 5 idle minutes
```

See [`configs/multi-model.example.toml`](configs/multi-model.example.toml) for a complete file with
channels. Each `[stt.models.<name>]` is a normal engine config (any `type`, any option) plus the
routing keys below.

| Key | Where | Meaning |
|-----|-------|---------|
| `default` | `[stt]` | Model used when the request's language matches no model, or has none. Defaults to the first model in the file. |
| `exclusive` | `[stt]` | `true`: at most one model in memory at a time. Default `false`. |
| `languages` | model | Language codes this model serves. `en-US` and `en_US` count as `en`. One code per language, across all models. |
| `keep_warm` | model | Load at start and keep loaded. |
| `idle_unload_s` | model | Unload after this many seconds without a request. The model loads again on demand. |

How it behaves:

- **Routing.** The language comes from the channel's top-level `language`, from the Wyoming
  client, or from a request. A model with exactly one language uses it when the request names none.
- **Always warm, on demand.** `keep_warm` models are loaded when the process starts. Other models
  load on their first request and unload when their `idle_unload_s` runs out. A model with
  neither is loaded on demand and stays loaded.
- **Exclusive mode.** Serving a request for one model first unloads the others. When an
  on-demand model times out, the router unloads it and loads the `keep_warm` models back, so the
  card returns to its resting state without a request. When at least one model is `keep_warm`,
  every other model needs an `idle_unload_s`, otherwise nothing would ever bring the warm model
  back. Use it when the models together do not fit next to whatever else uses the GPU.
- **The first request after a swap is slow.** It waits for the load: a few seconds with the
  weights in the OS page cache, up to about 30 s from a cold disk.
- **One request at a time.** The router serializes requests and model changes.
- **Mistakes fail at start.** An unknown `default`, a language claimed by two models, `keep_warm`
  together with `idle_unload_s`, or the exclusive-mode rule above stops the process with a message.

With `multi`, set `idle_unload_s` per model. Under `[stt]` itself only `default`, `exclusive` and
`models` are accepted; anything else stops the process with an error.

## Home Assistant (Wyoming)

Home Assistant's voice pipeline talks to speech-to-text servers over the
[Wyoming protocol](https://github.com/OHF-Voice/wyoming). A `[wyoming]` section makes `kolnote
run` listen as one, in the same process and with the same engine as the chat channels. A note
that arrives over WhatsApp and a question spoken to Home Assistant use one set of models.

```toml
[wyoming]
host = "0.0.0.0"      # default 127.0.0.1
port = 10300          # default 10300
```

Install the extra (`pip install -e '.[faster-whisper,wyoming]'`; the `full` Docker target
includes it). In Home Assistant, add the Wyoming Protocol integration with the host and port, then
choose it as the speech-to-text engine of a voice assistant.

- **What it advertises.** With `type = "multi"`, one model per `[stt.models.*]` entry, with its
  languages. Home Assistant sends the assistant's language and kolnote routes on it. With a
  single engine, set `languages = ["en"]` under `[wyoming]` (or the top-level `language`).
- **Audio.** Home Assistant streams raw PCM. kolnote wraps it as WAV and passes it to the engine;
  the reply is the transcript.
- **Failures.** If the engine fails, Home Assistant gets an empty transcript instead of a hang.
- **No authentication.** Wyoming has none. Bind to a trusted network only.
- **Latency matters here.** A voice assistant waits for the answer. Keep the model Home Assistant
  uses in `keep_warm`, and put slower, rarer models (a large Hebrew model, say) on demand.
  A request that needs a cold load can be slower than the client's timeout allows.
- **Only `[wyoming]` is enough to run.** A config with `[wyoming]` and no channels is valid.
- **A `multi` setup for Home Assistant plus voice notes** (English warm for the assistant, Hebrew
  on demand for notes, exclusive on a small GPU) is the example in
  [`configs/multi-model.example.toml`](configs/multi-model.example.toml).

### Connect Home Assistant, step by step

1. **Run kolnote** with a `[wyoming]` section and `docker compose up -d` (the compose example
   publishes port 10300 on loopback).
2. **Add the integration.** Home Assistant: Settings, Devices & services, Add integration,
   Wyoming Protocol. Host and port of kolnote, for example `127.0.0.1` and `10300`. Home Assistant
   in Docker with `network_mode: host` reaches a loopback port; on another host publish the port
   on the LAN address, or put both containers on one Docker network and use the service name.
   The integration appears as `kolnote` and adds one speech-to-text entity, `stt.kolnote`.
3. **Pick it in a voice assistant.** Settings, Voice assistants, open an assistant (or add one),
   set **Speech-to-text** to `kolnote` and choose the **Language**. The language you choose is
   the language kolnote receives, and a `multi` setup routes on it: English goes to the model
   with `languages = ["en"]`, Hebrew to the one with `["he"]`. A language no model lists falls
   back to `default`.
4. **One assistant per language** is the simplest way to use both models: for example "Home" with
   speech-to-text language English and a second one with Hebrew. Each assistant keeps its own
   wake word, conversation agent and text-to-speech voice. Pick one when you start talking (the
   assistant selector in the app, or the satellite device setting).
5. **Test.** Settings, Voice assistants, the microphone icon on the assistant. kolnote logs a
   line like `transcribed 3.0s of audio in 0.27s, 22 chars, language=en` per request, with no
   transcript text.

Speech-to-text is only the first stage. A Hebrew assistant also needs a conversation agent that
understands Hebrew (an LLM works; the built-in intents have limited Hebrew) and a text-to-speech
voice for Hebrew. The English assistant works with the built-in intents as is.

## Sharing the model with other programs

`kolnote serve` exposes the `[stt]` engine over the OpenAI transcription API, so other programs
(or other kolnote processes with `[stt] type = "openai_compat"`) can use one loaded model. You do
not need it for the setup above. Start it with
`STT_API_KEY=... kolnote serve -c configs/server.example.toml`. It loads the model at start
(`preload = true`) and answers `POST /v1/audio/transcriptions` and `GET /healthz`. The `server`
Docker target runs it.

Server options, in the `[server]` section (or `KOLNOTE_SERVER__<KEY>`). The engine comes from
`[stt]`, and the top-level `language` is the default when a request does not send one:

| Key | Default | Meaning |
|-----|---------|---------|
| `host` | `127.0.0.1` | Address to listen on. Use `0.0.0.0` only in a container or on a trusted network. |
| `port` | `8000` | Port. |
| `preload` | `true` | Load the model at start instead of on the first request. |
| `api_key_env` | none | Name of the environment variable that holds the API key. Without it there is no authentication. |
| `max_body_mb` | `64` | Largest accepted upload. |

What to know:

- **Authentication.** With `api_key_env` set, requests need `Authorization: Bearer <key>`. The
  key stops other machines or containers that can reach the port from using your GPU. It is
  optional on `127.0.0.1` or on a private container network with no published port. If the
  server listens on anything else without a key it logs a warning. `/healthz` never needs a key.
- **One request at a time.** Simultaneous requests queue.
- **Shared settings.** Beam size, VAD, prompt and language default live on the server and apply
  to every client. A request can override only `language`.
- **If the server is down** a kolnote client replies "Transcription failed." (`error_text`) and
  keeps running.
- **API compatibility.** The server implements `file`, `language` and `response_format`
  (`json` or `text`) of OpenAI's transcription endpoint. `model` is accepted and ignored: it
  always uses the engine it started with.

`openai_compat` is not tied to `kolnote serve`. In principle it works against any server that
speaks the same endpoint (OpenAI, Groq, a whisper.cpp server, Speaches). Only `kolnote serve`
has been tested here.

## Docker

```bash
docker build -t kolnote .
docker run -d --name kolnote --gpus all \
  -p 8765:8765 \
  -e KOLNOTE_CHANNEL__API_KEY -e KOLNOTE_CHANNEL__WEBHOOK_SECRET \
  -v "$PWD/models:/models" -v "$PWD/configs:/configs:ro" \
  kolnote run -c /configs/openwa.toml
```

`--gpus all` needs the NVIDIA Container Toolkit. Model files are cached in `/models`, so mount
a volume there. If the gateway runs in Docker too, put both on the same network and use the
gateway's container name as `base_url`. Publish port 8765 only to a network the gateway can
reach, not the open internet.

For Telegram, drop the `-p` flag (it polls outbound), pass `-e KOLNOTE_CHANNEL__TOKEN`, and use a
Telegram config. For several channels in one container, use a `[channels.*]` config (see
[Several channels, one model](#several-channels-one-model)) or the compose example.

The Dockerfile has three targets. `docker build .` builds the last one, `full`: the bot with a
local model, as above.

| Target | Contains | Use for |
|--------|----------|---------|
| `full` (default) | bots, Wyoming server, faster-whisper, CUDA libraries | one container with every channel, the model and Home Assistant's speech-to-text |
| `server` | faster-whisper, CUDA libraries, runs `kolnote serve` | exposing the model to other programs |
| `channel` | bots and `httpx` only, no model, no GPU | channels that call a remote STT server |

Build one with `docker build --target channel -t kolnote-channel .`. With `[wyoming]` enabled,
publish its port too, for example `-p 10300:10300`.

## Configuration

A TOML file, or environment variables, or both. Environment wins. Every key maps to
`KOLNOTE_<SECTION>__<KEY>`, for example:

```bash
KOLNOTE_STT__MODEL=large-v3
KOLNOTE_POLICY__ALLOW_CHATS="1203...@g.us,1204...@g.us"   # lists are comma separated
KOLNOTE_LANGUAGE=he
```

Secrets belong in environment variables. See `config.example.toml` and `configs/`.

| Key | Meaning |
|-----|---------|
| `language` | Whisper language code such as `he` or `en`. Omit to auto-detect. |
| `concurrency` | Notes transcribed in parallel (default 1; the model runs one at a time anyway). |
| `unsupported_text` | Reply for non-voice messages. Empty string disables it. |
| `loading_text` | Reply sent before the transcript when the note has to wait for a model to load (cold start). Empty string disables it. |
| `[channel]` / `[stt]` | `type` plus that adapter's options. |
| `[channels.<name>]` | Several channels at once, each with its own optional `policy`. Replaces `[channel]`. |
| `[wyoming]` | Also listen as a Wyoming speech-to-text server, see [Home Assistant (Wyoming)](#home-assistant-wyoming). Keys: `host`, `port`, `languages`. |
| `[server]` | Only for `kolnote serve`, see [Sharing the model with other programs](#sharing-the-model-with-other-programs). |
| `[policy]` | Who may use the bot, see below. |

## Access control

Deny-by-default. A message is accepted if its chat is in `allow_chats` **or** its sender is in
`allow_senders`. Listing a group in `allow_chats` therefore admits everyone in that group. Set
`open = true` only if you really want anyone. `max_duration_s` rejects long notes.

## Adapters

| Kind | `type` | Options |
|------|--------|---------|
| channel | `openwa` | `base_url`, `session_id`, `api_key`, `webhook_secret`, `host`, `port`, `path`, `timeout_s`, `mark_read` |
| channel | `telegram` | `token`, `api_base`, `poll_timeout_s`, `typing` |
| channel | `folder` | `inbox`, `poll_s`, `once` |
| stt | `faster_whisper` | `model` (any CTranslate2 model id or path), `device`, `compute_type`, `beam_size`, `vad_filter`, `initial_prompt`, `cpu_threads`, `download_root` |
| stt | `openai_compat` | `base_url`, `model`, `api_key_env`, `timeout_s` (needs `.[http]`) |
| stt | `multi` | `default`, `exclusive`, and `[stt.models.<name>]` tables, see [Several models, one GPU](#several-models-one-gpu) |

Write your own, for another messenger or a second WhatsApp gateway: see
[docs/ADDING-A-CHANNEL.md](docs/ADDING-A-CHANNEL.md). A channel is one small class; the core
does the allowlist, transcription and reply. Select it with a built-in name, a package entry
point (`kolnote.channels`), or `type = "yourpkg.module:YourClass"`. `kolnote.testing.check_channel`
checks your adapter against the contract.

## Benchmarks

Scored on [`ivrit-ai/eval-whatsapp`](https://huggingface.co/datasets/ivrit-ai/eval-whatsapp):
54 real Hebrew WhatsApp voice notes with reference transcripts. RTX 3080, `int8_float16`, beam
size 5, power-capped to 150 W.

| Model | WER | CER | Real-time factor |
|-------|-----|-----|------------------|
| `large-v3-turbo` (stock Whisper) | 13.4% | 5.9% | 0.03 |
| `ivrit-ai/whisper-large-v3-turbo-ct2` | 7.6% | 3.6% | 0.04 |
| `ivrit-ai/whisper-large-v3-ct2` | 6.8% | 3.3% | 0.27 |

WER is word error rate and CER is character error rate (lower is better), computed over the
whole corpus after stripping Hebrew vowel points and punctuation. Real-time factor is
processing time divided by audio length, so 0.04 means 25 times faster than real time.

The Hebrew fine-tune roughly halves the error rate against stock Whisper at the same speed.
The full large-v3 model is a little more accurate but about seven times slower, so the turbo
model is the default. In live use the turbo model transcribed an 8 s note in about 0.7 s once
loaded. Without `preload` the first note after a start takes 25 to 35 s because the model loads
lazily; `kolnote serve` preloads by default. With `idle_unload_s` the same delay applies after
every idle period.

Run it on your own audio (audio files plus same-named `.txt` references):

```bash
kolnote bench --dataset datasets/my-set \
  --stt configs/stock-turbo.toml --stt configs/ivrit-turbo.toml --out bench-results/run.json
```

The eval-whatsapp dataset is gated and licensed for training or academic research. It is not
included here; request access on Hugging Face yourself.

## Hardware

- The turbo model at `int8_float16` uses about 1.3 GB of VRAM. Measured on an RTX 3080.
- GPUs older than Volta (for example a GTX 1080 Ti) have no tensor cores. Use
  `compute_type = "int8"` there. This is expected to work but has not been tested.
- CPU works (`device = "cpu"`) but is much slower.

`scripts/gpu-guard.sh <container>` is an optional watchdog we use on a card with weak cooling:
it pauses a container at 78 C, resumes at 70 C, and kills it at 88 C or on a GPU error. Adjust
the thresholds to your hardware before using it.

If the card is shared with another model, set `idle_unload_s` so kolnote gives its VRAM back when
idle. Two models that together fill the card can make either one fail to load or fall back to the CPU.

## Privacy

Transcription runs locally. The only network traffic is the model download from Hugging Face,
the calls to your own gateway, and, if you use `kolnote serve` or `openai_compat`, the calls to
that server (which carry the audio, so keep that link on a private network). Transcript text is not written to the logs; they record
sender, audio length, timings and character counts. Voice notes are processed in memory and not
stored.

## License

MIT, see `LICENSE`. The ivrit.ai models and any dataset you use have their own licenses.
