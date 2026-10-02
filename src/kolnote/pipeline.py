from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence

from .models import Incoming, UnsupportedMessage
from .policy import Policy
from .ports import Channel, Service, STTEngine

log = logging.getLogger("kolnote")

NO_SPEECH = "(no speech detected)"
UNSUPPORTED = "I can only transcribe voice notes. Please send a voice message."
LOADING = "Loading the speech model, this note will take a few seconds longer."


async def handle(
    msg: Incoming,
    *,
    channel: Channel,
    engine: STTEngine,
    policy: Policy,
    language: str | None,
    error_text: str | None,
    unsupported_text: str | None = UNSUPPORTED,
    loading_text: str | None = None,
) -> None:
    reason = policy.check(msg)
    if reason:
        log.info("ignored %s/%s from %s: %s", msg.channel, msg.message_id, msg.sender_id, reason)
        return
    if isinstance(msg, UnsupportedMessage):
        log.info("unsupported %s/%s kind=%s from %s", msg.channel, msg.message_id, msg.kind, msg.sender_id)
        if unsupported_text:
            await channel.reply(msg, unsupported_text)
        return
    log.info(
        "received %s/%s from %s audio=%.1fs %d bytes",
        msg.channel, msg.message_id, msg.sender_id, msg.duration_s or 0, len(msg.audio),
    )
    needs_load = getattr(engine, "needs_load", None)
    if loading_text and needs_load and needs_load(language):
        try:
            await channel.reply(msg, loading_text)
        except Exception:
            log.exception("could not send the loading notice for %s/%s", msg.channel, msg.message_id)
    try:
        transcript = await engine.transcribe(msg.audio, mime_type=msg.mime_type, language=language)
    except Exception:
        log.exception("transcription failed for %s/%s", msg.channel, msg.message_id)
        if error_text:
            await channel.reply(msg, error_text)
        return
    await channel.reply(msg, transcript.text or NO_SPEECH)
    log.info(
        "replied %s/%s engine=%s stt=%.2fs total=%.2fs chars=%d%s",
        msg.channel, msg.message_id, transcript.engine, transcript.processing_s,
        time.monotonic() - msg.received_at, len(transcript.text), "" if transcript.text else " (empty)",
    )


async def run_channels(
    channels: Sequence[tuple[Channel, Policy]],
    engine: STTEngine,
    *,
    services: Sequence[Service] = (),
    language: str | None = None,
    concurrency: int = 1,
    error_text: str | None = "Transcription failed.",
    unsupported_text: str | None = UNSUPPORTED,
    loading_text: str | None = None,
) -> None:
    """Serve every channel and service with one shared engine. One of them failing stops them all."""
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(msg: Incoming, channel: Channel, policy: Policy) -> None:
        async with semaphore:
            try:
                await handle(
                    msg,
                    channel=channel,
                    engine=engine,
                    policy=policy,
                    language=language,
                    error_text=error_text,
                    unsupported_text=unsupported_text,
                    loading_text=loading_text,
                )
            except Exception:
                log.exception("unhandled error for %s/%s", msg.channel, msg.message_id)

    async def consume(channel: Channel, policy: Policy, tg: asyncio.TaskGroup) -> None:
        async for msg in channel.messages():
            tg.create_task(worker(msg, channel, policy))

    try:
        async with asyncio.TaskGroup() as tg:
            for channel, policy in channels:
                tg.create_task(consume(channel, policy, tg))
            for service in services:
                tg.create_task(service.serve_forever())
    finally:
        for channel, _ in channels:
            await channel.close()
        for service in services:
            await service.close()
        await engine.close()


async def run(
    channel: Channel,
    engine: STTEngine,
    policy: Policy,
    *,
    language: str | None = None,
    concurrency: int = 1,
    error_text: str | None = "Transcription failed.",
    unsupported_text: str | None = UNSUPPORTED,
    loading_text: str | None = None,
) -> None:
    await run_channels(
        [(channel, policy)],
        engine,
        language=language,
        concurrency=concurrency,
        error_text=error_text,
        unsupported_text=unsupported_text,
        loading_text=loading_text,
    )
