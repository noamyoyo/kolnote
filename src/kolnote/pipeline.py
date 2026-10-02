from __future__ import annotations

import asyncio
import logging
import time

from .models import Incoming, UnsupportedMessage
from .policy import Policy
from .ports import Channel, STTEngine

log = logging.getLogger("kolnote")

NO_SPEECH = "(no speech detected)"
UNSUPPORTED = "I can only transcribe voice notes. Please send a voice message."


async def handle(
    msg: Incoming,
    *,
    channel: Channel,
    engine: STTEngine,
    policy: Policy,
    language: str | None,
    error_text: str | None,
    unsupported_text: str | None = UNSUPPORTED,
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


async def run(
    channel: Channel,
    engine: STTEngine,
    policy: Policy,
    *,
    language: str | None = None,
    concurrency: int = 1,
    error_text: str | None = "Transcription failed.",
    unsupported_text: str | None = UNSUPPORTED,
) -> None:
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(msg: Incoming) -> None:
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
                )
            except Exception:
                log.exception("unhandled error for %s/%s", msg.channel, msg.message_id)

    try:
        async with asyncio.TaskGroup() as tg:
            async for msg in channel.messages():
                tg.create_task(worker(msg))
    finally:
        await channel.close()
        await engine.close()
