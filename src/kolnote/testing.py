"""Conformance check for channel adapters. Use it in your adapter's test suite:

    async def test_my_channel():
        channel = MyChannel(...)
        async def deliver():  # make one voice note arrive on the channel
            ...
        await check_channel(channel, deliver)
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from .models import IncomingAudio
from .ports import Channel


async def check_channel(
    channel: Channel, deliver: Callable[[], Awaitable[None]], *, timeout_s: float = 10.0
) -> IncomingAudio:
    """Feed one voice note through `channel`, reply to it, close twice. Returns the message.

    `deliver` must cause exactly one voice note to arrive on the channel (for a webhook
    channel: POST it; for a polling channel: queue it on the fake upstream).
    """
    assert isinstance(channel, Channel), "missing a Channel method (messages, reply, close, from_config, name)"
    assert isinstance(channel.name, str) and channel.name, "channel.name must be a non-empty string"

    messages = channel.messages()
    first = asyncio.ensure_future(anext(messages))
    try:
        await asyncio.sleep(0.1)  # let the channel start listening before delivery
        await deliver()
        msg = await asyncio.wait_for(first, timeout_s)
    finally:
        if not first.done():
            first.cancel()

    assert isinstance(msg, IncomingAudio), f"expected IncomingAudio, got {type(msg).__name__}"
    assert msg.channel == channel.name, "IncomingAudio.channel must equal channel.name"
    assert msg.chat_id and msg.message_id, "chat_id and message_id must be non-empty"
    assert isinstance(msg.sender_id, str), "sender_id must be a string (use '' if unknown)"
    assert isinstance(msg.audio, bytes) and msg.audio, "audio must be non-empty bytes"

    await channel.reply(msg, "kolnote contract check")
    await channel.close()
    await channel.close()  # closing twice must not raise
    return msg
