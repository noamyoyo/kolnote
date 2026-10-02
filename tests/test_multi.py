import asyncio

import pytest

from kolnote.adapters.stt.multi import MultiSTT
from kolnote.models import Transcript


class FakeEngine:
    def __init__(self, name):
        self.name = name
        self.loaded = False
        self.loads = 0
        self.seen_language = None

    async def preload(self):
        if not self.loaded:
            self.loaded, self.loads = True, self.loads + 1

    async def unload(self):
        self.loaded = False

    async def transcribe(self, audio, *, mime_type, language):
        await self.preload()
        self.seen_language = language
        return Transcript(text=self.name, engine=self.name, processing_s=0.0)

    async def close(self):
        self.loaded = False


def make(**kw):
    en, he = FakeEngine("en"), FakeEngine("he")
    opts = dict(
        default="en",
        languages={"en": ["en"], "he": ["he"]},
        keep_warm={"en"},
        idle_unload_s={"he": 0.05},
        exclusive=True,
    )
    opts.update(kw)
    return MultiSTT({"en": en, "he": he}, **opts), en, he


async def say(m, language):
    return (await m.transcribe(b"x", mime_type="audio/wav", language=language)).text


def test_routes_by_language_and_falls_back_to_default():
    async def go():
        m, en, he = make(exclusive=False, idle_unload_s={})
        assert await say(m, "he") == "he"
        assert await say(m, "en-US") == "en"
        assert await say(m, "fr") == "en"
        assert await say(m, None) == "en"

    asyncio.run(go())


def test_language_defaults_to_the_models_only_language_and_is_normalised():
    async def go():
        m, en, he = make(exclusive=False, idle_unload_s={})
        await say(m, None)
        assert en.seen_language == "en"
        await say(m, "en_US")
        assert en.seen_language == "en"

    asyncio.run(go())


def test_warm_up_loads_only_keep_warm_models():
    async def go():
        m, en, he = make()
        await m.warm_up()
        assert en.loaded and not he.loaded

    asyncio.run(go())


def test_needs_load_follows_the_routed_model():
    async def go():
        m, en, he = make()
        await m.warm_up()
        assert not m.needs_load("en") and m.needs_load("he") and not m.needs_load(None)
        await say(m, "he")
        assert not m.needs_load("he") and m.needs_load("en")
        await m.close()

    asyncio.run(go())


def test_exclusive_evicts_the_warm_model_and_brings_it_back_after_idle():
    async def go():
        m, en, he = make()
        await m.warm_up()
        await say(m, "he")
        assert he.loaded and not en.loaded
        await asyncio.sleep(0.25)
        assert en.loaded and not he.loaded
        assert en.loads == 2
        await m.close()

    asyncio.run(go())


def test_exclusive_swaps_back_immediately_when_the_warm_model_is_requested():
    async def go():
        m, en, he = make(idle_unload_s={"he": 30})
        await m.warm_up()
        await say(m, "he")
        await say(m, "en")
        assert en.loaded and not he.loaded
        await m.close()

    asyncio.run(go())


def test_non_exclusive_keeps_both_and_unloads_on_demand_model_when_idle():
    async def go():
        m, en, he = make(exclusive=False)
        await m.warm_up()
        await say(m, "he")
        assert en.loaded and he.loaded
        await asyncio.sleep(0.2)
        assert en.loaded and not he.loaded
        await m.close()

    asyncio.run(go())


def test_activity_postpones_the_idle_unload():
    async def go():
        m, en, he = make(idle_unload_s={"he": 0.15})
        for _ in range(3):
            await say(m, "he")
            await asyncio.sleep(0.08)
        assert he.loaded
        await m.close()

    asyncio.run(go())


@pytest.mark.parametrize(
    "kw, message",
    [
        (dict(default="zz"), "default model"),
        (dict(keep_warm={"en", "he"}, idle_unload_s={"he": 5}), "cannot be combined"),
        (dict(idle_unload_s={}), "exclusive mode"),
        (dict(languages={"en": ["en"], "he": ["en"]}), "claimed by both"),
    ],
)
def test_bad_combinations_are_rejected(kw, message):
    with pytest.raises(ValueError, match=message):
        make(**kw)


def test_from_config_builds_models_and_strips_routing_options():
    m = MultiSTT.from_config(
        {
            "default": "en",
            "exclusive": True,
            "models": {
                "en": {"type": "faster_whisper", "model": "tiny", "languages": ["en"], "keep_warm": True},
                "he": {"type": "faster_whisper", "model": "tiny", "languages": ["he"], "idle_unload_s": 300},
            },
        }
    )
    assert m.models == {"en": ["en"], "he": ["he"]}
    assert m.route("he") == "he" and m.route("en") == "en"
    assert m._models["he"]._idle_unload_s == 0  # the router owns the timer


def test_from_config_requires_models():
    with pytest.raises(ValueError, match="at least one"):
        MultiSTT.from_config({})
