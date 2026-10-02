import asyncio

from kolnote.adapters.stt.faster_whisper import FasterWhisperSTT


class FakeModel:
    def transcribe(self, audio, **kw):
        class Seg:
            text = " hi "

        class Info:
            language, duration = "he", 1.0

        return [Seg()], Info()


def engine(idle):
    e = FasterWhisperSTT("m", idle_unload_s=idle)
    e.loads = 0

    def load():
        e.loads += 1
        return FakeModel()

    e._load = load
    return e


async def say(e):
    return await e.transcribe(b"x", mime_type="audio/ogg", language="he")


def test_model_unloads_after_idle_and_reloads_on_next_request():
    async def go():
        e = engine(0.05)
        assert (await say(e)).text == "hi"
        assert e._model is not None
        await asyncio.sleep(0.2)
        assert e._model is None
        assert (await say(e)).text == "hi"
        assert e.loads == 2
        await e.close()

    asyncio.run(go())


def test_activity_postpones_unload_and_zero_disables_it():
    async def go():
        e = engine(0.15)
        for _ in range(3):
            await say(e)
            await asyncio.sleep(0.08)
        assert e._model is not None and e.loads == 1
        await e.close()
        off = engine(0)
        await say(off)
        await asyncio.sleep(0.1)
        assert off._model is not None

    asyncio.run(go())
