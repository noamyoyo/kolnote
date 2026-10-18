from types import SimpleNamespace

from kolnote.adapters.stt import faster_whisper as fw
from kolnote.diarize import Turn, Word, build_dialogue, speaker_at


def words(*items):
    return [Word(s, e, t) for s, e, t in items]


def test_speaker_at_uses_containing_turn_then_nearest():
    turns = [Turn(0, 5, 0), Turn(10, 15, 1)]
    assert speaker_at(turns, 2) == 0
    assert speaker_at(turns, 12) == 1
    assert speaker_at(turns, 6) == 0  # gap, nearer to the first turn
    assert speaker_at(turns, 9) == 1  # gap, nearer to the second turn


def test_dialogue_labels_speakers_in_order_of_appearance():
    turns = [Turn(0, 5, 7), Turn(5, 10, 3)]
    ws = words((0, 1, " hello"), (1, 2, " there"), (5.5, 6, " yes"), (6, 7, " ok"))
    assert build_dialogue(ws, turns) == "A: hello there\n\nB: yes ok"


def test_dialogue_returns_to_first_speaker_with_same_letter():
    turns = [Turn(0, 2, 0), Turn(2, 4, 1), Turn(4, 6, 0)]
    ws = words((0, 1, " a"), (2.2, 3, " b"), (4.2, 5, " c"))
    assert build_dialogue(ws, turns) == "A: a\n\nB: b\n\nA: c"


def test_dialogue_pause_inside_one_speaker_starts_new_block():
    turns = [Turn(0, 20, 0)]
    ws = words((0, 1, " a"), (1.2, 2, " b"), (6, 7, " c"))
    assert build_dialogue(ws, turns, paragraph_gap_s=2) == "A: a b\n\nA: c"
    assert build_dialogue(ws, turns) == "A: a b c"


def test_dialogue_without_turns_or_words_degrades_to_plain_text():
    assert build_dialogue(words((0, 1, " a"), (1, 2, " b")), []) == "a b"
    assert build_dialogue([], [Turn(0, 1, 0)]) == ""


class FakeModel:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, audio, **kw):
        self.kwargs = kw
        w = lambda s, e, t: SimpleNamespace(start=s, end=e, word=t)  # noqa: E731
        seg = SimpleNamespace(
            text=" hi yes", start=0, end=4, words=[w(0, 1, " hi"), w(3, 4, " yes")]
        )
        return [seg], SimpleNamespace(language="he", duration=200.0)


class FakeDiarizer:
    def __init__(self, turns=None, error=None):
        self.turns, self.error = turns, error

    def diarize(self, samples):
        if self.error:
            raise self.error
        return self.turns


def engine(diarizer, duration, monkeypatch):
    stt = fw.FasterWhisperSTT("m", paragraph_gap_s=0)
    stt._model = FakeModel()
    stt._diarizer = diarizer
    stt._diarize_min_s = 120
    monkeypatch.setattr(fw, "probe_duration_s", lambda audio: duration)
    monkeypatch.setattr(fw, "decode_16k_mono", lambda audio: object())
    return stt


def test_long_recording_is_labelled_and_asks_for_word_timestamps(monkeypatch):
    stt = engine(FakeDiarizer([Turn(0, 2, 0), Turn(2, 4, 1)]), 200, monkeypatch)
    text, lang, dur = stt._run(b"x", "he")
    assert text == "A: hi\n\nB: yes"
    assert stt._model.kwargs["word_timestamps"] is True


def test_short_recording_is_not_diarized(monkeypatch):
    stt = engine(FakeDiarizer(error=AssertionError("must not run")), 30, monkeypatch)
    text, _, _ = stt._run(b"x", "he")
    assert text == "hi yes"
    assert stt._model.kwargs["word_timestamps"] is False


def test_single_speaker_result_stays_plain(monkeypatch):
    stt = engine(FakeDiarizer([Turn(0, 4, 0)]), 200, monkeypatch)
    assert stt._run(b"x", "he")[0] == "hi yes"


def test_diarization_failure_falls_back_to_plain_text(monkeypatch):
    stt = engine(FakeDiarizer(error=RuntimeError("boom")), 200, monkeypatch)
    assert stt._run(b"x", "he")[0] == "hi yes"


def test_overlapping_turns_give_the_short_reply_to_its_own_speaker():
    turns = [Turn(0, 20, 0), Turn(8, 9, 1)]  # a short reply inside a long turn
    assert speaker_at(turns, 5) == 0
    assert speaker_at(turns, 8.5) == 1
    ws = words((1, 2, " one"), (8.1, 8.9, " yes"), (10, 11, " two"))
    assert build_dialogue(ws, turns) == "A: one\n\nB: yes\n\nA: two"


def test_tiny_second_speaker_is_treated_as_a_single_speaker_monologue(monkeypatch):
    # B owns 1 of 40 words: the fixed speaker count split a monologue
    class Many(FakeModel):
        def transcribe(self, audio, **kw):
            self.kwargs = kw
            w = lambda s, e, t: SimpleNamespace(start=s, end=e, word=t)  # noqa: E731
            words = [w(i, i + 0.5, f" w{i}") for i in range(40)]
            seg = SimpleNamespace(text=" ".join(x.word for x in words), start=0, end=40, words=words)
            return [seg], SimpleNamespace(language="he", duration=200.0)

    stt = engine(FakeDiarizer([Turn(0, 39, 0), Turn(39, 41, 1)]), 200, monkeypatch)
    stt._model = Many()
    text = stt._run(b"x", "he")[0]
    assert "w39" in text and "A:" not in text and "B:" not in text


def test_turns_with_a_second_speaker_that_owns_no_words_stay_plain(monkeypatch):
    stt = engine(FakeDiarizer([Turn(0, 100, 0), Turn(100, 101, 1)]), 200, monkeypatch)
    assert stt._run(b"x", "he")[0] == "hi yes"


def test_recording_above_the_maximum_is_not_diarized(monkeypatch):
    stt = engine(FakeDiarizer(error=AssertionError("must not run")), 4000, monkeypatch)
    assert stt._run(b"x", "he")[0] == "hi yes"
    assert stt._model.kwargs["word_timestamps"] is False


def test_transcribe_error_does_not_wait_for_diarization(monkeypatch):
    import threading
    import time

    release = threading.Event()

    class Slow:
        def diarize(self, samples):
            release.wait(5)
            return [Turn(0, 1, 0)]

    class Broken(FakeModel):
        def transcribe(self, audio, **kw):
            raise RuntimeError("cuda oom")

    stt = engine(Slow(), 200, monkeypatch)
    stt._model = Broken()
    started = time.monotonic()
    try:
        stt._run(b"x", "he")
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
    finally:
        elapsed = time.monotonic() - started
        release.set()
    assert elapsed < 2


def test_speaker_shares_count_words_per_speaker():
    from kolnote.diarize import speaker_shares

    ws = [Word(0, 1, " a"), Word(1, 2, " b"), Word(2, 3, " c"), Word(10, 11, " d")]
    assert speaker_shares(ws, [Turn(0, 3, 0), Turn(9, 12, 1)]) == {0: 0.75, 1: 0.25}
    assert speaker_shares([], [Turn(0, 1, 0)]) == {}
