from types import SimpleNamespace

from kolnote.adapters.stt import faster_whisper as fw
from kolnote.dialogue import MergeOptions, assign_words, build_dialogue, label_words, render_dialogue, speaker_shares
from kolnote.diarize import Turn, Word


def words(*items):
    return [Word(s, e, t) for s, e, t in items]


def test_dialogue_labels_speakers_in_order_of_appearance():
    turns = [Turn(0, 5, 7), Turn(5, 10, 3)]
    ws = words((0, 1, " hello"), (1, 2, " there"), (5.5, 6, " yes"), (6, 7, " ok"))
    assert build_dialogue(ws, turns) == "A: hello there\n\nB: yes ok"


def test_dialogue_returns_to_first_speaker_with_same_letter():
    turns = [Turn(0, 2, 0), Turn(2, 4, 1), Turn(4, 6, 0)]
    ws = words((0, 1, " a."), (2.2, 3, " b"), (4.2, 5, " c"))
    assert build_dialogue(ws, turns) == "A: a.\n\nB: b\n\nA: c"


def test_dialogue_pause_inside_one_speaker_starts_new_block():
    ws = words((0, 1, " a"), (1.2, 2, " b"), (6, 7, " c"))
    assert render_dialogue(ws, [0, 0, 0], paragraph_gap_s=2) == "A: a b\n\nA: c"
    assert render_dialogue(ws, [0, 0, 0]) == "A: a b c"


def test_not_a_conversation_gives_none():
    assert build_dialogue(words((0, 1, " a"), (1, 2, " b")), []) is None
    assert build_dialogue([], [Turn(0, 1, 0)]) is None
    assert build_dialogue(words((0, 1, " a"), (1, 2, " b")), [Turn(0, 5, 0)]) is None


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


def test_noise_speaker_is_dropped_and_real_speakers_are_kept(monkeypatch):
    class Many(FakeModel):
        def transcribe(self, audio, **kw):
            self.kwargs = kw
            w = lambda s, e, t: SimpleNamespace(start=s, end=e, word=t)  # noqa: E731
            words = [w(i, i + 0.5, f" w{i}") for i in range(60)]
            seg = SimpleNamespace(text="x", start=0, end=60, words=words)
            return [seg], SimpleNamespace(language="he", duration=200.0)

    # A talks 0-30, B 30-58, a third "speaker" owns one word at 58-60
    stt = engine(FakeDiarizer([Turn(0, 30, 0), Turn(30, 58, 1), Turn(58, 60, 2)]), 200, monkeypatch)
    stt._model = Many()
    text = stt._run(b"x", "he")[0]
    assert text.startswith("A: w0") and "\n\nB: w30" in text
    assert "C:" not in text and text.rstrip().endswith("w59")


def test_diarizer_auto_mode_uses_threshold_and_forced_mode_uses_count():
    from kolnote.diarize import SherpaDiarizer

    auto = SherpaDiarizer("seg", "emb")
    assert auto._num_speakers == -1 and auto._threshold == 0.8
    forced = SherpaDiarizer("seg", "emb", num_speakers=2, threshold=0.5)
    assert forced._num_speakers == 2 and forced._threshold == 0.5


def test_dialogue_uses_custom_labels_then_falls_back_to_letters():
    turns = [Turn(0, 2, 0), Turn(2, 4, 1), Turn(4, 6, 2)]
    ws = words((0.5, 1, " a"), (2.5, 3, " b"), (4.5, 5, " c"))
    assert build_dialogue(ws, turns, labels=["אדם א'", "אדם ב'"]) == "אדם א': a\n\nאדם ב': b\n\nA: c"


def test_speaker_labels_option_reaches_the_dialogue(monkeypatch):
    stt = engine(FakeDiarizer([Turn(0, 2, 0), Turn(2, 4, 1)]), 200, monkeypatch)
    stt._speaker_labels = ["אדם א'", "אדם ב'"]
    assert stt._run(b"x", "he")[0] == "אדם א': hi\n\nאדם ב': yes"


def test_diarize_options_parse_speaker_labels():
    stt = fw.FasterWhisperSTT(
        "m", diarize={"segmentation": "s", "embedding": "e", "speaker_labels": ["x", "y"], "min_speaker_share": 0.1}
    )
    assert stt._speaker_labels == ["x", "y"] and stt._min_speaker_share == 0.1


def test_labels_that_repeat_do_not_merge_two_speakers():
    turns = [Turn(0, 2, 0), Turn(2, 4, 1)]
    ws = words((0.5, 1, " a"), (2.5, 3, " b"))
    assert build_dialogue(ws, turns, labels=["B"]) == "B: a\n\nA: b"  # the fallback skips the letter a custom label already uses


def test_speaker_labels_are_validated():
    import pytest

    for bad in ("אדם", ["a", "a"], ["a", ""], [1, 2], None):
        with pytest.raises(ValueError):
            fw._speaker_labels(bad)
    assert fw._speaker_labels([" אדם א' ", "אדם ב'"]) == ["אדם א'", "אדם ב'"]
    assert fw._speaker_labels([]) == []


def test_continuation_rule_does_not_glue_words_in_scripts_without_spaces():
    turns = [Turn(0, 5, 0), Turn(5, 10, 1)]
    ws = [Word(0, 1, "你好"), Word(6, 7, "再见")]  # no leading spaces, but real words
    assert build_dialogue(ws, turns) == "A: 你好\n\nB: 再见"


def test_a_short_reply_inside_a_long_turn_goes_to_its_own_speaker():
    turns = [Turn(0, 20, 0), Turn(8, 9, 1)]  # a long turn blankets the reply
    ws = words((1, 2, " one."), (8.1, 8.9, " yes."), (10, 11, " two."))
    assert build_dialogue(ws, turns) == "A: one.\n\nB: yes.\n\nA: two."


def test_word_timing_jitter_is_tolerated_by_the_coverage_rule():
    turns = [Turn(0, 20, 0), Turn(10.0, 10.4, 1)]
    o = MergeOptions()
    assert assign_words(words((10.1, 10.6, " yes")), turns, o) == [1]  # 60% covered by the short turn
    assert assign_words(words((10.3, 10.9, " yes")), turns, o) == [0]  # only 17% covered


def test_interjection_in_an_unfinished_sentence_is_shown_inline():
    turns = [Turn(0, 20, 0), Turn(8, 9, 1)]
    ws = words((1, 2, " one"), (8.1, 8.9, " yes"), (10, 11, " two"), (12, 13, " three"), (14, 15, " four"))
    assert build_dialogue(ws, turns) == "A: one (B: yes) two three four"
    assert build_dialogue(ws, turns, options=MergeOptions(inline_max_words=0)) == "A: one\n\nB: yes\n\nA: two three four"


def test_interjection_after_a_finished_sentence_or_longer_than_the_limit_stays_a_block():
    turns = [Turn(0, 30, 0), Turn(8, 12, 1)]
    ws = words((1, 2, " one."), (8.1, 8.5, " yes"), (13, 14, " two"), (15, 16, " three"), (17, 18, " four"))
    assert build_dialogue(ws, turns) == "A: one.\n\nB: yes\n\nA: two three four"
    ws2 = words((1, 2, " one"), (8.1, 8.3, " a"), (8.4, 8.6, " b"), (8.7, 8.9, " c"), (9, 9.5, " d"), (13, 14, " two"))
    assert "(B:" not in build_dialogue(ws2, [Turn(0, 30, 0), Turn(8, 10, 1)])


def test_number_is_not_split_from_the_word_after_it():
    turns = [Turn(0, 10, 0), Turn(5.0, 5.4, 1)]
    ws = words((4.8, 5.0, " 48"), (5.05, 5.35, " months"), (7, 8, " done"))
    assert label_words(ws, turns, MergeOptions()) == [0, 0, 0]
    assert label_words(ws, turns, MergeOptions(num_gap_s=0)) == [0, 1, 0]


def test_token_without_leading_space_stays_with_the_previous_word():
    # "10" + ".377" + "%": a tiny overlap turn of the other speaker sits on ".377"
    turns = [Turn(0, 10, 0), Turn(5.0, 5.2, 1)]
    ws = words((4.0, 4.5, " 10"), (5.0, 5.2, ".377"), (5.3, 5.5, "%"), (6, 7, " done"))
    assert label_words(ws, turns) == [0, 0, 0, 0]
    assert build_dialogue(words((4.0, 4.5, " 10."), (5.0, 5.2, " yes")), turns) == "A: 10.\n\nB: yes"


def test_speaker_shares_count_words_per_speaker():
    assert speaker_shares([0, 0, 0, 1]) == {0: 0.75, 1: 0.25}
    assert speaker_shares([]) == {}


def test_every_word_survives_the_merge_exactly_once():
    import random
    import re

    rnd = random.Random(7)
    for _ in range(60):
        t, ws = 0.0, []
        for i in range(rnd.randint(5, 80)):
            t += rnd.choice([0.0, 0.05, 0.2, 0.9, 2.5])
            d = rnd.uniform(0.15, 0.7)
            ws.append(Word(t, t + d, (" " if rnd.random() > 0.1 else "") + f"w{i}" + rnd.choice(["", "", ".", "?", ","])))
            t += d
        end = t + 1
        turns = [Turn(0, end, 0)] + [Turn(a, a + rnd.uniform(0.2, 3), rnd.choice([1, 1, 2]))
                                    for a in (rnd.uniform(0, end) for _ in range(rnd.randint(1, 15)))]
        for opts in (MergeOptions(), MergeOptions(inline_max_words=0), MergeOptions(bc_max_s=1.0, min_run_words=3, snap_words=2)):
            text = build_dialogue(ws, turns, min_share=0.0, paragraph_gap_s=2, options=opts)
            if text is None:
                continue
            body = re.sub(r"(?m)^[A-Z]: ", "", text)
            body = re.sub(r" \([A-Z]: ", " ", body).replace(")", "")
            assert "".join(body.split()) == "".join("".join(w.text for w in ws).split()), (opts, text)


def test_diarize_merge_options_are_parsed_and_checked():
    import pytest

    stt = fw.FasterWhisperSTT("m", diarize={"segmentation": "s", "embedding": "e", "merge": {"inline_max_words": 5}})
    assert stt._merge.inline_max_words == 5
    with pytest.raises(ValueError, match="merge"):
        fw.FasterWhisperSTT("m", diarize={"segmentation": "s", "embedding": "e", "merge": {"nope": 1}})


def test_merge_failure_falls_back_to_plain_text(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    stt = engine(FakeDiarizer([Turn(0, 2, 0), Turn(2, 4, 1)]), 200, monkeypatch)
    monkeypatch.setattr(fw, "build_dialogue", boom)
    assert stt._run(b"x", "he")[0] == "hi yes"


def test_merge_option_values_are_checked_at_start():
    import pytest

    for bad in (
        {"min_cover": "0.5"},
        {"inline_max_words": True},
        {"num_gap_s": -1},
        {"min_cover": 1.5},
        {"snap_words": 1.5},
        {"min_cover": float("nan")},
        {"num_gap_s": float("inf")},
    ):
        with pytest.raises(ValueError, match="merge"):
            fw.FasterWhisperSTT("m", diarize={"segmentation": "s", "embedding": "e", "merge": bad})


def test_zero_length_word_inside_a_reply_belongs_to_the_reply():
    turns = [Turn(0, 5.4, 0), Turn(5, 6, 1)]
    ws = words((4, 5, " one"), (5.5, 5.5, " yes"), (7, 8, " two"))
    assert assign_words(ws, turns, MergeOptions()) == [0, 1, 1]


def test_number_rule_does_not_glue_a_reply_onto_a_finished_sentence():
    turns = [Turn(0, 10, 0), Turn(5.0, 5.4, 1)]
    ws = words((4.0, 4.8, " 500."), (5.0, 5.4, " wow"), (7, 8, " done"))
    assert label_words(ws, turns, MergeOptions()) == [0, 1, 0]


def test_fallback_letters_never_repeat_a_custom_label():
    ws = words((0, 1, " a"), (2, 3, " b"), (4, 5, " c"))
    assert render_dialogue(ws, [0, 1, 2], labels=["X", "C"]) == "X: a\n\nC: b\n\nA: c"


def test_whitespace_only_interjection_is_not_shown_inline():
    ws = words((0, 1, " a"), (1, 2, " b"), (2, 3, "  "), (3, 4, " c"))
    assert render_dialogue(ws, [0, 0, 1, 0]) == "A: a b c"


def test_parens_in_the_text_and_hebrew_labels_keep_every_word():
    turns = [Turn(0, 20, 0), Turn(8, 9, 1)]
    ws = words((1, 2, " אני"), (3, 4, " הולך"), (8.1, 8.9, " (צוחק)"), (10, 11, " מחר"), (12, 13, " לשם"), (14, 15, " עכשיו"))
    out = build_dialogue(ws, turns, labels=["אדם א'", "אדם ב'"])
    assert out == "אדם א': אני הולך (אדם ב': (צוחק)) מחר לשם עכשיו"
