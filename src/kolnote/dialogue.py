"""Merge Whisper words with speaker turns into a readable dialogue.

Word timestamps and diarization turn edges are each off by a fraction of a second, and a short turn that
sits inside another speaker's speech (a "yes", a "right") is the least reliable thing a diarizer outputs.
So the merge does not trust single turns: each word goes to the speaker with the most overlapping time,
very short turns under someone else's speech are dropped, minority runs inside a sentence are folded into
the sentence's main speaker, and a change is moved to the nearest real pause. A short interjection that
interrupts an unfinished sentence is shown inline, in brackets, so nothing is lost. Every step has a knob
in `MergeOptions`; `0` turns a step off.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import Sequence

from .diarize import Turn, Word

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SENTENCE_END = re.compile(r"[.?!…][\"')\]]*$")


COUNTS = ("inline_max_words", "min_run_words", "snap_words", "sent_max_words")


@dataclass(frozen=True)
class MergeOptions:
    min_cover: float = 0.5  # a speaker covers a word when its turns overlap at least this share of it
    num_gap_s: float = 0.3  # no change between a number and the word right after it
    inline_max_words: int = 3  # an interjection this short, inside an unfinished sentence, is shown inline
    # cleanup steps that proved destructive on real call audio (they absorb the quieter speaker's replies):
    # off by default, kept for tuning
    bc_max_s: float = 0.0  # a turn shorter than this ...
    bc_overlap: float = 0.5  # ... covered at least this much by other speakers is a backchannel and is dropped
    min_run_words: int = 0  # a minority run shorter than this inside a sentence is folded into the main speaker
    snap_gap_s: float = 0.15  # a longer run still needs a pause this long next to it
    snap_words: int = 0  # a change moves up to this many words to land on the biggest pause
    sent_gap_s: float = 0.8  # a pause this long also ends a sentence (for the folding step)
    sent_max_words: int = 40

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a number, got {value!r}")
            if name in COUNTS and not isinstance(value, int):
                raise ValueError(f"{name} must be a whole number, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must not be negative, got {value!r}")
        if self.min_cover > 1 or self.bc_overlap > 1:
            raise ValueError("min_cover and bc_overlap are shares between 0 and 1")


# ---- intervals ------------------------------------------------------------------------------------------


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(a, b) for a, b in out]


class _Spans:
    """Sorted, non-overlapping intervals of one speaker; overlap lookups by bisect."""

    def __init__(self, intervals: list[tuple[float, float]]) -> None:
        self.raw = sorted(intervals)
        self.spans = _merge(intervals)
        self._ends = [b for _, b in self.spans]

    def shortest(self, start: float, end: float) -> float:
        """Length of the shortest raw turn that overlaps [start, end]; inf when none does."""
        best = float("inf")
        for a, b in self.raw:
            if a >= end:
                break
            if b > start:
                best = min(best, b - a)
        return best

    def overlap(self, start: float, end: float) -> float:
        total = 0.0
        for i in range(bisect_right(self._ends, start), len(self.spans)):
            a, b = self.spans[i]
            if a >= end:
                break
            total += min(b, end) - max(a, start)
        return total


def _by_speaker(turns: Sequence[Turn]) -> dict[int, _Spans]:
    grouped: dict[int, list[tuple[float, float]]] = {}
    for t in turns:
        grouped.setdefault(t.speaker, []).append((t.start, t.end))
    return {speaker: _Spans(v) for speaker, v in grouped.items()}


# ---- steps ---------------------------------------------------------------------------------------------


def drop_backchannels(turns: Sequence[Turn], o: MergeOptions) -> list[Turn]:
    """Drop short turns that mostly sit under another speaker's speech. A short turn in silence stays."""
    if o.bc_max_s <= 0:
        return list(turns)
    spans = _by_speaker(turns)
    kept = []
    for t in turns:
        duration = t.end - t.start
        if 0 < duration < o.bc_max_s:
            covered = sum(s.overlap(t.start, t.end) for k, s in spans.items() if k != t.speaker)
            if min(covered, duration) / duration >= o.bc_overlap:
                continue
        kept.append(t)
    return kept


def _is_continuation(text: str) -> bool:
    """A token with no leading space and no letters (".377", "%", ",") continues the word before it."""
    return not text[:1].isspace() and not any(ch.isalpha() for ch in text)


def assign_words(words: Sequence[Word], turns: Sequence[Turn], o: MergeOptions) -> list[int]:
    """Speaker per word. A speaker covers a word when its turns overlap at least `min_cover` of the word
    (word times jitter, so a midpoint test is too strict). Several can cover it, because a long turn
    blankets the short replies inside it: the speaker with the shortest overlapping turn is the reply."""
    spans = _by_speaker(turns)
    nearest = lambda t: min(turns, key=lambda x: min(abs(t - x.start), abs(t - x.end))).speaker  # noqa: E731
    speakers: list[int] = []
    previous: int | None = None
    for word in words:
        if previous is not None and _is_continuation(word.text):
            speakers.append(previous)
            continue
        start, end = word.start, word.end
        if end - start < 0.02:  # a zero-length word still sits at a moment in time
            mid = (start + end) / 2
            start, end = mid - 0.01, mid + 0.01
        duration = end - start
        overlaps = {k: s.overlap(start, end) for k, s in spans.items()}
        covering = [k for k, v in overlaps.items() if v / duration >= o.min_cover]
        if covering:
            shortest = {k: spans[k].shortest(start, end) for k in covering}
            low = min(shortest.values())
            tied = [k for k in covering if shortest[k] <= low + 1e-9]
            current = previous if previous in tied else tied[0]
        elif any(v > 0 for v in overlaps.values()):
            current = max(overlaps, key=lambda k: overlaps[k])
        else:
            current = previous if previous is not None else nearest((word.start + word.end) / 2)
        speakers.append(current)
        previous = current
    return speakers


def _gap(words: Sequence[Word], i: int) -> float:
    """Silence before word i."""
    return words[i].start - words[i - 1].end if i > 0 else float("inf")


def keep_numbers_whole(words: Sequence[Word], speakers: list[int], o: MergeOptions) -> None:
    if o.num_gap_s <= 0:
        return
    for i in range(1, len(words)):
        if (
            speakers[i] != speakers[i - 1]
            and any(ch.isdigit() for ch in words[i - 1].text)
            and not SENTENCE_END.search(words[i - 1].text.strip())
            and _gap(words, i) < o.num_gap_s
        ):
            speakers[i] = speakers[i - 1]


def _sentences(words: Sequence[Word], o: MergeOptions) -> list[tuple[int, int]]:
    out, start = [], 0
    for i, word in enumerate(words):
        last = i + 1 == len(words)
        if (
            last
            or SENTENCE_END.search(word.text.strip())
            or _gap(words, i + 1) >= o.sent_gap_s
            or i + 1 - start >= o.sent_max_words
        ):
            out.append((start, i + 1))
            start = i + 1
    return out


def realign_sentences(words: Sequence[Word], speakers: list[int], o: MergeOptions) -> None:
    """Inside one sentence a short minority run is not a turn: give it to the sentence's main speaker."""
    if o.min_run_words <= 0:
        return
    for lo, hi in _sentences(words, o):
        spent: dict[int, float] = {}
        for i in range(lo, hi):
            spent[speakers[i]] = spent.get(speakers[i], 0.0) + max(words[i].end - words[i].start, 0.01)
        if len(spent) < 2:
            continue
        major = max(spent, key=lambda k: spent[k])
        i = lo
        while i < hi:
            j = i
            while j < hi and speakers[j] == speakers[i]:
                j += 1
            if speakers[i] != major:
                at_pause = max(_gap(words, i), _gap(words, j) if j < len(words) else 0.0) >= o.snap_gap_s
                if not (j - i >= o.min_run_words and at_pause):
                    for k in range(i, j):
                        speakers[k] = major
            i = j


def snap_changes(words: Sequence[Word], speakers: list[int], o: MergeOptions) -> None:
    """Move a speaker change up to `snap_words` words, onto the biggest pause there."""
    if o.snap_words <= 0:
        return
    i = 1
    while i < len(words):
        if speakers[i] == speakers[i - 1]:
            i += 1
            continue
        here = _gap(words, i)
        best, best_gap = i, here
        for j in range(max(1, i - o.snap_words), min(len(words), i + o.snap_words + 1)):
            if j == i or _gap(words, j) <= best_gap + 0.1:
                continue
            moved = range(j, i) if j < i else range(i, j)
            before = speakers[i - 1] if j < i else speakers[i]  # the side that gives the words away
            if all(speakers[k] == before for k in moved):
                best, best_gap = j, _gap(words, j)
        if best != i:
            after = speakers[i] if best < i else speakers[i - 1]
            for k in range(min(best, i), max(best, i)):
                speakers[k] = after
        i += 1


def label_words(words: Sequence[Word], turns: Sequence[Turn], o: MergeOptions | None = None) -> list[int]:
    """Speaker per word, after every step. Empty turns give an empty list."""
    o = o or MergeOptions()
    if not words or not turns:
        return []
    turns = drop_backchannels(turns, o) or list(turns)
    speakers = assign_words(words, turns, o)
    keep_numbers_whole(words, speakers, o)
    realign_sentences(words, speakers, o)
    snap_changes(words, speakers, o)
    return speakers


def speaker_shares(speakers: Sequence[int]) -> dict[int, float]:
    """Share of the words each speaker owns."""
    if not speakers:
        return {}
    counts: dict[int, int] = {}
    for speaker in speakers:
        counts[speaker] = counts.get(speaker, 0) + 1
    return {k: n / len(speakers) for k, n in counts.items()}


# ---- rendering -----------------------------------------------------------------------------------------


def render_dialogue(
    words: Sequence[Word],
    speakers: Sequence[int],
    *,
    paragraph_gap_s: float = 0,
    labels: Sequence[str] | None = None,
    options: MergeOptions | None = None,
) -> str:
    """One block per speaker change, as `A: text`, blocks separated by a blank line.

    Speakers are named in order of first appearance: `labels` first (for example Hebrew names, so the
    label does not fight right-to-left text), then A, B, ... A pause of `paragraph_gap_s` inside one
    speaker's talk starts a new block with the same label (0 disables that). A very short interjection
    inside an unfinished sentence is shown inline, as `(B: yes)`.
    """
    o = options or MergeOptions()
    names = list(labels or [])
    named: dict[int, str] = {}
    for speaker in speakers:
        if speaker not in named:
            n = len(named)
            if n < len(names):
                named[speaker] = names[n]
            else:
                used = set(named.values()) | set(names)
                free = [c for c in LABELS if c not in used]
                named[speaker] = free[0] if free else f"{LABELS[n % len(LABELS)]}{n + 1}"
    # blocks: [speaker, [text or (speaker, text)]]
    blocks: list[list] = []
    for i, word in enumerate(words):
        pause = paragraph_gap_s and i > 0 and _gap(words, i) >= paragraph_gap_s
        if not blocks or blocks[-1][0] != speakers[i] or pause:
            blocks.append([speakers[i], []])
        blocks[-1][1].append(word.text)

    def plain(block: list) -> str:
        return "".join(p for p in block[1] if isinstance(p, str)).strip()

    if o.inline_max_words > 0:
        i = 1
        while i < len(blocks) - 1:
            sp, parts = blocks[i]
            text = plain(blocks[i])
            before, after = blocks[i - 1], blocks[i + 1]
            if (
                text
                and len(text.split()) <= o.inline_max_words
                and before[0] == after[0] != sp
                and not SENTENCE_END.search(plain(before))
            ):
                before[1] += [(sp, text)] + after[1]
                del blocks[i : i + 2]
            elif not text and before[0] == after[0] != sp:
                before[1] += after[1]
                del blocks[i : i + 2]
            else:
                i += 1

    def render(block: list) -> str:
        out = []
        for part in block[1]:
            out.append(part if isinstance(part, str) else f" ({named[part[0]]}: {part[1]})")
        return "".join(out).strip()

    return "\n\n".join(f"{named[sp]}: {text}" for sp, text in ((b[0], render(b)) for b in blocks) if text)


def build_dialogue(
    words: Sequence[Word],
    turns: Sequence[Turn],
    *,
    min_share: float = 0.05,
    paragraph_gap_s: float = 0,
    labels: Sequence[str] | None = None,
    options: MergeOptions | None = None,
) -> str | None:
    """The dialogue text, or None when the audio is not a conversation (fewer than two speakers own at
    least `min_share` of the words): the caller then returns the plain transcript."""
    o = options or MergeOptions()
    speakers = label_words(words, turns, o)
    if not speakers:
        return None
    shares = speaker_shares(speakers)
    real = {k for k, v in shares.items() if v >= min_share}
    if len(real) < 2:
        return None
    last = next(s for s in speakers if s in real)
    for i, s in enumerate(speakers):  # words of a noise speaker go to the speaker before them
        if s in real:
            last = s
        else:
            speakers[i] = last
    return render_dialogue(words, speakers, paragraph_gap_s=paragraph_gap_s, labels=labels, options=o)
