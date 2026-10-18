"""Speaker diarization (who spoke when) for long mono recordings, and merging it with Whisper words.

The diarizer is sherpa-onnx: a pyannote segmentation model plus a speaker-embedding model, both
ONNX, so it needs neither torch nor a Hugging Face token. It runs on the CPU. Install with the
`diarize` extra and see the README for where to download the two model files.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger("kolnote.diarize")

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: int


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


def speaker_at(turns: Sequence[Turn], t: float) -> int:
    """Speaker at time `t`. Turns can overlap (a short reply inside a long turn): the shortest containing
    turn wins. Between turns, the speaker of the nearest turn."""
    inside = [x for x in turns if x.start <= t <= x.end]
    if inside:
        return min(inside, key=lambda x: x.end - x.start).speaker
    return min(turns, key=lambda x: min(abs(t - x.start), abs(t - x.end))).speaker


def speaker_shares(words: Sequence[Word], turns: Sequence[Turn]) -> dict[int, float]:
    """Share of the words each speaker owns (only speakers that own at least one word)."""
    if not words or not turns:
        return {}
    counts: dict[int, int] = {}
    for word in words:
        speaker = speaker_at(turns, (word.start + word.end) / 2)
        counts[speaker] = counts.get(speaker, 0) + 1
    return {speaker: n / len(words) for speaker, n in counts.items()}


def build_dialogue(words: Sequence[Word], turns: Sequence[Turn], *, paragraph_gap_s: float = 0) -> str:
    """One block per speaker change, as `A: text`, blocks separated by a blank line.

    Speakers are lettered in order of first appearance. A pause of `paragraph_gap_s` or more inside
    one speaker's talk starts a new block with the same letter (0 disables that).
    """
    if not words or not turns:
        return " ".join(w.text.strip() for w in words).strip()
    letters: dict[int, str] = {}
    blocks: list[tuple[str, list[str]]] = []
    previous_end = None
    for word in words:
        speaker = speaker_at(turns, (word.start + word.end) / 2)
        letter = letters.setdefault(speaker, LABELS[len(letters) % len(LABELS)])
        pause = paragraph_gap_s and previous_end is not None and word.start - previous_end >= paragraph_gap_s
        if not blocks or blocks[-1][0] != letter or pause:
            blocks.append((letter, []))
        blocks[-1][1].append(word.text)
        previous_end = word.end
    texts = [(letter, "".join(parts).strip()) for letter, parts in blocks]
    return "\n\n".join(f"{letter}: {text}" for letter, text in texts if text)


class SherpaDiarizer:
    """Offline diarization with a fixed number of speakers (a phone call has two)."""

    def __init__(
        self,
        segmentation: str,
        embedding: str,
        *,
        num_speakers: int = 2,
        threads: int = 4,
        min_duration_on: float = 0.3,
        min_duration_off: float = 0.5,
    ) -> None:
        self._segmentation = segmentation
        self._embedding = embedding
        self._num_speakers = num_speakers
        self._threads = threads
        self._min_on, self._min_off = min_duration_on, min_duration_off
        self._engine: Any = None
        self._lock = threading.Lock()

    def _build(self) -> Any:
        import sherpa_onnx

        seg = Path(self._segmentation)
        seg_model = seg / "model.onnx" if seg.is_dir() else seg
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg_model)),
                num_threads=self._threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=self._embedding, num_threads=self._threads),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=self._num_speakers),
            min_duration_on=self._min_on,
            min_duration_off=self._min_off,
        )
        if not config.validate():
            raise ValueError("invalid diarization config: check the segmentation and embedding model paths")
        return sherpa_onnx.OfflineSpeakerDiarization(config)

    def diarize(self, samples: Any) -> list[Turn]:
        """`samples`: float32 mono at 16 kHz."""
        with self._lock:
            if self._engine is None:
                self._engine = self._build()
            result = self._engine.process(samples).sort_by_start_time()
        return [Turn(r.start, r.end, r.speaker) for r in result]
