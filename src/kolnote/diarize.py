"""Speaker diarization (who spoke when) for long mono recordings. `dialogue.py` merges it with Whisper words.

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


class SherpaDiarizer:
    """Offline diarization. `num_speakers` > 0 forces that many speakers; 0 (default) lets the clustering
    decide with `threshold`, so one voice stays one speaker. Measured on a two-person call and a one-voice
    control: 0.8 to 0.9 finds 2 and 1; 0.7 and below invents speakers, 1.0 merges everyone."""

    def __init__(
        self,
        segmentation: str,
        embedding: str,
        *,
        num_speakers: int = 0,
        threshold: float = 0.8,
        threads: int = 4,
        min_duration_on: float = 0.3,
        min_duration_off: float = 0.5,
    ) -> None:
        self._segmentation = segmentation
        self._embedding = embedding
        self._num_speakers = num_speakers if num_speakers > 0 else -1
        self._threshold = threshold
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
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=self._num_speakers, threshold=self._threshold),
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
