"""Small PyAV helpers shared by the pipeline and the STT adapters. PyAV and numpy come with faster-whisper."""

from __future__ import annotations

import io
from typing import Any


def probe_duration_s(audio: bytes) -> float | None:
    """Container duration in seconds, or None when PyAV is missing or cannot read the file."""
    try:
        import av

        with av.open(io.BytesIO(audio)) as container:
            return container.duration / 1_000_000 if container.duration else None
    except Exception:
        return None


def decode_16k_mono(audio: bytes) -> Any:
    """Decode any audio file to a float32 numpy array at 16 kHz mono."""
    import av
    import numpy as np

    chunks = []
    with av.open(io.BytesIO(audio)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray()[0])
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray()[0])
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
