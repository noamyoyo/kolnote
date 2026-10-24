import io
import math
import struct
import wave

import pytest

pytest.importorskip("av")
np = pytest.importorskip("numpy")

from kolnote.audio import decode_16k_mono, probe_duration_s  # noqa: E402


def wav_bytes(seconds, rate=8000):
    frames = b"".join(
        struct.pack("<h", int(0.5 * 32767 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(int(seconds * rate))
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


def test_probe_duration_reads_the_container_length():
    assert abs(probe_duration_s(wav_bytes(3.0)) - 3.0) < 0.05


def test_probe_duration_is_none_for_garbage():
    assert probe_duration_s(b"not audio at all") is None


def test_decode_resamples_to_16k_mono_float32():
    x = decode_16k_mono(wav_bytes(2.0, rate=8000))
    assert x.dtype == np.float32
    assert abs(len(x) - 2 * 16000) < 400
    assert 0.3 < float(np.abs(x).max()) <= 1.0
