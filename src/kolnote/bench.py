"""Score STT engines against a folder of audio files with hand-written reference transcripts.

Dataset layout: `name.ogg` (any extension ffmpeg/PyAV decodes) next to `name.txt` (the reference).
"""

from __future__ import annotations

import mimetypes
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .metrics import char_errors, rate, word_errors
from .models import AUDIO_EXTENSIONS
from .ports import STTEngine


@dataclass
class FileResult:
    file: str
    reference: str
    hypothesis: str
    wer: float
    cer: float
    latency_s: float
    audio_s: float | None


@dataclass
class EngineResult:
    engine: str
    files: list[FileResult]
    wer: float
    cer: float
    total_latency_s: float
    total_audio_s: float
    rtf: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def find_cases(dataset: Path) -> list[tuple[Path, str]]:
    cases = []
    for audio in sorted(dataset.iterdir()):
        ref = audio.with_suffix(".txt")
        if audio.suffix.lower() in AUDIO_EXTENSIONS and ref.exists():
            cases.append((audio, ref.read_text(encoding="utf-8").strip()))
    return cases


async def run_bench(
    dataset: Path, engine: STTEngine, *, language: str | None, warmup: bool = True
) -> EngineResult:
    cases = find_cases(dataset)
    if not cases:
        raise SystemExit(f"no audio+txt pairs found in {dataset}")

    async def transcribe(path: Path):
        mime = mimetypes.guess_type(path.name)[0] or "audio/ogg"
        started = time.perf_counter()
        transcript = await engine.transcribe(path.read_bytes(), mime_type=mime, language=language)
        return transcript, time.perf_counter() - started

    if warmup:
        await transcribe(cases[0][0])

    files: list[FileResult] = []
    w_err = w_tot = c_err = c_tot = 0
    total_latency = total_audio = 0.0
    for path, reference in cases:
        transcript, latency = await transcribe(path)
        we, wn = word_errors(reference, transcript.text)
        ce, cn = char_errors(reference, transcript.text)
        w_err, w_tot, c_err, c_tot = w_err + we, w_tot + wn, c_err + ce, c_tot + cn
        total_latency += latency
        total_audio += transcript.duration_s or 0.0
        files.append(
            FileResult(path.name, reference, transcript.text, rate(we, wn), rate(ce, cn), latency, transcript.duration_s)
        )

    return EngineResult(
        engine=engine.name,
        files=files,
        wer=rate(w_err, w_tot),
        cer=rate(c_err, c_tot),
        total_latency_s=total_latency,
        total_audio_s=total_audio,
        rtf=(total_latency / total_audio) if total_audio else None,
    )


def format_summary(results: list[tuple[str, EngineResult]]) -> str:
    rows = [f"{'label':<40} {'WER':>7} {'CER':>7} {'RTF':>6} {'files':>5}"]
    for label, r in results:
        rtf = f"{r.rtf:.2f}" if r.rtf is not None else "n/a"
        rows.append(f"{label:<40} {r.wer:>7.1%} {r.cer:>7.1%} {rtf:>6} {len(r.files):>5}")
    return "\n".join(rows)
