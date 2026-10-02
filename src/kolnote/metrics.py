from __future__ import annotations

import re
import unicodedata

_NIQQUD = re.compile("[֑-ׇ]")
_MAQAF = "־"


def normalize(text: str) -> str:
    """Strip Hebrew points, punctuation and case so WER measures words, not typography."""
    text = unicodedata.normalize("NFKC", text)
    text = _NIQQUD.sub("", text.replace(_MAQAF, " "))
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.lower().split())


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    """Return (errors, reference_word_count) after normalization."""
    r, h = normalize(ref).split(), normalize(hyp).split()
    return _edit_distance(r, h), len(r)


def char_errors(ref: str, hyp: str) -> tuple[int, int]:
    """Return (errors, reference_char_count) after normalization."""
    r, h = list(normalize(ref)), list(normalize(hyp))
    return _edit_distance(r, h), len(r)


def rate(errors: int, total: int) -> float:
    return errors / total if total else (0.0 if errors == 0 else 1.0)
