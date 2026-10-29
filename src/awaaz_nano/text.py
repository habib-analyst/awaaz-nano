"""Text normalization and the character vocabulary for Awaaz-Nano.

The normalization is deliberately small, rule-based, and documented in
DATA.md. It is applied identically at data-preparation, training,
evaluation, and inference time, and it defines the model's target
space: a normalized Urdu character sequence.

Rules, in order:
  1. Fold letter variants that the corpus uses inconsistently in only a
     handful of places (Arabic yeh/kaf/heh and alef maksura/wasla to
     their Urdu forms).
  2. Remove Arabic diacritics (harakat) and tatweel. In FLEURS ur_pk
     they are sparse and inconsistent (671 occurrences across 258,199
     train characters): the same word appears with and without them,
     so keeping them would make identical pronunciations different
     targets.
  3. Lowercase (affects only the small number of Latin letters).
  4. Keep only: space, Western digits 0-9, Latin a-z, and letters in
     the Arabic block (U+0600-U+06FF, Unicode category L*). Everything
     else -- punctuation, symbols, zero-width and bidi controls, the
     U+FDFA ligature -- is removed.
  5. Collapse runs of whitespace and strip.
"""

from __future__ import annotations

import unicodedata

_FOLD = {
    "ي": "ی",  # Arabic yeh -> Urdu yeh
    "ك": "ک",  # Arabic kaf -> Urdu keheh
    "ه": "ہ",  # Arabic heh -> Urdu heh goal
    "ى": "ی",  # alef maksura -> Urdu yeh
    "ٱ": "ا",  # alef wasla -> alef
}

_DIACRITICS = frozenset(
    [chr(c) for c in range(0x064B, 0x0653)]  # fathatan .. sukun
    + ["ٰ", "ٖ", "ـ", "ٔ"]  # superscript alef, subscript alef, tatweel, high hamza
)


def normalize(text: str) -> str:
    out = []
    for ch in text:
        ch = _FOLD.get(ch, ch)
        if ch in _DIACRITICS:
            continue
        ch = ch.lower()
        if ch == " " or "0" <= ch <= "9" or "a" <= ch <= "z":
            out.append(ch)
        elif 0x0600 <= ord(ch) <= 0x06FF and unicodedata.category(ch).startswith("L"):
            out.append(ch)
        # everything else is dropped
    return " ".join("".join(out).split())


class CharVocab:
    """Character vocabulary with CTC blank at index 0."""

    BLANK = "<blank>"

    def __init__(self, chars: list[str]):
        self.chars = list(chars)
        self.stoi = {c: i + 1 for i, c in enumerate(self.chars)}

    def __len__(self) -> int:
        return len(self.chars) + 1  # + blank

    def encode(self, text: str) -> list[int]:
        """Encode normalized text; characters outside the vocab are skipped."""
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode_ids(self, ids: list[int]) -> str:
        return "".join(self.chars[i - 1] for i in ids if i != 0)

    def greedy_decode(self, frame_ids: list[int]) -> str:
        """CTC greedy decode: collapse repeats, then remove blanks."""
        collapsed = []
        prev = None
        for i in frame_ids:
            if i != prev:
                collapsed.append(i)
            prev = i
        return self.decode_ids(collapsed)

    def to_json(self) -> dict:
        return {"blank": self.BLANK, "chars": self.chars}

    @classmethod
    def from_json(cls, obj: dict) -> "CharVocab":
        return cls(list(obj["chars"]))


def build_vocab(texts: list[str], min_count: int = 3) -> tuple[CharVocab, dict]:
    """Build a vocab from normalized texts, keeping chars seen >= min_count times.

    Returns (vocab, stats) where stats reports the dropped characters so
    the threshold's effect is on the record, not silent.
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    for t in texts:
        counts.update(t)
    kept = sorted(c for c, n in counts.items() if n >= min_count)
    dropped = {c: n for c, n in counts.items() if n < min_count}
    return CharVocab(kept), {"kept": len(kept), "dropped": dropped, "counts": dict(counts)}
