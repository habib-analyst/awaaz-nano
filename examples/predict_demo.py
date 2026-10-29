#!/usr/bin/env python3
"""Transcribe the packaged sample clips with the packaged weights.

The samples are four FLEURS ur_pk *test* utterances (CC BY 4.0) stored
as 16-bit WAV under assets/samples/, chosen by a fixed rule (see
assets/samples/index.json and DATA.md). The demo prints the reference
and the model's live transcription for each, with the utterance CER.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from awaaz_nano.infer import Transcriber
from awaaz_nano.metrics import edit_distance


def main() -> None:
    package = Path(__file__).resolve().parents[1]
    index = json.loads((package / "assets" / "samples" / "index.json").read_text(encoding="utf-8"))
    asr = Transcriber(package)
    for s in index["samples"]:
        hyp = asr.transcribe_wav(package / "assets" / "samples" / s["file"])
        ref = s["reference"]
        cer = edit_distance(list(ref), list(hyp)) / len(ref)
        print(f"--- {s['file']} ({s['duration_s']}s, CER {cer:.3f})")
        print(f"reference : {ref}")
        print(f"hypothesis: {hyp}")


if __name__ == "__main__":
    main()
