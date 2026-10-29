#!/usr/bin/env python3
"""Re-derive the packaged metrics from the released weights.

Two checks:

1. Packaged samples (no download needed): transcribe the four sample
   clips in assets/samples/ with model.safetensors and confirm every
   hypothesis string and the subset CER/WER match
   assets/samples/index.json exactly.

2. Full test split (needs the prepared feature cache): re-run
   scripts/evaluate.py against model.safetensors and confirm the
   headline CER/WER and utterance count match assets/eval_results.json.

Usage:
  python3 examples/verify_metrics.py [--full-test --work-dir <work>]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from awaaz_nano.infer import Transcriber
from awaaz_nano.metrics import corpus_cer, corpus_wer


def check_samples(package: Path) -> bool:
    index = json.loads((package / "assets" / "samples" / "index.json").read_text(encoding="utf-8"))
    asr = Transcriber(package)
    pairs = []
    ok = True
    for s in index["samples"]:
        hyp = asr.transcribe_wav(package / "assets" / "samples" / s["file"])
        pairs.append((s["reference"], hyp))
        if hyp != s["hypothesis"]:
            ok = False
            print(f"MISMATCH {s['file']}:\n  stored: {s['hypothesis']}\n  live  : {hyp}")
    cer, wer = corpus_cer(pairs), corpus_wer(pairs)
    print(f"samples: n={len(pairs)} CER={cer:.4f} (stored {index['cer']}) WER={wer:.4f} (stored {index['wer']})")
    if round(cer, 4) != index["cer"] or round(wer, 4) != index["wer"]:
        ok = False
    print("samples check:", "OK" if ok else "FAILED")
    return ok


def check_full_test(package: Path, work_dir: str) -> bool:
    stored = json.loads((package / "assets" / "eval_results.json").read_text(encoding="utf-8"))
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        [
            sys.executable, str(package / "scripts" / "evaluate.py"),
            "--package-dir", str(package), "--work-dir", work_dir,
            "--split", "test", "--out", tmp_path,
        ],
        check=True, capture_output=True, text=True,
    )
    fresh = json.loads(Path(tmp_path).read_text(encoding="utf-8"))
    ok = (
        fresh["cer"] == stored["cer"]
        and fresh["wer"] == stored["wer"]
        and fresh["n_utterances"] == stored["n_utterances"]
    )
    print(f"full test: CER={fresh['cer']} (stored {stored['cer']}) WER={fresh['wer']} (stored {stored['wer']}) n={fresh['n_utterances']}")
    print("full-test check:", "OK" if ok else "FAILED")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-test", action="store_true")
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()
    package = Path(__file__).resolve().parents[1]
    ok = check_samples(package)
    if args.full_test:
        if not args.work_dir:
            print("--full-test requires --work-dir")
            sys.exit(2)
        ok = check_full_test(package, args.work_dir) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
