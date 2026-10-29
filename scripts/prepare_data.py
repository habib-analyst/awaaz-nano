#!/usr/bin/env python3
"""Prepare FLEURS ur_pk for Awaaz-Nano training.

Reads the release TSVs and extracted WAVs, normalizes the transcripts,
builds the character vocabulary from the train split, extracts log-mel
features into a resumable per-utterance cache, and writes JSONL
manifests into the package (data/manifests/).

The release TSVs must be parsed by plain tab-splitting, NOT with the
csv module: some transcription fields contain double quotes, and a csv
reader merges quoted lines, silently dropping ~200 train rows (found
during the build; see DATA.md). One test row wraps its text fields in
double quotes; a field that starts and ends with '"' is unwrapped.

Usage:
  python3 scripts/prepare_data.py \
      --tsv-dir ~/workspace/earning/models/awaaz-nano-work/raw \
      --audio-dir ~/workspace/earning/models/awaaz-nano-work/audio \
      --work-dir ~/workspace/earning/models/awaaz-nano-work \
      --package-dir .
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from awaaz_nano.features import log_mel, read_wav
from awaaz_nano.model import encoder_out_len
from awaaz_nano.text import build_vocab, normalize

SPLITS = ["train", "dev", "test"]


def parse_tsv(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        parts = line.split("\t")
        parts = [
            p[1:-1] if len(p) > 1 and p.startswith('"') and p.endswith('"') else p
            for p in parts
        ]
        if len(parts) != 7:
            raise ValueError(f"{path}: row with {len(parts)} fields: {parts[:2]}")
        rows.append(parts)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv-dir", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--package-dir", required=True)
    ap.add_argument("--min-count", type=int, default=3)
    args = ap.parse_args()

    tsv_dir = Path(args.tsv_dir)
    audio_dir = Path(args.audio_dir)
    work_dir = Path(args.work_dir)
    package_dir = Path(args.package_dir)

    raw = {sp: parse_tsv(tsv_dir / f"{sp}.tsv") for sp in SPLITS}

    # Vocabulary from normalized train transcripts.
    norm_train = [normalize(r[3]) for r in raw["train"]]
    vocab, stats = build_vocab(norm_train, min_count=args.min_count)
    (package_dir / "vocab.json").write_text(
        json.dumps(vocab.to_json(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"vocab size (incl. blank): {len(vocab)}; kept chars: {stats['kept']}")
    print(f"chars dropped by min_count={args.min_count}: {stats['dropped']}")

    manifest_dir = package_dir / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for sp in SPLITS:
        feat_dir = work_dir / "features" / sp
        feat_dir.mkdir(parents=True, exist_ok=True)
        records = []
        n_infeasible = 0
        for r in raw[sp]:
            sent_id, fname, _transcription, raw_transcription, _phon, n_samples, gender = r
            target = "".join(c for c in normalize(raw_transcription) if c in vocab.stoi)
            wav_path = audio_dir / sp / fname
            cache = feat_dir / (Path(fname).stem + ".npy")
            if cache.exists():
                mel_frames = int(np.load(cache, mmap_mode="r").shape[0])
            else:
                feats = log_mel(read_wav(str(wav_path)))
                np.save(cache, feats.astype(np.float16))
                mel_frames = int(feats.shape[0])
            enc_frames = encoder_out_len(mel_frames)
            target_ids = vocab.encode(target)
            trainable = enc_frames >= len(target_ids)
            if not trainable:
                n_infeasible += 1
            records.append(
                {
                    "sentence_id": sent_id,
                    "file": fname,
                    "gender": gender,
                    "num_samples": int(n_samples),
                    "duration_s": round(int(n_samples) / 16000, 3),
                    "raw_transcription": raw_transcription,
                    "target": target,
                    "target_len": len(target_ids),
                    "mel_frames": mel_frames,
                    "encoder_frames": enc_frames,
                    "trainable": trainable,
                }
            )
        out = manifest_dir / f"{sp}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        total_h = sum(rec["duration_s"] for rec in records) / 3600
        print(
            f"{sp}: {len(records)} utterances, {total_h:.2f} h, "
            f"CTC-infeasible: {n_infeasible} -> {out}"
        )


if __name__ == "__main__":
    main()
