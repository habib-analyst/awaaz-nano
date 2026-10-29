#!/usr/bin/env python3
"""Evaluate a trained Awaaz-Nano checkpoint on a FLEURS ur_pk split.

Writes a JSON record with corpus CER/WER, per-utterance results, and a
failure analysis (error rate by duration quartile and by digit content,
character substitution/deletion/insertion profile, worst utterances).
All numbers are computed from the packaged weights and the cached
features of the requested split.

Usage:
  python3 scripts/evaluate.py --package-dir . --work-dir <work> \
      --split test --out assets/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from safetensors.torch import load_file

from awaaz_nano.metrics import alignment_counts, corpus_cer, corpus_wer, edit_distance
from awaaz_nano.model import AwaazNano
from awaaz_nano.text import CharVocab


def load_manifest(package_dir: Path, split: str) -> list[dict]:
    path = package_dir / "data" / "manifests" / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def alignment_pairs(ref: str, hyp: str):
    """One optimal alignment's (substitution pairs, deletions, insertions)."""
    a, b = list(ref), list(hyp)
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(1, la + 1):
        dp[i][0] = i
    for j in range(1, lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    subs, dels, ins = Counter(), Counter(), Counter()
    i, j = la, lb
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] != b[j - 1]:
                subs[(a[i - 1], b[j - 1])] += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels[a[i - 1]] += 1
            i -= 1
        else:
            ins[b[j - 1]] += 1
            j -= 1
    return subs, dels, ins


def quartile_bounds(values: list[float]) -> list[float]:
    s = sorted(values)
    n = len(s)
    return [s[n // 4], s[n // 2], s[3 * n // 4]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", default=None, help="default: <package-dir>/model.safetensors")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    package_dir = Path(args.package_dir)
    work_dir = Path(args.work_dir)
    config = json.loads((package_dir / "config.json").read_text(encoding="utf-8"))
    vocab = CharVocab.from_json(json.loads((package_dir / "vocab.json").read_text(encoding="utf-8")))
    arch = config["architecture"]
    model = AwaazNano(
        vocab_size=len(vocab), n_mels=arch["n_mels"],
        d_model=arch["d_model"], n_layers=arch["n_layers"],
        n_heads=arch["n_heads"], d_ff=arch["d_ff"], dropout=0.0,
    )
    weights_path = args.weights or str(package_dir / "model.safetensors")
    model.load_state_dict(load_file(weights_path))
    model.eval()

    records = load_manifest(package_dir, args.split)
    per_utt = []
    with torch.no_grad():
        recs = sorted(records, key=lambda r: -r["mel_frames"])
        for start in range(0, len(recs), 32):
            batch = recs[start:start + 32]
            feats = [
                np.load(work_dir / "features" / args.split / (Path(r["file"]).stem + ".npy")).astype(np.float32)
                for r in batch
            ]
            lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
            max_t = int(lengths.max())
            x = torch.zeros(len(feats), max_t, feats[0].shape[1], dtype=torch.float32)
            for i, f in enumerate(feats):
                x[i, : f.shape[0]] = torch.from_numpy(f)
            log_probs, out_lengths = model(x, lengths)
            for i, r in enumerate(batch):
                ids = log_probs[: int(out_lengths[i]), i].argmax(dim=-1).tolist()
                hyp = vocab.greedy_decode(ids)
                ref = r["target"]
                d = edit_distance(list(ref), list(hyp))
                s, de, ins = alignment_counts(list(ref), list(hyp))
                per_utt.append({
                    "file": r["file"],
                    "sentence_id": r["sentence_id"],
                    "duration_s": r["duration_s"],
                    "reference": ref,
                    "hypothesis": hyp,
                    "char_edits": d,
                    "cer": round(d / len(ref), 4) if ref else 0.0,
                    "subs": s, "dels": de, "ins": ins,
                })
    per_utt.sort(key=lambda u: u["file"])
    pairs = [(u["reference"], u["hypothesis"]) for u in per_utt]
    cer = corpus_cer(pairs)
    wer = corpus_wer(pairs)

    # Failure analysis
    bounds = quartile_bounds([u["duration_s"] for u in per_utt])
    def bucket(dur: float) -> int:
        return sum(dur > b for b in bounds)
    by_bucket: dict[int, list] = {}
    for u in per_utt:
        by_bucket.setdefault(bucket(u["duration_s"]), []).append(u)
    duration_buckets = []
    for b in sorted(by_bucket):
        us = by_bucket[b]
        duration_buckets.append({
            "quartile": b + 1,
            "n": len(us),
            "duration_range_s": [round(min(u["duration_s"] for u in us), 1), round(max(u["duration_s"] for u in us), 1)],
            "cer": round(corpus_cer([(u["reference"], u["hypothesis"]) for u in us]), 4),
        })
    digit = [u for u in per_utt if any("0" <= c <= "9" for c in u["reference"])]
    nodig = [u for u in per_utt if u not in digit]
    subs_c, dels_c, ins_c = Counter(), Counter(), Counter()
    for u in per_utt:
        s, d, i = alignment_pairs(u["reference"], u["hypothesis"])
        subs_c.update(s); dels_c.update(d); ins_c.update(i)
    worst = sorted(per_utt, key=lambda u: -u["cer"])[:10]
    analysis = {
        "duration_quartiles": duration_buckets,
        "digit_utterances": {
            "n": len(digit),
            "cer": round(corpus_cer([(u["reference"], u["hypothesis"]) for u in digit]), 4) if digit else None,
        },
        "non_digit_utterances": {
            "n": len(nodig),
            "cer": round(corpus_cer([(u["reference"], u["hypothesis"]) for u in nodig]), 4) if nodig else None,
        },
        "empty_hypotheses": sum(1 for u in per_utt if not u["hypothesis"]),
        "top_substitutions": [
            {"ref": a, "hyp": b, "count": n} for (a, b), n in subs_c.most_common(15)
        ],
        "top_deletions": [{"char": c, "count": n} for c, n in dels_c.most_common(10)],
        "top_insertions": [{"char": c, "count": n} for c, n in ins_c.most_common(10)],
        "worst_utterances": [
            {"file": u["file"], "duration_s": u["duration_s"], "cer": u["cer"],
             "reference": u["reference"], "hypothesis": u["hypothesis"]}
            for u in worst
        ],
    }
    result = {
        "split": args.split,
        "n_utterances": len(per_utt),
        "weights": weights_path,
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "total_ref_chars": sum(len(u["reference"]) for u in per_utt),
        "total_ref_words": sum(len(u["reference"].split()) for u in per_utt),
        "analysis": analysis,
        "per_utterance": per_utt,
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{args.split}: n={len(per_utt)} CER={cer:.4f} WER={wer:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
