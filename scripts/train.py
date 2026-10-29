#!/usr/bin/env python3
"""Train Awaaz-Nano (CTC) on the prepared FLEURS ur_pk features.

Resumable by design: every epoch ends by atomically writing
``run/last.pt`` (model + optimizer + scheduler + epoch) and, when dev
CER improves, ``run/best.pt`` (model + selection metadata). Re-running
the script picks up from ``last.pt``. Progress lines go to stdout and
to ``run/train_stdout.log`` via the caller's redirection.

Batching is by padded-frame budget: records are shuffled, sorted inside
megabatches by length, and greedily packed so that
max_len x batch_size <= --batch-frames mel frames.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn as nn

from awaaz_nano.metrics import corpus_cer, corpus_wer
from awaaz_nano.model import AwaazNano
from awaaz_nano.text import CharVocab

SEED = 20261029


def load_manifest(package_dir: Path, split: str) -> list[dict]:
    path = package_dir / "data" / "manifests" / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_feats(work_dir: Path, split: str, fname: str) -> np.ndarray:
    return np.load(work_dir / "features" / split / (Path(fname).stem + ".npy")).astype(np.float32)


def make_batches(records: list[dict], budget: int, rng: np.random.Generator) -> list[list[dict]]:
    order = list(records)
    rng.shuffle(order)
    batches: list[list[dict]] = []
    mega = 1024
    for start in range(0, len(order), mega):
        chunk = sorted(order[start:start + mega], key=lambda r: -r["mel_frames"])
        cur: list[dict] = []
        cur_max = 0
        for rec in chunk:
            new_max = max(cur_max, rec["mel_frames"])
            if cur and new_max * (len(cur) + 1) > budget:
                batches.append(cur)
                cur, cur_max = [rec], rec["mel_frames"]
            else:
                cur.append(rec)
                cur_max = new_max
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def collate(batch: list[dict], work_dir: Path, split: str, vocab: CharVocab):
    feats = [load_feats(work_dir, split, r["file"]) for r in batch]
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    max_t = int(lengths.max())
    x = torch.zeros(len(feats), max_t, feats[0].shape[1], dtype=torch.float32)
    for i, f in enumerate(feats):
        x[i, : f.shape[0]] = torch.from_numpy(f)
    targets = [vocab.encode(r["target"]) for r in batch]
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    flat_targets = torch.tensor([t for seq in targets for t in seq], dtype=torch.long)
    return x, lengths, flat_targets, target_lengths


@torch.no_grad()
def evaluate_split(model, records, work_dir, split, vocab, device) -> tuple[float, float]:
    model.eval()
    pairs = []
    recs = sorted(records, key=lambda r: -r["mel_frames"])
    for start in range(0, len(recs), 32):
        batch = recs[start:start + 32]
        feats = [load_feats(work_dir, split, r["file"]) for r in batch]
        lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
        max_t = int(lengths.max())
        x = torch.zeros(len(feats), max_t, feats[0].shape[1], dtype=torch.float32)
        for i, f in enumerate(feats):
            x[i, : f.shape[0]] = torch.from_numpy(f)
        log_probs, out_lengths = model(x.to(device), lengths)
        for i, r in enumerate(batch):
            ids = log_probs[: int(out_lengths[i]), i].argmax(dim=-1).tolist()
            pairs.append((r["target"], vocab.greedy_decode(ids)))
    model.train()
    return corpus_cer(pairs), corpus_wer(pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--package-dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--batch-frames", type=int, default=96000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device("cpu")

    work_dir = Path(args.work_dir)
    package_dir = Path(args.package_dir)
    run_dir = work_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    vocab = CharVocab.from_json(json.loads((package_dir / "vocab.json").read_text(encoding="utf-8")))
    train_records = [r for r in load_manifest(package_dir, "train") if r["trainable"]]
    dev_records = load_manifest(package_dir, "dev")
    print(f"trainable train utterances: {len(train_records)}; dev: {len(dev_records)}", flush=True)

    model = AwaazNano(vocab_size=len(vocab))
    print(f"parameters: {model.count_parameters():,}", flush=True)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    rng = np.random.default_rng(args.seed)
    steps_per_epoch = len(make_batches(train_records, args.batch_frames, np.random.default_rng(0)))
    total_steps = steps_per_epoch * args.epochs
    warmup = steps_per_epoch  # 1 epoch linear warmup

    def lr_at(step: int) -> float:
        # LambdaLR multiplies the return value by the optimizer's base
        # lr, so this returns a dimensionless factor in [min_ratio, 1].
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    start_epoch = 0
    best_dev_cer = float("inf")
    last_path = run_dir / "last.pt"
    if last_path.exists() and not args.no_resume:
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_dev_cer = ckpt["best_dev_cer"]
        print(f"resumed at epoch {start_epoch}; best dev CER so far {best_dev_cer:.4f}", flush=True)

    log_path = run_dir / "train_log.jsonl"
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        batches = make_batches(train_records, args.batch_frames, rng)
        total_loss, total_frames = 0.0, 0
        for bi, batch in enumerate(batches):
            x, lengths, flat_targets, target_lengths = collate(batch, work_dir, "train", vocab)
            log_probs, out_lengths = model(x, lengths)
            loss = ctc(log_probs, flat_targets, out_lengths, target_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss) * len(batch)
            total_frames += int(lengths.sum())
            if bi % 20 == 0:
                print(f"  epoch {epoch} batch {bi}/{len(batches)} loss {float(loss):.3f}", flush=True)
        dev_cer, dev_wer = evaluate_split(model, dev_records, work_dir, "dev", vocab, device)
        improved = dev_cer < best_dev_cer
        if improved:
            best_dev_cer = dev_cer
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "dev_cer": dev_cer, "dev_wer": dev_wer},
                run_dir / "best.pt.tmp",
            )
            (run_dir / "best.pt.tmp").replace(run_dir / "best.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_dev_cer": best_dev_cer,
            },
            run_dir / "last.pt.tmp",
        )
        (run_dir / "last.pt.tmp").replace(run_dir / "last.pt")
        rec = {
            "epoch": epoch,
            "train_ctc_loss_per_utt": round(total_loss / len(train_records), 4),
            "dev_cer": round(dev_cer, 4),
            "dev_wer": round(dev_wer, 4),
            "improved": improved,
            "seconds": round(time.time() - t0, 1),
            "lr": scheduler.get_last_lr()[0],
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print("EPOCH " + json.dumps(rec), flush=True)
    print("TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
