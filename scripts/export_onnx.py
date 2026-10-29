#!/usr/bin/env python3
"""Export the packaged Awaaz-Nano model to ONNX and verify parity.

The exported graph maps log-mel features (B, T, 80) to log-probabilities
(B, T', V), with a dynamic time axis — the same contract the PyTorch
module exposes, so a runtime (e.g. ONNX Runtime Web in a browser demo)
can reproduce greedy decoding exactly: mel extraction in the host,
then argmax + collapse + deblank with the packaged vocabulary.

Parity is checked, not assumed: for a few cached test utterances the
ONNX Runtime output is compared against the PyTorch model (max
log-prob difference) and the greedy decodings must agree exactly.

Usage:
  python3 scripts/export_onnx.py --package-dir . --work-dir <work> \
      --out model.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from safetensors.torch import load_file

from awaaz_nano.model import AwaazNano
from awaaz_nano.text import CharVocab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-check", type=int, default=5)
    args = ap.parse_args()

    package_dir = Path(args.package_dir)
    work_dir = Path(args.work_dir)
    config = json.loads((package_dir / "config.json").read_text(encoding="utf-8"))
    vocab = CharVocab.from_json(json.loads((package_dir / "vocab.json").read_text(encoding="utf-8")))
    arch = config["architecture"]
    model = AwaazNano(
        vocab_size=len(vocab), n_mels=arch["n_mels"], d_model=arch["d_model"],
        n_layers=arch["n_layers"], n_heads=arch["n_heads"], d_ff=arch["d_ff"],
        dropout=0.0,
    )
    model.load_state_dict(load_file(str(package_dir / "model.safetensors")))
    model.eval()

    dummy_feats = torch.randn(1, 200, arch["n_mels"])
    dummy_lengths = torch.tensor([200])
    torch.onnx.export(
        model,
        (dummy_feats, dummy_lengths),
        args.out,
        input_names=["feats", "lengths"],
        output_names=["log_probs", "out_lengths"],
        dynamic_axes={
            "feats": {0: "batch", 1: "time"},
            "log_probs": {0: "time_out", 1: "batch"},
        },
        opset_version=14,
    )
    print(f"exported -> {args.out}")

    # torch.onnx.export may externalize larger tensors into a
    # "<out>.data" sidecar; embed them so the artifact is one file.
    import onnx

    model_proto = onnx.load(args.out)  # loads external data
    onnx.save_model(model_proto, args.out, save_as_external_data=False)
    sidecar = Path(args.out + ".data")
    if sidecar.exists():
        sidecar.unlink()
    print("weights embedded: single-file ONNX")

    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    manifest = [
        json.loads(line)
        for line in (package_dir / "data" / "manifests" / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rng = np.random.default_rng(0)
    picks = [manifest[i] for i in rng.choice(len(manifest), size=args.n_check, replace=False)]
    max_diff = 0.0
    for rec in picks:
        feats = np.load(
            work_dir / "features" / "test" / (Path(rec["file"]).stem + ".npy")
        ).astype(np.float32)
        x = torch.from_numpy(feats).unsqueeze(0)
        lengths = torch.tensor([feats.shape[0]])
        with torch.no_grad():
            lp_t, ol_t = model(x, lengths)
        lp_o, _ol_o = sess.run(
            None, {"feats": feats[None], "lengths": np.array([feats.shape[0]], dtype=np.int64)}
        )
        t = int(ol_t[0])
        diff = float(np.abs(lp_t[:t, 0].numpy() - lp_o[:t, 0]).max())
        max_diff = max(max_diff, diff)
        dec_t = vocab.greedy_decode(lp_t[:t, 0].argmax(dim=-1).tolist())
        dec_o = vocab.greedy_decode(lp_o[:t, 0].argmax(axis=-1).tolist())
        status = "OK" if dec_t == dec_o else "DECODE MISMATCH"
        print(f"{rec['file']}: max|dlogprob|={diff:.2e} decode {status}")
        if dec_t != dec_o:
            print(f"  torch: {dec_t}\n  onnx : {dec_o}")
            sys.exit(1)
    print(f"parity: {args.n_check} utterances, max |dlogprob| = {max_diff:.2e}, decodes identical")


if __name__ == "__main__":
    main()
