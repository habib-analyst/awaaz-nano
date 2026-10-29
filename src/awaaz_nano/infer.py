"""Inference: load the packaged weights and transcribe a WAV file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .features import log_mel, read_wav
from .model import AwaazNano
from .text import CharVocab


class Transcriber:
    """Greedy CTC transcription with the packaged Awaaz-Nano weights."""

    def __init__(self, package_dir: str | Path):
        package_dir = Path(package_dir)
        with open(package_dir / "config.json", encoding="utf-8") as f:
            self.config = json.load(f)
        with open(package_dir / "vocab.json", encoding="utf-8") as f:
            self.vocab = CharVocab.from_json(json.load(f))
        arch = self.config["architecture"]
        self.model = AwaazNano(
            vocab_size=len(self.vocab),
            n_mels=arch["n_mels"],
            d_model=arch["d_model"],
            n_layers=arch["n_layers"],
            n_heads=arch["n_heads"],
            d_ff=arch["d_ff"],
            dropout=0.0,
        )
        state = load_file(str(package_dir / "model.safetensors"))
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def transcribe_features(self, feats: np.ndarray) -> str:
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0)
        lengths = torch.tensor([x.shape[1]], dtype=torch.long)
        log_probs, out_lengths = self.model(x, lengths)
        ids = log_probs[: int(out_lengths[0]), 0].argmax(dim=-1).tolist()
        return self.vocab.greedy_decode(ids)

    def transcribe_wav(self, wav_path: str | Path) -> str:
        return self.transcribe_features(log_mel(read_wav(str(wav_path))))
