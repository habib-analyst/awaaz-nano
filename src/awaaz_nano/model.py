"""Awaaz-Nano model: a small CNN + Transformer encoder trained with CTC.

Architecture (deliberately plain; the artifact's claim is a transparent,
fully documented from-scratch pipeline, not architectural novelty):

  log-mel (T, 80)
    -> Conv2d 1->32, 3x3, stride 2x2, BatchNorm, ReLU
    -> Conv2d 32->32, 3x3, stride 2x2, BatchNorm, ReLU   (time downsampled x4)
    -> flatten freq*channels (640) -> Linear -> 192
    -> + sinusoidal positions
    -> 4-layer pre-norm Transformer encoder (d 192, 4 heads, FF 768,
       dropout 0.1), padding-masked
    -> Linear 192 -> vocab size (incl. CTC blank)
    -> log-softmax over the vocabulary

With the packaged vocab this is 1,926,313 parameters.

Design note (measured, see REPORT.md): the first version of this model
used a 3-layer BiGRU stack. On this CPU-only machine the sequential GRU
cost ~65 s per training batch against ~5.5 s for this Transformer at
the same workload -- roughly 48 min vs 4 min per epoch -- because the
Transformer's work is dense GEMMs while the GRU's is sequential. The
encoder was switched before any training run; both figures are from
timed benchmarks on identical synthetic batches.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

D_MODEL = 192
N_LAYERS = 4
N_HEADS = 4
D_FF = 768
DROPOUT = 0.1


def conv_out_len(length: int) -> int:
    """Time length after one 3x3 stride-2 pad-1 conv: floor((L-1)/2)+1."""
    return (length - 1) // 2 + 1


def encoder_out_len(length: int) -> int:
    """Time length after the two stride-2 convs (total downsample x4)."""
    return conv_out_len(conv_out_len(length))


def _sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class AwaazNano(nn.Module):
    def __init__(self, vocab_size: int, n_mels: int = 80, d_model: int = D_MODEL,
                 n_layers: int = N_LAYERS, n_heads: int = N_HEADS, d_ff: int = D_FF,
                 dropout: float = DROPOUT, max_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        freq_after = (n_mels - 1) // 2 + 1
        freq_after = (freq_after - 1) // 2 + 1  # 80 -> 40 -> 20
        self.proj = nn.Linear(32 * freq_after, d_model)
        self.register_buffer("pos", _sinusoidal(max_len, d_model), persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor):
        """feats: (B, T, F) padded log-mel; lengths: (B,) true frame counts.

        Returns (log_probs (T', B, V), out_lengths (B,)).
        """
        x = feats.transpose(1, 2).unsqueeze(1)  # (B, 1, F, T)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # (B, T', C*F')
        x = self.proj(x) + self.pos[:t].unsqueeze(0)
        out_lengths = torch.tensor(
            [encoder_out_len(int(n)) for n in lengths], dtype=torch.long
        )
        pad_mask = torch.arange(t, device=feats.device).unsqueeze(0) >= out_lengths.to(
            feats.device
        ).unsqueeze(1)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        logits = self.out(x)  # (B, T', V)
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
        return log_probs, out_lengths

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
