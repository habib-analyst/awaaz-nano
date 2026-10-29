"""Audio loading and log-mel feature extraction.

FLEURS ur_pk ships 16 kHz mono WAV in 32-bit IEEE float (format tag 3),
which the stdlib ``wave`` module cannot read. The reader here walks the
RIFF chunks directly and supports tag 3 (float32) and tag 1 (16-bit PCM,
used for the small packaged sample clips). ``torchaudio.load`` is the
fallback for anything else.

Features are 80-bin log-mel spectrograms: 25 ms window (400 samples),
12.5 ms hop (200 samples), n_fft 512, mel range 0-8000 Hz, then
per-utterance mean/variance normalization over the whole spectrogram
(the source recordings are quiet and unevenly leveled; the first train
file peaks at 0.020). The 12.5 ms hop puts the encoder (after the
model's x4 convolutional downsampling) at 20 frames/second; the corpus's
target character rate is 10.6 chars/s at the median and 14.5 at p99, so
CTC has roughly two frames per character to align with (exactly one of
the 2,109 train utterances is too dense for this rate and is excluded
by the trainable flag in the manifest).
"""

from __future__ import annotations

import struct

import numpy as np
import torch

SAMPLE_RATE = 16000
N_FFT = 512
WIN = 400
HOP = 200
N_MELS = 80

_mel_transform = None


def read_wav(path: str) -> np.ndarray:
    """Read a mono 16 kHz WAV (float32 or int16) as a float32 numpy array."""
    with open(path, "rb") as f:
        riff = f.read(12)
        if len(riff) < 12 or riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError(f"{path}: not a RIFF/WAVE file")
        fmt = None
        data = None
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            cid = hdr[:4]
            size = struct.unpack("<I", hdr[4:])[0]
            if cid == b"fmt ":
                raw = f.read(size)
                tag, ch, rate, _br, _ba, bits = struct.unpack("<HHIIHH", raw[:16])
                fmt = (tag, ch, rate, bits)
                if size % 2:
                    f.read(1)
            elif cid == b"data":
                data = f.read(size)
                break
            else:
                f.seek(size + (size % 2), 1)
    if fmt is None or data is None:
        # Unknown layout: let torchaudio/torchcodec deal with it.
        import torchaudio

        wav, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            raise ValueError(f"{path}: sample rate {sr}, expected {SAMPLE_RATE}")
        return wav.squeeze(0).numpy().astype(np.float32)
    tag, ch, rate, bits = fmt
    if rate != SAMPLE_RATE or ch != 1:
        raise ValueError(f"{path}: rate={rate} ch={ch}, expected 16000 Hz mono")
    if tag == 3 and bits == 32:
        return np.frombuffer(data, dtype="<f4").astype(np.float32)
    if tag == 1 and bits == 16:
        return (np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0)
    raise ValueError(f"{path}: unsupported WAV format tag={tag} bits={bits}")


def _get_mel():
    global _mel_transform
    if _mel_transform is None:
        import torchaudio

        _mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN,
            hop_length=HOP,
            n_mels=N_MELS,
            f_min=0.0,
            f_max=SAMPLE_RATE / 2,
            power=2.0,
            center=True,
        )
    return _mel_transform


def log_mel(waveform: np.ndarray) -> np.ndarray:
    """Waveform (float32, 16 kHz) -> log-mel features, shape (T, 80), float32.

    Per-utterance normalization: subtract the spectrogram mean and divide
    by its standard deviation.
    """
    wav = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
    mel = _get_mel()(wav)  # (80, T)
    feats = torch.log(mel + 1e-6).numpy().T  # (T, 80)
    mean = feats.mean()
    std = feats.std()
    if std > 0:
        feats = (feats - mean) / std
    return feats.astype(np.float32)
