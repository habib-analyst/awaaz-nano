"""Tests for Awaaz-Nano. All synthetic: no dataset download is required."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from awaaz_nano.features import log_mel, read_wav
from awaaz_nano.metrics import alignment_counts, corpus_cer, corpus_wer, edit_distance
from awaaz_nano.model import AwaazNano, conv_out_len, encoder_out_len
from awaaz_nano.text import CharVocab, build_vocab, normalize


# ---------- text normalization ----------

def test_normalize_folds_arabic_variants():
    assert normalize("كيه") == "کیہ"  # Arabic kaf/yeh/heh -> Urdu forms


def test_normalize_strips_diacritics_and_punctuation():
    # damma + Urdu full stop + Arabic comma are removed
    assert normalize("یہ کتابُ ہے۔") == "یہ کتاب ہے"
    assert normalize("سلام، دنیا!") == "سلام دنیا"


def test_normalize_digits_latin_and_spaces():
    assert normalize("35  ایم   ایم") == "35 ایم ایم"
    assert normalize("ABC xyz") == "abc xyz"


def test_normalize_removes_tatweel_and_controls():
    assert normalize("کتـــاب") == "کتاب"
    assert normalize("a\u200fb") == "ab"


def test_vocab_threshold_and_roundtrip():
    vocab, stats = build_vocab(["اب اب اب", "ابج"], min_count=3)
    assert "ج" not in vocab.stoi  # seen once, dropped
    assert stats["dropped"].get("ج") == 1
    ids = vocab.encode("اب")
    assert vocab.decode_ids(ids) == "اب"
    assert len(vocab) == len(vocab.chars) + 1  # blank


def test_greedy_decode_collapses_and_deblanks():
    vocab = CharVocab(["ا", "ب"])
    # ids: 0 blank, 1 alef, 2 beh; repeats collapse, blanks removed
    assert vocab.greedy_decode([1, 1, 0, 2, 2, 0, 1]) == "ابا"
    assert vocab.greedy_decode([0, 0, 0]) == ""
    # a blank between repeats means the character is genuinely doubled
    assert vocab.greedy_decode([1, 0, 1]) == "اا"


# ---------- metrics ----------

def test_edit_distance():
    assert edit_distance(list("kitten"), list("sitting")) == 3
    assert edit_distance([], ["a"]) == 1
    assert edit_distance(["a"], []) == 1
    assert edit_distance([], []) == 0


def test_alignment_counts():
    s, d, i = alignment_counts(list("abc"), list("axc"))
    assert (s, d, i) == (1, 0, 0)
    s, d, i = alignment_counts(list("abc"), list("ac"))
    assert (s, d, i) == (0, 1, 0)
    s, d, i = alignment_counts(list("ac"), list("abc"))
    assert (s, d, i) == (0, 0, 1)


def test_corpus_rates():
    pairs = [("اب", "اب"), ("اب", "ا")]
    assert corpus_cer(pairs) == pytest.approx(0.25)
    assert corpus_wer([("ایک دو", "ایک")]) == pytest.approx(0.5)


# ---------- features ----------

def _write_wav(path: Path, samples: np.ndarray, fmt: str):
    import struct

    if fmt == "float":
        data = samples.astype("<f4").tobytes()
        tag, bits = 3, 32
    else:
        data = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
        tag, bits = 1, 16
    byte_rate = 16000 * bits // 8
    block = bits // 8
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, tag, 1, 16000, byte_rate, block, bits)
    header += b"data" + struct.pack("<I", len(data))
    path.write_bytes(header + data)


def test_read_wav_float_and_int16(tmp_path):
    t = np.arange(16000, dtype=np.float32) / 16000
    wave = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    for fmt in ("float", "int16"):
        p = tmp_path / f"a_{fmt}.wav"
        _write_wav(p, wave, fmt)
        back = read_wav(str(p))
        assert back.shape == (16000,)
        assert np.abs(back - wave).max() < 2e-3


def test_log_mel_shape_and_normalization():
    rng = np.random.default_rng(0)
    wave = (0.05 * rng.standard_normal(16000)).astype(np.float32)
    feats = log_mel(wave)
    assert feats.shape[1] == 80
    assert feats.shape[0] == 81  # center=True: 1 + 16000/200
    assert abs(float(feats.mean())) < 1e-4
    assert abs(float(feats.std()) - 1.0) < 1e-3


# ---------- model ----------

def test_encoder_out_len():
    assert conv_out_len(100) == 50
    assert encoder_out_len(101) == 26
    assert encoder_out_len(4) == 1


def test_model_forward_shapes_and_ctc():
    torch.manual_seed(0)
    vocab = CharVocab(["ا", "ب", " "])
    model = AwaazNano(vocab_size=len(vocab))
    model.eval()
    feats = torch.randn(2, 120, 80)
    lengths = torch.tensor([120, 80])
    with torch.no_grad():
        log_probs, out_lengths = model(feats, lengths)
    assert out_lengths.tolist() == [encoder_out_len(120), encoder_out_len(80)]
    assert log_probs.shape == (encoder_out_len(120), 2, len(vocab))
    # log-probabilities normalize over the vocab
    probs = log_probs.exp().sum(dim=-1)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-4)


def test_ctc_loss_runs_and_backprops():
    torch.manual_seed(0)
    vocab = CharVocab(["ا", "ب"])
    model = AwaazNano(vocab_size=len(vocab))
    feats = torch.randn(2, 100, 80)
    lengths = torch.tensor([100, 100])
    log_probs, out_lengths = model(feats, lengths)
    targets = torch.tensor([1, 2, 1])
    target_lengths = torch.tensor([2, 1])
    loss = torch.nn.CTCLoss(blank=0)(log_probs, targets, out_lengths, target_lengths)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
