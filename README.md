# Awaaz-Nano

**A tiny Urdu speech recognizer, trained from scratch — no pretrained
weights — on one openly licensed corpus, with every number in this file
re-derivable from the released weights.** 1,926,313 parameters.
It is an educational / showcase artifact: it demonstrates an honest,
reproducible end-to-end ASR training run for a low-resource language,
not a competitor to any production Urdu recognizer.

> آواز (*āwāz*) — Urdu for "voice / sound".

## Headline numbers (all measured from the artifacts in this package)

| | |
|---|---|
| Parameters | 1,926,313 (CNN front end + 4-layer Transformer encoder, CTC) |
| Vocabulary | 73 symbols: 72 characters + CTC blank, from a documented normalization (see `DATA.md`) |
| Training data | FLEURS ur_pk train split — 2,108 of 2,109 utterances used, 7.03 h of audio, CC BY 4.0 |
| Held-out test (299 utterances, never trained on) | **CER 0.5131 · WER 0.9852** |
| Dev (267 utterances, used for checkpoint selection) | CER 0.5365 |
| Training | 18 of 36 scheduled epochs (stopped on a dev plateau; selected checkpoint = epoch 10), AdamW + cosine schedule, seed 20261029, CPU-only, 5,799 s of summed epoch time |

Read plainly: on held-out Urdu read speech, about **every second
character is wrong**, and at WER 0.9852 on average **every word is
wrong** — this is not usable transcription. What the model has learned
is real but shallow: frequent short words and formulaic fragments
surface intact, longer content words dissolve, and word boundaries
blur. Even its best test utterance has 35% character errors; none of
the 299 comes out empty or totally collapsed. Production Urdu ASR —
large pretrained models fine-tuned on far more data — operates in a
different regime entirely. The value here is the pipeline: openly
licensed data, a from-scratch model, and an evaluation a stranger can
re-run and get the same numbers (see `examples/verify_metrics.py`).

## Quickstart

```
python3 examples/predict_demo.py
```

transcribes the four packaged sample clips (`assets/samples/`, FLEURS
ur_pk *test* utterances) with the packaged weights and prints the
reference next to the model's output for each. Or in code:

```
import sys
sys.path.insert(0, "src")
from awaaz_nano.infer import Transcriber

asr = Transcriber(".")            # loads config.json + vocab.json + weights
print(asr.transcribe_wav("assets/samples/16678506964949649329.wav"))
```

## Architecture

Standard small CTC recognizer, deliberately unoriginal: 80-bin log-mel
features (25 ms window, 12.5 ms hop, per-utterance normalized) → two
3×3 stride-2 convolutions (time downsampled ×4, so the encoder runs at
20 frames/second) → linear projection to 192 → sinusoidal positions →
4 pre-norm Transformer encoder layers (4 heads, feed-forward 768,
dropout 0.1) → linear head over the character vocabulary → CTC.
Greedy decoding; no language model, no beam search, no augmentation
during training.

One design decision is worth stating because it was measured, not
assumed: the first version used a BiGRU encoder stack. On the CPU-only
machine this was trained on, the GRU cost ~65 s per training batch
against ~5.5 s for the Transformer at identical workload (~48 min vs
~4 min per epoch), so the encoder was switched before training. The
full account is in `REPORT.md`.

## Data (summary — full card in `DATA.md`)

Only one source: **FLEURS ur_pk** (Conneau et al. 2022), CC BY 4.0,
fetched from the dataset's official Hugging Face repository with
SHA-256 hashes recorded in `DATA.md`. The official splits are used
as-is: their sentence IDs are pairwise disjoint (verified). Two
properties of this split are stated rather than hidden:

- The release ships **no speaker IDs**, so speaker disjointness between
  train and test cannot be audited. The test score is a
  held-out-*sentences* score.
- The ur_pk **dev and test splits are all female voices**; train is
  mixed (1,178 F / 931 M). Checkpoint selection and the headline score
  therefore measure female speech.

Targets are the release's `raw_transcription` column under a small
documented normalization (variant folding, diacritic stripping,
punctuation removal; `DATA.md` gives exact rules and counts). One train
utterance (24.8 characters/second — the corpus p99 is 14.5) is too
dense for CTC at this encoder rate and is excluded, flagged in the
manifest.

## Evaluation and failure modes

Full record: `assets/eval_results.json` (every utterance's reference,
hypothesis, and error counts). Corpus figures over 38,581 reference
characters / 8,443 words: **CER 0.5131, WER 0.9852**. The measured
failure profile:

- **Length is not the problem.** CER by duration quartile is flat —
  0.517 / 0.515 / 0.512 / 0.511 from the shortest quartile (3.7-7.1 s)
  to the longest (11.9-25.8 s). The errors are a capacity/data limit,
  not a long-utterance effect.
- **Numbers are measurably worse.** The 51 digit-bearing test
  utterances score CER 0.5474 against 0.5054 for the rest, and several
  of the worst utterances are number-dense (a "1135" fire report, a
  cricket score line). The model has seen digits in only a small
  minority of train utterances and never learned them well.
- **Deletions dominate, and word boundaries blur.** The single most
  common error is a deleted space (1,148 instances): words merge.
  Short, quiet letters — ا ی ہ و ر ں — are dropped most after that.
- **The top confusions are the expected Urdu ones**: ی ↔ ے (yeh vs
  yeh barree; 277 + 167 substitutions), ہ → ے (130), and د → ت
  (dental vs retroflex-adjacent stops, 123).
- **Uniformly mediocre, not bimodal.** Per-utterance CER: mean 0.514,
  median 0.515, best 0.354, worst 0.741, zero empty hypotheses. The
  model never transcribes an utterance well and never totally fails
  one either.

Re-derive everything:

```
python3 scripts/evaluate.py --package-dir . --work-dir <prepared-work-dir> --split test --out /tmp/test_eval.json
python3 examples/verify_metrics.py            # packaged samples, offline
```

## Limitations — read before using

- **It is tiny and trained on 7 hours of read speech.** At CER 0.5131
  the output is partially intelligible gist at best — see the verbatim
  samples in `examples/predict_demo.py` output before forming any
  expectation.
- **Read, formal, Wikipedia-style Urdu only** — the FLEURS sentences
  are translations of English Wikipedia text. Conversational Urdu,
  dialect variation, code-switching with English, noisy or far-field
  audio, and children's or elderly voices are essentially out of
  scope, and the all-female test split means even the in-scope number
  says nothing measured about male voices.
- **Numbers are fragile targets.** Digit-bearing utterances measure
  ~4 CER points worse than the rest (0.547 vs 0.505); do not trust any
  number this model outputs.
- **Normalization is lossy by design.** Diacritics are stripped and
  punctuation is never predicted; the output is a plain character
  stream, not publishable Urdu orthography.
- Real Urdu ASR (large pretrained models fine-tuned on far more data)
  lives elsewhere. This package's claim is only: *from scratch, openly
  licensed data, every number reproducible.*

## Files

| File | What |
|---|---|
| `model.safetensors` | trained weights (the selected checkpoint) |
| `checkpoint.pt` | selection record: epoch + dev metrics of the shipped checkpoint |
| `config.json` | architecture + training/evaluation record |
| `vocab.json` | the 72-character vocabulary (+ blank) |
| `model.onnx` | ONNX export of the packaged model (log-mel → log-probs), parity-checked against PyTorch |
| `data/manifests/` | per-utterance manifests for all three splits (file, sentence ID, gender, durations, raw + normalized transcripts, frame counts) |
| `scripts/` | `download_data.sh`, `prepare_data.py`, `train.py`, `evaluate.py`, `export_onnx.py` — the whole pipeline |
| `src/awaaz_nano/` | text normalization, features, model, metrics, inference |
| `examples/` | `predict_demo.py` (packaged samples), `verify_metrics.py` (re-derive the recorded numbers) |
| `assets/eval_results.json` | the full test evaluation: corpus metrics, failure analysis, every utterance's reference + hypothesis |
| `assets/samples/` | four test utterances (16-bit WAV) with references and measured outputs |
| `training_log.txt` | the complete per-epoch training log |

## License

Code: MIT (`LICENSE`). The model weights were trained exclusively on
FLEURS ur_pk, which is **CC BY 4.0** (attribution: Conneau et al.,
*FLEURS*, 2022; see `DATA.md`) — the weights are released for research
and education with that attribution carried along.
