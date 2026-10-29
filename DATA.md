# DATA.md — corpus card for Awaaz-Nano

## Source (the only source)

**FLEURS, ur_pk configuration** — Conneau et al., *FLEURS: Few-shot
Learning Evaluation of Universal Representations of Speech*, 2022.
FLEURS is an n-way parallel speech corpus: sentences sourced from
English Wikipedia, translated into 102 languages (the FLoRes-101
sentence set), and read aloud by native speakers.

- Official distribution used: the `google/fleurs` dataset repository on
  Hugging Face, files under `data/ur_pk/`, fetched 2026-10-29
  (Asia/Karachi) with `scripts/download_data.sh`. The tarballs are not
  redistributed here; they are freely downloadable from that repository.
- **License: CC BY 4.0.** The dataset card metadata declares
  `cc-by-4.0`, and the dataset README states: "All datasets are
  licensed under the Creative Commons license (CC-BY)". Attribution:
  this card and the README citation.
- No other corpus was used. Nothing was scraped; no utterance in this
  package comes from anywhere but the files below.

## Exact files used (all verified in the foreground at build time)

| File | Bytes | SHA-256 |
|---|---|---|
| `audio/train.tar.gz` | 1,224,331,874 | `00fbfdd4b98fa4c264b843355aa7450a3922642b73cb13e54ffa8950c4cdd93a` |
| `train.tsv` | 1,742,600 | `7fda1bd8c7e852547cf2c0a8d06fa45cbc37b09633c067056b593cbb88e94162` |
| `audio/dev.tar.gz` | 142,910,101 | `ee0a7a18242a70ea9088e9618337e4835ce1972557bdc2cd3d500555ce89520f` |
| `dev.tsv` | 217,526 | `80fe94d485d78e6277af6ca308c5ceb40f9a19352907cfada9ed07006ac20eed` |
| `audio/test.tar.gz` | 149,549,883 | `618840ad5fb21a59cc179ec535ddbc8e756e2fb40624689c6d59c93e019cd745` |
| `test.tsv` | 261,554 | `39cd4fd14ddcf730a9f2ddfd590181255555ba060a53506dcff598a8b9ecc476` |

## Splits, as verified from the downloaded TSVs and tarballs

| Split | Utterances | Unique sentences | Audio | Speakers (release metadata) |
|---|---|---|---|---|
| train | 2,109 | 1,272 | 7.03 h | 1,178 FEMALE / 931 MALE |
| dev | 267 | 142 | 0.76 h | 267 FEMALE |
| test | 299 | 230 | 0.81 h | 299 FEMALE |

- The utterance counts are exact: for every split, the set of WAV files
  in the tarball and the set of file names in the TSV are identical.
- **Sentence hygiene is verifiable and clean:** the sentence IDs of the
  three splits are pairwise disjoint (zero overlap) — no sentence is
  read in two splits.
- **Speaker hygiene is NOT verifiable from the release:** the TSVs
  carry a gender column but no speaker IDs, so whether the same voice
  appears in train and test cannot be audited from the distributed
  metadata. This is a limitation of the evaluation, stated plainly:
  treat the test score as a held-out-sentences score, not a proven
  held-out-speakers score.
- **Dev and test are all-female** in ur_pk, while train is mixed. Model
  selection (dev) and the headline evaluation (test) therefore measure
  female voices only.
- Audio format: 16 kHz, mono, 32-bit IEEE-float WAV (format tag 3;
  the stdlib `wave` module cannot read it — the package ships its own
  reader). Frame counts in the TSV `num_samples` column match the
  actual WAV frame counts (verified on samples from every split).
- Utterance durations: train median 11.3 s, mean 12.0 s, max 33.7 s.

## Parsing caveat (this bit a real pipeline; read before reusing the TSVs)

Parse the release TSVs by plain tab-splitting, **not** with a csv
reader. Some transcription fields contain double-quote characters; a
csv reader treats them as quote delimiters and merges physical lines,
silently collapsing the train split from 2,109 rows to 1,904. One test
row (sentence 1898) wraps its text fields in double quotes; the
preparation script unwraps a field that starts and ends with `"`.
`scripts/prepare_data.py` implements the correct parse.

## Targets and normalization (`scripts/prepare_data.py`, rerunnable)

The training target is the TSV `raw_transcription` column, normalized
by `awaaz_nano.text.normalize` (identical at train/eval/inference):

1. Fold rare letter variants to their Urdu forms (Arabic yeh/kaf/heh,
   alef maksura, alef wasla; the corpus is already near-uniform — e.g.
   Arabic yeh occurs exactly once in train).
2. Remove Arabic diacritics and tatweel (671 diacritic occurrences
   across 258,199 raw train characters; they are applied
   inconsistently, so the same word appears with and without them).
3. Lowercase Latin letters; keep only space, Western digits 0-9,
   Latin a-z, and Arabic-block letters. Punctuation and symbols are
   removed (the transcripts use Western digits exclusively; there are
   no Eastern Arabic-Indic digits in ur_pk).
4. Collapse whitespace.

The vocabulary is every normalized train character occurring at least
3 times: **72 characters + CTC blank = 73**. The threshold drops seven
characters with counts of 1-2 (Latin g, x, j, h, k, r and U+0674);
occurrences of dropped characters in targets are removed by the same
rule everywhere. Normalized targets total 256,715 characters (train),
31,865 (dev), 38,581 (test).

## CTC feasibility exclusion

At the model's encoder rate (20 frames/second), one train utterance is
too dense to align: `14244550635754859822.wav` packs 262 target
characters into 10.56 s (24.8 chars/s; corpus p99 is 14.5). It is
flagged `trainable: false` in the manifest and excluded from training,
leaving **2,108 training utterances**. No dev or test utterance is
excluded from evaluation.

## Features

80-bin log-mel spectrograms (25 ms window, 12.5 ms hop, n_fft 512,
0-8000 Hz), per-utterance mean/variance normalized, cached per
utterance by `scripts/prepare_data.py`. The manifests in
`data/manifests/` record every utterance's file name, sentence ID,
gender, duration, raw transcription, normalized target, and frame
counts, so the preparation is auditable without re-downloading.
