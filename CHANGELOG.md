# Changelog

## 0.1.0 — 2026-10-29

Initial release.

- From-scratch CTC speech recognizer for Urdu: 80-bin log-mel front end,
  2-layer convolutional downsampler, 4-layer pre-norm Transformer
  encoder (d 192), character vocabulary of 73 symbols (incl. CTC blank),
  1,926,313 parameters.
- Trained only on the FLEURS ur_pk train split (2,108 of 2,109
  utterances; CC BY 4.0) with the documented normalization in DATA.md.
  No pretrained weights, no augmentation, no language model, greedy
  decoding.
- Held-out evaluation on the full FLEURS ur_pk test split (299
  utterances), re-derivable from the released weights with
  `scripts/evaluate.py`; packaged-sample subset checkable offline with
  `examples/verify_metrics.py`.
- Measured failure analysis ships in `assets/eval_results.json` and is
  summarized in the README.
