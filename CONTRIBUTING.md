# Contributing

Issues and pull requests are welcome on GitHub.

Ground rules for this repository:

- **Numbers must be re-derivable.** Any metric in the README or in
  `assets/` must be reproducible from the released weights with the
  scripts in this repository. If you change the model, the data
  preparation, or the normalization, re-run the evaluation and update
  the recorded numbers in the same change.
- **Data provenance is part of the artifact.** Training data is FLEURS
  ur_pk only (CC BY 4.0); `DATA.md` records the exact files, sizes, and
  hashes. Do not add data from any other source without documenting it
  there first, license included.
- Run the tests before submitting: `python -m pytest`.
