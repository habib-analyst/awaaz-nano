"""Awaaz-Nano: a tiny from-scratch Urdu speech recognizer (CTC).

Trained only on the FLEURS ur_pk corpus (CC BY 4.0). See README.md and
DATA.md for the full record, including the measured failure modes.
"""

from .text import CharVocab, normalize

__version__ = "0.1.0"
__all__ = ["CharVocab", "normalize", "__version__"]
