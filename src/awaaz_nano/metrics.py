"""Edit-distance metrics: corpus CER / WER and alignment error counts."""

from __future__ import annotations


def edit_distance(a: list, b: list) -> int:
    """Levenshtein distance between two sequences (chars or words)."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = cur
    return prev[lb]


def alignment_counts(ref: list, hyp: list) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) of hyp against ref via backtrace."""
    la, lb = len(ref), len(hyp)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(1, la + 1):
        dp[i][0] = i
    for j in range(1, lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
    subs = dels = ins = 0
    i, j = la, lb
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return subs, dels, ins


def corpus_cer(pairs: list[tuple[str, str]]) -> float:
    """Corpus character error rate over (reference, hypothesis) pairs."""
    dist = sum(edit_distance(list(r), list(h)) for r, h in pairs)
    total = sum(len(r) for r, _ in pairs)
    return dist / total if total else 0.0


def corpus_wer(pairs: list[tuple[str, str]]) -> float:
    """Corpus word error rate over (reference, hypothesis) pairs."""
    dist = sum(edit_distance(r.split(), h.split()) for r, h in pairs)
    total = sum(len(r.split()) for r, _ in pairs)
    return dist / total if total else 0.0
