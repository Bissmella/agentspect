"""Deterministic, spread-out sampling of row indices.

Given a known total and a sample size, pick indices spread across the whole
source (stratified: one per evenly-sized bucket), with a seeded choice within
each bucket for variety. Deterministic for a given seed, so runs are reproducible.
"""

from __future__ import annotations

import random


def stratified_indices(total: int, sample_size: int, seed: int = 0) -> list[int]:
    if total <= 0 or sample_size <= 0:
        return []

    k = min(sample_size, total)
    if k == total:
        return list(range(total))

    rng = random.Random(seed)
    bounds = [round(i * total / k) for i in range(k + 1)]

    chosen: set[int] = set()
    for i in range(k):
        lo = bounds[i]
        hi = max(bounds[i + 1], lo + 1)
        chosen.add(rng.randrange(lo, min(hi, total)))

    return sorted(chosen)
