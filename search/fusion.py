"""Reciprocal Rank Fusion (RRF).

Combines several ranked lists into one. Each document gets the score
    sum over all lists of 1 / (k + rank)
so documents that rank high in *any* list move up, and documents that appear
in *several* lists move up even more. Only ranks are used, not raw scores,
which is why RRF works well for mixing BM25 scores and cosine similarities
that live on completely different scales.

Reference: Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods".
"""

from collections import defaultdict
from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]], k: int = 60
) -> list[tuple[Hashable, float]]:
    """Fuse ranked lists (best first) into one list of (item, score), best first."""
    scores: dict[Hashable, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    # Sort by score descending; ties broken by item for deterministic output.
    return sorted(scores.items(), key=lambda pair: (-pair[1], str(pair[0])))
