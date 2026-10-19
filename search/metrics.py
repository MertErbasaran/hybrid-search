"""Standard information-retrieval metrics.

All functions take
    ranked:   retrieved doc ids, best first
    relevant: {doc_id: relevance grade} for one query (grades > 0)
"""

import math


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """Share of the relevant documents that appear in the top k."""
    if not relevant:
        return 0.0
    found = sum(1 for d in ranked[:k] if d in relevant)
    return found / len(relevant)


def mrr_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """1 / rank of the first relevant hit (0 if none in the top k)."""
    for rank, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """Normalized Discounted Cumulative Gain: rewards relevant hits near the top."""
    dcg = sum(relevant.get(d, 0) / math.log2(rank + 1) for rank, d in enumerate(ranked[:k], start=1))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0
