"""Offline metrics (DATA_AND_ML §6.3). Lists are community ids, best first.

"Relevant" for the binary metrics means graded relevance >= RELEVANT (top two thirds of
communities the future walk reached); NDCG uses the grades directly.
"""

from collections.abc import Callable, Sequence

import numpy as np

RELEVANT = 2


def ndcg_at_k(ranked: Sequence[int], rel: dict[int, int], k: int = 5) -> float:
    def dcg(grades: Sequence[int]) -> float:
        return sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(grades))

    ideal = dcg(sorted(rel.values(), reverse=True)[:k])
    return dcg([rel.get(c, 0) for c in ranked[:k]]) / ideal if ideal > 0 else 0.0


def recall_at_k(ranked: Sequence[int], rel: dict[int, int], k: int = 10) -> float:
    relevant = {c for c, g in rel.items() if g >= RELEVANT}
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def mrr(ranked: Sequence[int], rel: dict[int, int]) -> float:
    for i, c in enumerate(ranked):
        if rel.get(c, 0) >= RELEVANT:
            return 1 / (i + 1)
    return 0.0


def novelty(ranked: Sequence[int], pop_share: dict[int, float]) -> float:
    """Mean -log2 popularity share: higher = less mainstream."""
    return float(np.mean([-np.log2(pop_share[c]) for c in ranked])) if ranked else 0.0


def diversity(ranked: Sequence[int], vec: dict[int, np.ndarray]) -> float:
    """Intra-list diversity: mean pairwise cosine distance between community centroids."""
    if len(ranked) < 2:
        return 0.0
    m = np.stack([vec[c] for c in ranked])
    sim = m @ m.T
    iu = np.triu_indices(len(ranked), 1)
    return float(np.mean(1 - sim[iu]))


def serendipity(ranked: Sequence[int], rel: dict[int, int], obvious: Sequence[int]) -> float:
    """Share of the list that is relevant AND not in the nearest-neighbour top 10."""
    if not ranked:
        return 0.0
    hits = [c for c in ranked if rel.get(c, 0) >= RELEVANT and c not in set(obvious)]
    return len(hits) / len(ranked)


def coverage(lists: Sequence[Sequence[int]], n_catalog: int) -> float:
    """Share of the catalog that appears in at least one list."""
    return len({c for lst in lists for c in lst}) / n_catalog


def bootstrap_ci(
    values: np.ndarray, n: int = 1000, seed: int = 0, stat: Callable[[np.ndarray], float] = np.mean
) -> tuple[float, float, float]:
    """(point estimate, 2.5%, 97.5%) by resampling queries with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    boots = np.array([stat(values[i]) for i in idx])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(stat(values)), float(lo), float(hi)


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 0
) -> tuple[float, float, float, float]:
    """Mean of a - b over the same queries, its 95% CI, and a two-sided bootstrap p-value."""
    diff = a - b
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(diff.mean()), float(lo), float(hi), float(min(p, 1.0))
