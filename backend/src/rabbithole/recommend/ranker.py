"""Heuristic ranker v1 (DATA_AND_ML §4.3) and the baselines it is compared against (§6.2).

Every ranker returns candidate positions (indices into `Candidates`), best first.
"""

import numpy as np

from rabbithole.recommend.candidates import Candidates
from rabbithole.recommend.catalog import Catalog


def percentile(x: np.ndarray) -> np.ndarray:
    """Rank within the candidate set scaled to 0..1 (ties broken by order)."""
    if len(x) < 2:
        return np.zeros(len(x))
    return np.argsort(np.argsort(x, kind="stable"), kind="stable") / (len(x) - 1)


def goldilocks(d: np.ndarray, surprise: float, sigma: float = 0.12) -> np.ndarray:
    """Gaussian band around target distance d* = 0.25 + 0.5 * surprise."""
    target = 0.25 + 0.5 * surprise
    return np.exp(-((d - target) ** 2) / (2 * sigma**2))


def heuristic_scores(
    catalog: Catalog,
    cands: Candidates,
    surprise: float = 0.5,
    sigma: float = 0.12,
    use_rh: bool = True,
    use_bridge: bool = True,
) -> np.ndarray:
    score = goldilocks(percentile(cands.sem_dist), surprise, sigma)
    if use_rh:
        score = score * (0.5 + catalog.rh_score[cands.idx] / 200)
    if use_bridge:
        score = score * (0.7 + 0.3 * percentile(cands.bridge_strength))
    return score


def mmr(scores: np.ndarray, vecs: np.ndarray, k: int, lam: float = 0.7) -> list[int]:
    """Maximal marginal relevance: trade score (scaled to 0..1) against cosine similarity
    to what is already picked, so the list does not repeat one corner of the map."""
    rel = scores / scores.max() if scores.max() > 0 else scores
    sim = vecs @ vecs.T
    picked: list[int] = []
    left = list(range(len(scores)))
    while left and len(picked) < k:
        penalty = sim[np.ix_(left, picked)].max(axis=1) if picked else np.zeros(len(left))
        best = left[int(np.argmax(lam * rel[left] - (1 - lam) * penalty))]
        picked.append(best)
        left.remove(best)
    return picked


def top_k(scores: np.ndarray, k: int) -> list[int]:
    return np.argsort(-scores, kind="stable")[:k].tolist()


def heuristic(
    catalog: Catalog,
    cands: Candidates,
    k: int,
    surprise: float = 0.5,
    sigma: float = 0.12,
    lam: float | None = 0.7,
    use_rh: bool = True,
    use_bridge: bool = True,
) -> list[int]:
    """v1: band x rabbit-holeness x bridge; MMR re-ranking unless lam is None."""
    scores = heuristic_scores(catalog, cands, surprise, sigma, use_rh, use_bridge)
    if lam is None:
        return top_k(scores, k)
    return mmr(scores, catalog.centroids[cands.idx], k, lam)


def random_ranker(cands: Candidates, k: int, rng: np.random.Generator) -> list[int]:
    return rng.permutation(len(cands.idx))[:k].tolist()


def popularity(catalog: Catalog, cands: Candidates, k: int) -> list[int]:
    return top_k(catalog.views[cands.idx], k)


def nearest(cands: Candidates, k: int) -> list[int]:
    return top_k(-cands.sem_dist, k)
