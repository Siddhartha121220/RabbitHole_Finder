"""Ground truth from FUTURE reader behaviour (DATA_AND_ML §4.4) and the query set (§6.1).

Features and communities come from the feature month M; labels come only from the clicks
of months M+1..M+2, so the recommender is graded on where readers actually went next.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from rabbithole.recommend.catalog import Catalog

SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Query:
    seeds: list[int]
    relevance: dict[int, int]  # community_id -> graded relevance 0..3


def transition_matrix(label_edges: pl.DataFrame, n: int) -> csr_matrix:
    """Row-normalised click probabilities from the label months (edges summed over months)."""
    e = label_edges.group_by("src_id", "dst_id").agg(pl.col("n").sum())
    e = e.with_columns((pl.col("n") / pl.col("n").sum().over("src_id")).alias("p"))
    return csr_matrix(
        (e["p"].to_numpy(), (e["src_id"].to_numpy(), e["dst_id"].to_numpy())), shape=(n, n)
    )


def walk_mass(t: csr_matrix, seeds: list[int]) -> np.ndarray:
    """Probability of being on each article after 2 or 3 steps of a random walk that starts
    uniformly on the seeds. Seeds and their direct (1-hop) neighbours are zeroed: the label
    rewards discovery, not links the reader could already see."""
    x0 = np.zeros(t.get_shape()[0])
    x0[seeds] = 1 / len(seeds)
    tt = t.T.tocsr()
    x1 = tt @ x0
    x2 = tt @ x1
    mass = x2 + tt @ x2
    mass[x1 > 0] = 0
    mass[seeds] = 0
    return mass


def grade(mass: np.ndarray) -> np.ndarray:
    """0 for no mass; positive masses split into tertiles within the query -> 1, 2, 3."""
    out = np.zeros(len(mass), dtype=int)
    pos = mass > 0
    if pos.any():
        cuts = np.quantile(mass[pos], [1 / 3, 2 / 3])
        out[pos] = 1 + np.searchsorted(cuts, mass[pos], side="right")
    return out


def relevance(catalog: Catalog, t: csr_matrix, seeds: list[int]) -> dict[int, int]:
    """Graded relevance for every eligible community outside the seeds' communities."""
    mass = walk_mass(t, seeds)
    ids = catalog.community_ids
    # Community mass = sum over member articles (non-eligible communities are dropped).
    per = np.bincount(catalog.article_community, weights=mass, minlength=int(ids.max()) + 1)
    seed_comms = set(catalog.article_community[seeds].tolist())
    cand = [int(c) for c in ids.tolist() if c not in seed_comms]
    grades = grade(per[cand])
    return dict(zip(cand, grades.tolist(), strict=True))


def community_splits(catalog: Catalog, seed: int = 7) -> dict[str, list[int]]:
    """Seed communities split 60/20/20 so no query community is shared across splits."""
    ids = np.random.default_rng(seed).permutation(catalog.community_ids).tolist()
    a, b = round(0.6 * len(ids)), round(0.8 * len(ids))
    return {"train": ids[:a], "valid": ids[a:b], "test": ids[b:]}


def sample_queries(
    catalog: Catalog,
    t: csr_matrix,
    communities: list[int],
    n: int,
    seed: int = 13,
    max_tries: int = 50,
) -> list[Query]:
    """n seed sets of 1-3 articles from one community each; sets whose future walk reaches no
    relevant (grade >= 2) community are skipped, since they cannot be scored."""
    rng = np.random.default_rng(seed)
    has_out = np.diff(t.indptr) > 0
    queries: list[Query] = []
    for _ in range(n * max_tries):
        if len(queries) == n:
            break
        c = int(rng.choice(communities))
        members = np.flatnonzero((catalog.article_community == c) & has_out)
        if len(members) == 0:
            continue
        size = int(rng.integers(1, 4))
        seeds = rng.choice(members, size=min(size, len(members)), replace=False).tolist()
        rel = relevance(catalog, t, seeds)
        if max(rel.values(), default=0) >= 2:
            queries.append(Query(seeds, rel))
    return queries
