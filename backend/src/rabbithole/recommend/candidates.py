"""Candidate communities and their features for a seed set (DATA_AND_ML §4.2, §4.5)."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import dijkstra

from rabbithole.recommend.catalog import Catalog

NO_PRED = -9999  # scipy's "no predecessor" marker


@dataclass(frozen=True)
class Candidates:
    """Parallel arrays, one entry per candidate community (indices into the catalog)."""

    idx: np.ndarray  # positions in catalog.community_ids
    sem_dist: np.ndarray  # 1 - cosine(user centroid, community centroid)
    graph_dist: np.ndarray  # hops on the cheapest path, max_hops + 1 if unreachable
    bridge_strength: np.ndarray  # exp(-cost) of the cheapest path, 0 if unreachable
    entry_article: np.ndarray  # member at the end of the cheapest path, -1 if unreachable
    predecessors: np.ndarray  # shortest-path tree over articles, for bridge_path()


def _hops(predecessors: np.ndarray, node: int) -> int:
    hops = 0
    while predecessors[node] != NO_PRED:
        node = int(predecessors[node])
        hops += 1
    return hops


def candidates(
    catalog: Catalog, seeds: list[int], shortlist: int = 200, max_hops: int = 6
) -> Candidates:
    """All eligible communities except the seeds' own, shortlisted to the `shortlist`
    semantically nearest, with graph features from one multi-source Dijkstra run."""
    seed_comms = set(catalog.article_community[seeds].tolist())
    user = catalog.article_vecs[seeds].mean(axis=0)
    user /= np.linalg.norm(user)
    sem = 1 - catalog.centroids @ user

    keep = np.array([c not in seed_comms for c in catalog.community_ids.tolist()])
    idx = np.flatnonzero(keep)
    idx = idx[np.argsort(sem[idx], kind="stable")][:shortlist]

    dist, pred, _ = dijkstra(catalog.graph, indices=seeds, min_only=True, return_predecessors=True)
    graph_dist = np.full(len(idx), max_hops + 1)
    strength = np.zeros(len(idx))
    entry = np.full(len(idx), -1)
    for j, ci in enumerate(idx.tolist()):
        in_comm = catalog.article_community == catalog.community_ids[ci]
        # Hubs ("Europe", "United States") make meaningless bridge endpoints (DATA_AND_ML §8).
        members = np.flatnonzero(in_comm & ~catalog.hub)
        if len(members) == 0:
            members = np.flatnonzero(in_comm)
        best = int(members[np.argmin(dist[members])])
        if np.isfinite(dist[best]):
            hops = _hops(pred, best)
            if hops <= max_hops:
                graph_dist[j], strength[j], entry[j] = hops, np.exp(-dist[best]), best
    return Candidates(idx, sem[idx], graph_dist, strength, entry, pred)


def bridge_path(cands: Candidates, j: int, max_hops: int = 4) -> list[int]:
    """Article ids from a seed to candidate j's entry article, or [] if none within max_hops.
    Paths come only from the click graph, never from the LLM."""
    node = int(cands.entry_article[j])
    if node < 0 or cands.graph_dist[j] > max_hops:
        return []
    path = [node]
    while cands.predecessors[node] != NO_PRED:
        node = int(cands.predecessors[node])
        path.append(node)
    return path[::-1]
