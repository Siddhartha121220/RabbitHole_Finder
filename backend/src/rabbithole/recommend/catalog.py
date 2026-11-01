"""Everything the recommender needs in memory, built once from the pipeline's Parquet files."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class Catalog:
    titles: list[str]  # indexed by article_id
    article_vecs: np.ndarray  # (n_articles, d), unit rows
    article_community: np.ndarray  # (n_articles,) community_id per article
    community_ids: np.ndarray  # (n_eligible,) eligible communities, sorted
    centroids: np.ndarray  # (n_eligible, d), unit rows
    rh_score: np.ndarray  # (n_eligible,) 0..100
    views: np.ndarray  # (n_eligible,) total views in the feature month
    graph: csr_matrix  # pruned graph, entries = path cost (-log p + hub penalty)
    hub: np.ndarray  # (n_articles,) bool: linked from the most communities; never endpoints

    def index_of(self, community_id: int) -> int:
        return int(np.searchsorted(self.community_ids, community_id))


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def from_frames(
    titles: pl.DataFrame,
    embeddings: pl.DataFrame,
    membership: pl.DataFrame,
    rh_scores: pl.DataFrame,
    edges_pruned: pl.DataFrame,
) -> Catalog:
    """titles: article_id, title. embeddings: id, embedding. rh_scores: one row per eligible
    community with rh_score and popularity (= log1p views). edges_pruned: src_id, dst_id, cost."""
    n = titles.height
    vecs = _unit(embeddings.sort("id")["embedding"].to_numpy().astype(np.float32))
    community = membership.sort("article_id")["community_id"].to_numpy()
    scores = rh_scores.sort("community_id")
    ids = scores["community_id"].to_numpy()
    centroids = _unit(np.stack([vecs[community == c].mean(axis=0) for c in ids]))
    # csgraph treats stored zeros as missing edges, so keep every cost strictly positive.
    cost = edges_pruned["cost"].to_numpy() + 1e-9
    graph = csr_matrix(
        (cost, (edges_pruned["src_id"].to_numpy(), edges_pruned["dst_id"].to_numpy())),
        shape=(n, n),
    )
    # Hub = linked from many *different* communities ("Europe", "India"); a topic head like
    # "Titanic" has huge in-degree but from its own community, so it stays a valid entry.
    src, dst = edges_pruned["src_id"].to_numpy(), edges_pruned["dst_id"].to_numpy()
    pairs = np.unique(np.stack([dst, community[src]]), axis=1)
    reach = np.bincount(pairs[0], minlength=n)
    return Catalog(
        titles=titles.sort("article_id")["title"].to_list(),
        article_vecs=vecs,
        article_community=community,
        community_ids=ids,
        centroids=centroids,
        rh_score=scores["rh_score"].to_numpy(),
        views=np.expm1(scores["popularity"].to_numpy()),
        graph=graph,
        hub=reach > np.quantile(reach, 0.995),
    )


def load(processed_dir: Path) -> Catalog:
    def read(name: str) -> pl.DataFrame:
        return pl.read_parquet(processed_dir / f"{name}.parquet")

    return from_frames(
        read("article_ids"),
        read("article_embeddings"),
        read("article_communities"),
        read("rh_scores"),
        read("edges_pruned"),
    )
