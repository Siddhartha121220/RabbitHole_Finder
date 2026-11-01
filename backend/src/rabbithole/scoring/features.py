"""Rabbit-holeness features per community (DATA_AND_ML §3.1-3.2).

A rabbit hole is a topic where readers who enter keep going and stay inside; clickstream
measures that directly.
"""

import numpy as np
import polars as pl
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import spsolve

FEATURES = [
    "expected_depth",
    "stickiness",
    "onward_ratio",
    "breadth",
    "content_richness",
    "discovery_share",
]


def expected_depth(src: np.ndarray, dst: np.ndarray, q: np.ndarray, entries: np.ndarray) -> float:
    """Expected number of pages a reader views inside a community before leaving.

    Absorbing Markov chain over the m members: `q[k]` is the probability that a reader on
    `src[k]` next opens `dst[k]` (local indices 0..m-1); each row's missing mass is "leave"
    (click outside or stop reading). With Q the m x m matrix and pi the normalized entry
    distribution, the answer is pi . (I - Q)^-1 . 1. We solve (I - Q) y = 1 with a sparse
    solver instead of inverting; y[i] = expected pages viewed when starting at member i.
    """
    m = len(entries)
    if m == 0 or entries.sum() <= 0:
        return 0.0
    q_mat = csc_matrix((q, (src, dst)), shape=(m, m))
    y = np.asarray(spsolve(csc_matrix(identity(m) - q_mat), np.ones(m)), dtype=np.float64)
    y = y.reshape(m)  # spsolve returns a scalar when m == 1
    pi = entries / entries.sum()
    return float(pi @ y)


def community_features(
    articles: pl.DataFrame,
    edges: pl.DataFrame,
    membership: pl.DataFrame,
    page_length: pl.DataFrame,
) -> pl.DataFrame:
    """One row per community with FEATURES plus `popularity` and `size`.

    `articles`: article_id, views_in, search_in, internal_in, link_in, onward_clicks,
    onward_ratio. `edges`: src_id, dst_id, n. `membership`: article_id, community_id.
    `page_length`: article_id, page_length (bytes).
    """
    m = membership.select("article_id", "community_id")
    internal = (
        edges.join(m.rename({"article_id": "src_id", "community_id": "c"}), on="src_id")
        .join(m.rename({"article_id": "dst_id", "community_id": "c_dst"}), on="dst_id")
        .filter(pl.col("c") == pl.col("c_dst"))
        .select("c", "src_id", "dst_id", "n")
    )
    arts = (
        articles.join(m, on="article_id")
        .join(page_length, on="article_id", how="left")
        .join(
            internal.group_by("dst_id").agg(pl.col("n").sum().alias("in_from_members")),
            left_on="article_id",
            right_on="dst_id",
            how="left",
        )
        .with_columns(pl.col("in_from_members").fill_null(0))
        # Entries = visits that did not come from another member of the same community.
        .with_columns((pl.col("views_in") - pl.col("in_from_members")).clip(0).alias("entries"))
    )

    base = (
        arts.group_by("community_id")
        .agg(
            pl.len().alias("size"),
            pl.col("views_in").sum().alias("views"),
            pl.col("onward_clicks").sum().alias("onward_clicks"),
            pl.col("in_from_members").sum().alias("internal_clicks"),
            (pl.col("onward_ratio") * pl.col("views_in")).sum().alias("onward_weighted"),
            pl.col("search_in").sum().alias("search_in"),
            pl.col("internal_in").sum().alias("other_internal_in"),
            pl.col("link_in").sum().alias("link_in"),
            pl.col("page_length").fill_null(0).median().alias("median_page_length"),
        )
        .with_columns(
            (pl.col("internal_clicks") / pl.col("onward_clicks")).fill_nan(0).alias("stickiness"),
            (pl.col("onward_weighted") / pl.col("views")).fill_nan(0).alias("onward_ratio"),
            pl.col("size").log().alias("breadth"),
            pl.col("median_page_length").log1p().alias("content_richness"),
            pl.col("views").log1p().alias("popularity"),
            # Wandering in = other-internal + links from outside the community.
            (pl.col("other_internal_in") + pl.col("link_in") - pl.col("internal_clicks")).alias(
                "wander_in"
            ),
        )
        .with_columns(
            (pl.col("wander_in") / (pl.col("wander_in") + pl.col("search_in")))
            .fill_nan(0)
            .alias("discovery_share")
        )
    )

    # Per-visit transition probability: clicks i->j / visits to i. Using views (not total
    # onward clicks) keeps "stopped reading" as part of leaving. max() and the +1 keep every
    # row sum strictly below 1 (onward clicks can exceed views), so I - Q is never singular.
    denom = arts.select(
        "article_id",
        (pl.max_horizontal("views_in", "onward_clicks") + 1).alias("denom"),
    )
    q_edges = internal.join(denom, left_on="src_id", right_on="article_id").with_columns(
        (pl.col("n") / pl.col("denom")).alias("q")
    )
    depths: list[dict[str, float]] = []
    for (cid,), group in arts.sort("article_id").group_by(["community_id"], maintain_order=True):
        local = {a: i for i, a in enumerate(group["article_id"].to_list())}
        e = q_edges.filter(pl.col("c") == cid)
        depths.append(
            {
                "community_id": cid,
                "expected_depth": expected_depth(
                    np.array([local[a] for a in e["src_id"].to_list()], dtype=np.int64),
                    np.array([local[a] for a in e["dst_id"].to_list()], dtype=np.int64),
                    e["q"].to_numpy(),
                    group["entries"].cast(pl.Float64).to_numpy(),
                ),
            }
        )
    return (
        base.join(pl.DataFrame(depths), on="community_id")
        .select("community_id", "size", *FEATURES, "popularity")
        .sort("community_id")
    )
