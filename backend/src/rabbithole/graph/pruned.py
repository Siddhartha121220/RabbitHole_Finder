"""Pruned directed graph for serving-time path search (ARCHITECTURE §4, DATA_AND_ML §8)."""

import polars as pl


def prune(edges: pl.DataFrame, k: int = 10, hub_penalty: float = 0.1) -> pl.DataFrame:
    """Keep each article's top-k outgoing edges by p; cost = -log p + hub_penalty * log(1 + deg).

    `deg` is the destination's total degree in the unpruned graph, so paths that route
    through hubs ("United States", "India") cost more and bridges stay topical.
    """
    degree = (
        pl.concat(
            [edges.select(pl.col("src_id").alias("id")), edges.select(pl.col("dst_id").alias("id"))]
        )
        .group_by("id")
        .len(name="degree")
    )
    return (
        edges.sort(["src_id", "p", "dst_id"], descending=[False, True, False])
        .group_by("src_id", maintain_order=True)
        .head(k)
        .join(degree.rename({"id": "dst_id"}), on="dst_id", how="left")
        .with_columns((-pl.col("p").log() + hub_penalty * pl.col("degree").log1p()).alias("cost"))
        .select("src_id", "dst_id", "n", "p", "cost")
    )
