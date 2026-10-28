"""Per-article stats and the transition edge table (DATA_AND_ML §1.1)."""

from pathlib import Path

import polars as pl


def article_stats(clickstream: Path, ids: pl.DataFrame) -> pl.DataFrame:
    """articles table: views_in, onward_clicks, onward_ratio, search_entry_share (+ entry mix)."""
    cs = pl.scan_parquet(clickstream)
    inbound = (
        cs.group_by("curr")
        .agg(
            pl.col("n").sum().alias("views_in"),
            pl.col("n").filter(pl.col("prev") == "other-search").sum().alias("search_in"),
            pl.col("n").filter(pl.col("prev") == "other-internal").sum().alias("internal_in"),
            pl.col("n").filter(pl.col("type") == "link").sum().alias("link_in"),
        )
        .rename({"curr": "title"})
    )
    onward = (
        cs.filter(pl.col("type") == "link")
        .group_by("prev")
        .agg(pl.col("n").sum().alias("onward_clicks"))
        .rename({"prev": "title"})
    )
    return (
        ids.lazy()
        .join(inbound, on="title", how="left")
        .join(onward, on="title", how="left")
        .with_columns(pl.exclude("article_id", "title").fill_null(0))
        .with_columns(
            (pl.col("onward_clicks") / pl.col("views_in"))
            .fill_nan(0)
            .clip(0, 5)
            .alias("onward_ratio"),
            (pl.col("search_in") / pl.col("views_in")).fill_nan(0).alias("search_entry_share"),
        )
        .sort("article_id")
        .collect()
    )


def edges(clickstream: Path, ids: pl.DataFrame) -> pl.DataFrame:
    """Link edges inside the subset; p = n / sum of the source's link clicks within the subset."""
    src = ids.rename({"title": "prev", "article_id": "src_id"}).lazy()
    dst = ids.rename({"title": "curr", "article_id": "dst_id"}).lazy()
    return (
        pl.scan_parquet(clickstream)
        .filter(pl.col("type") == "link")
        .join(src, on="prev")
        .join(dst, on="curr")
        .filter(pl.col("src_id") != pl.col("dst_id"))
        .group_by("src_id", "dst_id")
        .agg(pl.col("n").sum())
        .with_columns((pl.col("n") / pl.col("n").sum().over("src_id")).alias("p"))
        .sort("src_id", "dst_id")
        .collect()
    )
