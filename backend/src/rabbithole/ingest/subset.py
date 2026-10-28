"""Choose the article subset: DEV by deterministic BFS, FULL by a views threshold."""

import logging
from pathlib import Path

import polars as pl

from rabbithole.ingest.titles import is_excluded_expr

log = logging.getLogger(__name__)


def link_edges(clickstream: Path) -> pl.LazyFrame:
    """Article-to-article link clicks, excluding non-content pages at either end."""
    return (
        pl.scan_parquet(clickstream)
        .filter(pl.col("type") == "link")
        .filter(~is_excluded_expr("prev") & ~is_excluded_expr("curr"))
        .select("prev", "curr", "n")
    )


def dev_subset(
    clickstream: Path, seeds: tuple[str, ...], target: int, per_node: int = 20
) -> list[str]:
    """Level-by-level BFS from `seeds`, following each article's `per_node` most-clicked links.

    The level that would overshoot `target` is trimmed by click count, so the result is
    deterministic and every kept article is reachable from a seed.
    """
    edges = link_edges(clickstream).collect()
    known = set(edges["prev"].to_list())
    chosen = [s for s in seeds if s in known]
    missing = sorted(set(seeds) - set(chosen))
    if missing:
        log.warning("dev seeds not in clickstream: %s", missing)

    seen = set(chosen)
    frontier = chosen
    while frontier and len(seen) < target:
        nxt = (
            edges.filter(pl.col("prev").is_in(frontier))
            .sort(["prev", "n", "curr"], descending=[False, True, False])
            .group_by("prev", maintain_order=True)
            .head(per_node)
            .filter(~pl.col("curr").is_in(list(seen)))
            .group_by("curr")
            .agg(pl.col("n").sum())
            .sort(["n", "curr"], descending=[True, False])
            .head(target - len(seen))
        )
        frontier = nxt["curr"].to_list()
        seen.update(frontier)
        chosen.extend(frontier)
    return chosen


def full_subset(clickstream: Path, min_views: int) -> list[str]:
    """Every content article with at least `min_views` incoming views in the month."""
    return (
        pl.scan_parquet(clickstream)
        .group_by("curr")
        .agg(pl.col("n").sum().alias("views"))
        .filter((pl.col("views") >= min_views) & ~is_excluded_expr("curr"))
        .sort("curr")
        .collect()["curr"]
        .to_list()
    )


def assign_ids(titles: list[str]) -> pl.DataFrame:
    """Stable integer ids: position in sorted title order."""
    return pl.DataFrame({"title": sorted(set(titles))}).with_row_index("article_id")
