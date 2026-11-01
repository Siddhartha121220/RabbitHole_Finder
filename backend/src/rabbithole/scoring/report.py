"""Face-validity report: the 20 deepest and 20 shallowest rabbit holes."""

import polars as pl

from rabbithole.graph.report import md_table
from rabbithole.scoring.features import FEATURES


def extremes_report(scored: pl.DataFrame, titles: pl.DataFrame, n: int = 20) -> str:
    """`scored`: eligible communities with label, top_article_ids, FEATURES, rh_score."""
    name = dict(zip(titles["article_id"].to_list(), titles["title"].to_list(), strict=True))

    def section(df: pl.DataFrame) -> str:
        table = df.select(
            pl.col("rh_score").round(1),
            "label",
            "size",
            *[pl.col(f).round(3) for f in FEATURES],
            pl.col("top_article_ids")
            .map_elements(
                lambda ids: ", ".join(name[i].replace("_", " ") for i in ids[:5]),
                return_dtype=pl.String,
            )
            .alias("top articles"),
        )
        return md_table(table)

    ranked = scored.sort("rh_score", descending=True)
    return (
        "# Rabbit-holeness extremes\n\n"
        "Face-validity check: the top should look like classic rabbit holes, the bottom like "
        "shallow look-up topics. `expected_depth` = expected pages read inside before leaving "
        "(absorbing Markov chain); see DATA_AND_ML.md §3.\n\n"
        f"## Deepest {n}\n\n{section(ranked.head(n))}\n\n"
        f"## Shallowest {n}\n\n{section(ranked.tail(n).reverse())}\n"
    )
