"""Combine features into the rabbit-holeness score / Depth Meter (DATA_AND_ML §3.3)."""

from typing import Any

import polars as pl

from rabbithole.scoring.features import FEATURES


def score(features: pl.DataFrame, weights: dict[str, float]) -> pl.DataFrame:
    """Add z_<feature>, rh_raw (weighted mean of z-scores) and rh_score (0-100 percentile).

    Z-scores and percentiles are computed over the rows given, i.e. pass eligible
    communities only so tiny clusters do not distort the scale.
    """
    total = sum(weights.get(f, 0.0) for f in FEATURES)
    z = [
        ((pl.col(f) - pl.col(f).mean()) / pl.col(f).std()).fill_nan(0).alias(f"z_{f}")
        for f in FEATURES
    ]
    return (
        features.with_columns(z)
        .with_columns(
            (
                pl.sum_horizontal(weights.get(f, 0.0) * pl.col(f"z_{f}") for f in FEATURES) / total
            ).alias("rh_raw")
        )
        .with_columns(
            # Percentile rank in (0, 100]: the deepest community scores 100.
            (pl.col("rh_raw").rank("average") / pl.len() * 100).alias("rh_score")
        )
    )


def breakdown(row: dict[str, Any]) -> dict[str, Any]:
    """JSON-ready breakdown for the UI: raw value and z-score per feature."""
    return {f: {"value": row[f], "z": row[f"z_{f}"]} for f in FEATURES} | {
        "rh_raw": row["rh_raw"],
    }
