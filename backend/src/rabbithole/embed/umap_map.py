"""2D map coordinates (UMAP) for articles and communities."""

import numpy as np
import polars as pl


def umap_coords(vectors: np.ndarray, seed: int = 42) -> np.ndarray:
    """(n, 2) coordinates scaled to [0, 1]. Cosine metric, deterministic for a seed."""
    import umap

    reducer = umap.UMAP(
        n_components=2, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=seed
    )
    xy = np.asarray(reducer.fit_transform(vectors), dtype=np.float64)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    return (xy - lo) / np.where(hi > lo, hi - lo, 1.0)


def community_geometry(
    membership: pl.DataFrame, vectors: np.ndarray, xy: np.ndarray
) -> pl.DataFrame:
    """Per community: unit-length centroid of member embeddings and the mean member position.

    Map positions are the mean of member positions rather than a separate UMAP of centroids,
    so every community sits on top of its own articles.
    """
    labels = membership.sort("article_id")["community_id"].to_numpy()
    rows: list[dict[str, object]] = []
    for c in np.unique(labels):
        mask = labels == c
        centroid = vectors[mask].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        x, y = xy[mask].mean(axis=0)
        rows.append(
            {
                "community_id": int(c),
                "centroid": centroid.astype(np.float32).tolist(),
                "umap_x": float(x),
                "umap_y": float(y),
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("centroid").cast(pl.Array(pl.Float32, vectors.shape[1]))
    )
