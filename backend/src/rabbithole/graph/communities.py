"""Rabbit holes = Leiden communities of the undirected click graph (DATA_AND_ML §2)."""

import random

import igraph as ig
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

MIN_COMMUNITY_SIZE = 8


def undirected_graph(edges: pl.DataFrame, n_articles: int) -> ig.Graph:
    """Undirected graph over all articles; w(a, b) = n(a->b) + n(b->a)."""
    pairs = (
        edges.with_columns(
            pl.min_horizontal("src_id", "dst_id").alias("a"),
            pl.max_horizontal("src_id", "dst_id").alias("b"),
        )
        .group_by("a", "b")
        .agg(pl.col("n").sum().alias("weight"))
        .sort("a", "b")
    )
    return ig.Graph(
        n=n_articles,
        edges=list(zip(pairs["a"].to_list(), pairs["b"].to_list(), strict=True)),
        edge_attrs={"weight": pairs["weight"].cast(pl.Float64).to_list()},
    )


def leiden(g: ig.Graph, resolution: float, seed: int = 42) -> list[int]:
    """Modularity-based Leiden, deterministic for a given seed. Returns membership per vertex."""
    ig.set_random_number_generator(random.Random(seed))
    part = g.community_leiden(
        objective_function="modularity", weights="weight", resolution=resolution, n_iterations=-1
    )
    return list(part.membership)


def coherence(membership: list[int], vectors: np.ndarray) -> float:
    """Mean intra-community cosine similarity of unit-length embeddings, averaged over
    communities with >= 8 members.

    For unit vectors, the sum of pairwise cosines is (|sum v|^2 - m) / 2, so no m x m matrix.
    """
    labels = np.asarray(membership)
    scores: list[float] = []
    for c in np.unique(labels):
        rows = vectors[labels == c]
        m = rows.shape[0]
        if m < MIN_COMMUNITY_SIZE:
            continue
        total = rows.sum(axis=0)
        scores.append(float((total @ total - np.square(rows).sum()) / (m * (m - 1))))
    return float(np.mean(scores)) if scores else 0.0


def sweep(g: ig.Graph, resolutions: list[float], vectors: np.ndarray) -> pl.DataFrame:
    """Leiden at each resolution: modularity (at resolution 1), size distribution, coherence.

    `coherence_lift` subtracts the mean cosine of *random* article pairs: sentence embeddings
    are anisotropic (any two bge vectors score ~0.46), so raw coherence barely moves.
    """
    baseline = coherence([0] * len(vectors), vectors) if len(vectors) >= MIN_COMMUNITY_SIZE else 0.0
    rows: list[dict[str, float]] = []
    for r in resolutions:
        membership = leiden(g, r)
        sizes = np.bincount(membership)
        eligible = sizes[sizes >= MIN_COMMUNITY_SIZE]
        rows.append(
            {
                "resolution": r,
                "communities": len(sizes),
                "eligible": len(eligible),
                "median_size": float(np.median(eligible)) if len(eligible) else 0.0,
                "max_size": int(sizes.max()),
                "share_in_eligible": float(eligible.sum() / sizes.sum()),
                "modularity": g.modularity(membership, weights="weight"),
                "coherence": coherence(membership, vectors),
            }
        )
    return pl.DataFrame(rows).with_columns(
        (pl.col("coherence") - baseline).alias("coherence_lift"),
        (pl.col("modularity") * (pl.col("coherence") - baseline)).alias("score"),
    )


def choose_resolution(
    metrics: pl.DataFrame, size_band: tuple[int, int] = (20, 200), tolerance: float = 0.05
) -> float:
    """Finest resolution whose score (modularity x coherence lift) is within `tolerance` of the
    best, among resolutions with median eligible community size in `size_band`.

    Modularity alone favours a few huge communities, coherence alone favours tiny ones. The
    product is often flat near its peak; then we prefer smaller, more specific rabbit holes.
    Falls back to all resolutions if none lands in the band.
    """
    lo, hi = size_band
    in_band = metrics.filter(pl.col("median_size").is_between(lo, hi))
    pool = in_band if in_band.height else metrics
    best = float(pool["score"].max())  # pyright: ignore[reportArgumentType]
    near = pool.filter(pl.col("score") >= best * (1 - tolerance))
    return float(near["resolution"].max())  # pyright: ignore[reportArgumentType]


def central_articles(edges: pl.DataFrame, membership: pl.DataFrame) -> pl.DataFrame:
    """Per community, members ranked by summed transition probability from other members.

    Each member casts at most one vote, split by where its readers go, so a single huge
    edge (e.g. a celebrity's page -> their child's page) cannot crown an article on its own.
    Raw weighted in-degree did exactly that in DEV; see DECISIONS.md.
    Returns article_id, community_id, centrality, rank (0 = most central).
    """
    m = membership.select("article_id", "community_id")
    internal = (
        edges.join(m.rename({"article_id": "src_id", "community_id": "c_src"}), on="src_id")
        .join(m.rename({"article_id": "dst_id", "community_id": "c_dst"}), on="dst_id")
        .filter(pl.col("c_src") == pl.col("c_dst"))
        .group_by("dst_id")
        .agg(pl.col("p").sum().alias("centrality"))
        .rename({"dst_id": "article_id"})
    )
    return (
        m.join(internal, on="article_id", how="left")
        .with_columns(pl.col("centrality").fill_null(0.0))
        .sort(["community_id", "centrality", "article_id"], descending=[False, True, False])
        .with_columns(pl.int_range(pl.len()).over("community_id").alias("rank"))
    )


def ctfidf_keywords(
    community_ids: list[int], docs: list[str], top_n: int = 5
) -> dict[int, list[str]]:
    """c-TF-IDF: one document per community; words frequent in it but rare across communities.

    tf normalized per class; idf = log(1 + mean words per class / word frequency across classes).
    """
    # No min_df: a word that appears in only one community is the best possible keyword.
    vec = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
    counts = csr_matrix(vec.fit_transform(docs)).astype(np.float64)
    vocab = np.asarray(vec.get_feature_names_out())
    tf = csr_matrix(normalize(counts, norm="l1", axis=1))
    word_totals = np.asarray(counts.sum(axis=0)).ravel()
    avg_words = float(counts.sum()) / len(community_ids)
    idf = np.log1p(avg_words / np.maximum(word_totals, 1))
    scores = tf.multiply(idf).tocsr()
    out: dict[int, list[str]] = {}
    for i, cid in enumerate(community_ids):
        row = scores.getrow(i).toarray().ravel()
        top = np.argsort(-row, kind="stable")[:top_n]
        out[cid] = [str(vocab[j]) for j in top if row[j] > 0]
    return out


def community_table(
    membership: pl.DataFrame, centrality: pl.DataFrame, texts: pl.DataFrame, top_k: int = 10
) -> pl.DataFrame:
    """communities table: id, size, eligible, central article, label, keywords, top articles.

    `membership`: article_id, community_id. `texts`: article_id, title, lead_text.
    """
    ranked = centrality.join(texts.select("article_id", "title"), on="article_id")
    summary = (
        ranked.group_by("community_id")
        .agg(
            pl.len().alias("size"),
            pl.col("article_id").sort_by("rank").first().alias("central_article_id"),
            pl.col("title").sort_by("rank").first().alias("central_title"),
            pl.col("article_id").sort_by("rank").head(top_k).alias("top_article_ids"),
        )
        .with_columns((pl.col("size") >= MIN_COMMUNITY_SIZE).alias("eligible"))
        .sort("community_id")
    )
    docs = (
        membership.join(texts.select("article_id", "lead_text"), on="article_id")
        .group_by("community_id")
        .agg(pl.col("lead_text").fill_null("").str.join(" "))
        .sort("community_id")
    )
    keywords = ctfidf_keywords(docs["community_id"].to_list(), docs["lead_text"].to_list())
    return summary.with_columns(
        pl.col("community_id")
        .map_elements(lambda c: keywords.get(c, []), return_dtype=pl.List(pl.String))
        .alias("keywords"),
    ).with_columns(
        (
            pl.col("central_title").str.replace_all("_", " ")
            + pl.lit(" (")
            + pl.col("keywords").list.head(3).list.join(", ")
            + pl.lit(")")
        ).alias("label")
    )
