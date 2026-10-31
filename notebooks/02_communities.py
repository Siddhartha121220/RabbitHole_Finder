import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import adjusted_mutual_info_score

    from rabbithole.config import REPO_ROOT, get_settings
    from rabbithole.graph import communities as comm
    from rabbithole.pipelines.definitions import processed

    settings = get_settings()
    FIG = REPO_ROOT / "reports" / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    edges = pl.read_parquet(processed(settings, "edges"))
    leiden = pl.read_parquet(processed(settings, "article_communities")).sort("article_id")
    mo.md(
        "# Leiden (click graph) vs HDBSCAN (text)\n\n"
        "Leiden groups articles that **readers move between**; HDBSCAN groups articles whose "
        "**text is similar** (sentence embeddings, reduced to 10-d with UMAP). "
        "Which gives better rabbit holes?"
    )
    return (
        FIG,
        HDBSCAN,
        adjusted_mutual_info_score,
        comm,
        edges,
        leiden,
        mo,
        np,
        pl,
        plt,
        processed,
        settings,
    )


@app.cell
def _(HDBSCAN, np, pl, processed, settings):
    import umap

    embeddings = pl.read_parquet(processed(settings, "article_embeddings")).sort("id")
    vecs = embeddings["embedding"].to_numpy()
    # HDBSCAN struggles in 384-d; reduce to 10-d first (the usual BERTopic-style recipe).
    reduced = umap.UMAP(n_components=10, metric="cosine", random_state=0).fit_transform(vecs)
    hdb_labels = HDBSCAN(min_cluster_size=8, min_samples=4, copy=True).fit_predict(reduced)
    hdb_labels = np.asarray(hdb_labels)
    return hdb_labels, vecs


@app.cell
def _(comm, edges, np, pl, vecs):
    def stickiness(labels: np.ndarray) -> float:
        """Share of link clicks whose source and target are in the same cluster (noise excluded)."""
        lab = pl.DataFrame({"id": np.arange(len(labels)), "c": labels})
        e = edges.join(lab.rename({"id": "src_id", "c": "cs"}), on="src_id").join(
            lab.rename({"id": "dst_id", "c": "cd"}), on="dst_id"
        )
        e = e.filter(pl.col("cs") >= 0)
        return float(e.filter(pl.col("cs") == pl.col("cd"))["n"].sum() / e["n"].sum())

    def summarize(name: str, labels: np.ndarray) -> dict[str, object]:
        clustered = labels >= 0
        sizes = np.bincount(labels[clustered])
        sizes = sizes[sizes >= 8]
        # Noise points get unique ids so they never count as a community in coherence.
        lab = np.where(clustered, labels, -np.arange(1, len(labels) + 1) - labels.max())
        return {
            "method": name,
            "clusters (>=8)": len(sizes),
            "median size": float(np.median(sizes)),
            "noise share": round(1 - clustered.mean(), 3),
            "embedding coherence": round(comm.coherence(lab.tolist(), vecs), 4),
            "click stickiness": round(stickiness(labels), 3),
        }

    return stickiness, summarize


@app.cell
def _(adjusted_mutual_info_score, comm, edges, hdb_labels, leiden, mo, np, pl, summarize):
    leiden_labels = leiden["community_id"].to_numpy()

    # Fair comparison: restrict both partitions to the articles HDBSCAN did not discard,
    # then compare modularity (stickiness relative to what cluster sizes alone would give).
    kept = np.where(hdb_labels >= 0)[0]
    index = {int(a): i for i, a in enumerate(kept)}
    shared = edges.filter(
        pl.col("src_id").is_in(kept.tolist()) & pl.col("dst_id").is_in(kept.tolist())
    ).with_columns(pl.col("src_id").replace_strict(index), pl.col("dst_id").replace_strict(index))
    g_shared = comm.undirected_graph(shared, len(kept))

    rows = []
    for name, labels in [("Leiden (clicks)", leiden_labels), ("HDBSCAN (text)", hdb_labels)]:
        row = summarize(name, labels)
        row["modularity on shared articles"] = round(
            g_shared.modularity(labels[kept].tolist(), weights="weight"), 3
        )
        rows.append(row)
    comparison = pl.DataFrame(rows)
    ami = adjusted_mutual_info_score(leiden_labels[kept], hdb_labels[kept])
    mo.vstack([comparison, mo.md(f"Adjusted mutual information on shared articles: **{ami:.3f}**")])
    return (comparison,)


@app.cell
def _(FIG, comparison, plt):
    cols = [
        "noise share",
        "embedding coherence",
        "click stickiness",
        "modularity on shared articles",
    ]
    fig, axes = plt.subplots(1, len(cols), figsize=(13, 3.5))
    for ax, col in zip(axes, cols, strict=True):
        ax.bar(comparison["method"], comparison[col], color=["#6b4f3a", "#3a6b5c"])
        ax.set_title(col)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "02_leiden_vs_hdbscan.png", dpi=120)
    fig
    return


@app.cell
def _(mo):
    mo.md(
        """
**Reading it (DEV, 2026-07, bge-small embeddings):**

- The two methods find related but clearly different structure (AMI ~0.69): reader
  navigation and text similarity overlap without being the same thing.
- HDBSCAN wins on embedding coherence (expected: it clusters those embeddings) and on raw
  click stickiness (0.80 vs 0.71). The raw comparison is not fair, though. HDBSCAN labels
  **~17% of articles as noise** (~17% of views), mostly the bridging articles that any
  partition would struggle to keep inside, and those could never be recommended.
- Restricted to the same articles, **Leiden has clearly higher modularity** (0.68 vs 0.58):
  relative to what cluster sizes alone would give, Leiden's clusters keep readers inside more.
- Leiden also covers every article.

**Decision:** Leiden on the click graph for rabbit holes, since a rabbit hole is defined by
readers *staying inside*. Embedding coherence stays a secondary sanity check (and is part of
the resolution choice).
"""
    )
    return


if __name__ == "__main__":
    app.run()
