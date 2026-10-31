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
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import adjusted_mutual_info_score
    from sklearn.preprocessing import normalize

    from rabbithole.config import REPO_ROOT, get_settings
    from rabbithole.graph import communities as comm
    from rabbithole.pipelines.definitions import processed

    settings = get_settings()
    FIG = REPO_ROOT / "reports" / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    edges = pl.read_parquet(processed(settings, "edges"))
    texts = pl.read_parquet(processed(settings, "article_text")).sort("article_id")
    leiden = pl.read_parquet(processed(settings, "article_communities")).sort("article_id")
    mo.md(
        "# Leiden (click graph) vs HDBSCAN (text)\n\n"
        "Leiden groups articles that **readers move between**; HDBSCAN groups articles whose "
        "**text is similar**. Which gives better rabbit holes? Text vectors are TF-IDF -> 100-d "
        "LSA for now; re-run with sentence embeddings after Phase 3."
    )
    return (
        FIG,
        HDBSCAN,
        TruncatedSVD,
        adjusted_mutual_info_score,
        comm,
        edges,
        leiden,
        mo,
        normalize,
        np,
        pl,
        plt,
        texts,
    )


@app.cell
def _(HDBSCAN, TruncatedSVD, comm, normalize, texts):
    tfidf = comm.text_vectors(texts["lead_text"].to_list())
    lsa = normalize(TruncatedSVD(n_components=100, random_state=0).fit_transform(tfidf))
    hdb_labels = HDBSCAN(min_cluster_size=8, min_samples=4).fit_predict(lsa)
    return hdb_labels, tfidf


@app.cell
def _(comm, edges, np, pl, tfidf):
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
            "text coherence": round(comm.coherence(lab.tolist(), tfidf), 4),
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
    cols = ["noise share", "text coherence", "click stickiness", "modularity on shared articles"]
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
**Reading it (DEV, 2026-07):**

- The two methods largely agree (AMI ~0.8): reader navigation and text similarity find
  mostly the same topics.
- HDBSCAN wins on text coherence (expected: it clusters the text vectors) and on raw click
  stickiness (0.85 vs 0.71). The raw comparison is not fair, though. HDBSCAN labels **43% of
  articles as noise** (about 28% of all views), and those are mostly the bridging articles that
  any partition would struggle to keep inside. Its larger clusters also make staying inside easier.
- Restricted to the same articles, **Leiden has the higher modularity** (0.635 vs 0.602):
  relative to what cluster sizes alone would give, Leiden's clusters keep readers inside more.
- Leiden also covers every article, so nothing becomes unrecommendable.

**Decision:** Leiden on the click graph for rabbit holes, since a rabbit hole is defined by
readers *staying inside*. Text coherence stays a secondary sanity check. Re-run with sentence
embeddings after Phase 3.
"""
    )
    return


if __name__ == "__main__":
    app.run()
