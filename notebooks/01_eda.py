import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import polars as pl

    from rabbithole.config import REPO_ROOT, get_settings
    from rabbithole.pipelines.definitions import clickstream_path, feature_month, processed

    settings = get_settings()
    month = feature_month(settings)
    FIG = REPO_ROOT / "reports" / "figures"
    FIG.mkdir(parents=True, exist_ok=True)

    articles = pl.read_parquet(processed(settings, "articles"))
    edges = pl.read_parquet(processed(settings, "edges"))
    text = pl.read_parquet(processed(settings, "article_text"))
    cs = pl.scan_parquet(clickstream_path(settings, month))
    mo.md(
        f"# EDA — {settings.dataset_mode.upper()} subset, clickstream {month}\n\n"
        f"- articles: **{articles.height:,}**\n"
        f"- edges: **{edges.height:,}**\n"
        f"- lead text matched: **{text['lead_text'].is_not_null().mean():.1%}**"
    )
    return FIG, articles, cs, edges, mo, pl, plt


@app.cell
def _(FIG, articles, plt):
    # Views are heavy-tailed: a handful of hubs dominate; most articles get modest traffic.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(articles["views_in"].log10(), bins=50, color="#6b4f3a")
    axes[0].set(title="Views per article (log10)", xlabel="log10(views_in)", ylabel="articles")
    axes[1].hist(articles["onward_ratio"], bins=50, color="#3a6b5c")
    axes[1].set(
        title="Onward ratio (clipped to 5)", xlabel="onward clicks / views", ylabel="articles"
    )
    fig.tight_layout()
    fig.savefig(FIG / "01_views_onward.png", dpi=120)
    fig
    return


@app.cell
def _(FIG, edges, pl, plt):
    # Degree inside the subset: how many distinct articles readers click to / arrive from.
    out_deg = edges.group_by("src_id").len()["len"]
    in_deg = edges.group_by("dst_id").len()["len"]
    fig2, ax = plt.subplots(figsize=(6, 4))
    for deg, label in [(out_deg, "out-degree"), (in_deg, "in-degree")]:
        counts = deg.value_counts().sort(deg.name)
        ax.loglog(counts[deg.name], counts["count"], ".", label=label)
    ax.set(title="Degree distribution (log-log)", xlabel="degree", ylabel="articles")
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(FIG / "01_degree.png", dpi=120)
    pl.DataFrame({"median_out_degree": [out_deg.median()], "median_in_degree": [in_deg.median()]})
    return


@app.cell
def _(articles, edges, mo, pl):
    titles = articles.select("article_id", "title")
    top = (
        edges.sort("n", descending=True)
        .head(20)
        .join(titles.rename({"article_id": "src_id", "title": "from"}), on="src_id")
        .join(titles.rename({"article_id": "dst_id", "title": "to"}), on="dst_id")
        .select("from", "to", "n", pl.col("p").round(3))
    )
    mo.vstack([mo.md("## Top 20 transitions inside the subset"), top])
    return


@app.cell
def _(FIG, cs, pl, plt):
    # Where do readers come from? Whole month, all articles.
    sources = (
        cs.with_columns(
            pl.when(pl.col("prev").str.starts_with("other-"))
            .then(pl.col("prev"))
            .otherwise(pl.lit("link (internal)"))
            .alias("source")
        )
        .group_by("source")
        .agg(pl.col("n").sum())
        .with_columns((pl.col("n") / pl.col("n").sum()).alias("share"))
        .sort("share", descending=True)
        .collect()
    )
    fig3, ax3 = plt.subplots(figsize=(7, 3.5))
    ax3.barh(sources["source"], sources["share"], color="#8a6d3b")
    ax3.invert_yaxis()
    ax3.set(title="Entry source share (all enwiki)", xlabel="share of page views")
    fig3.tight_layout()
    fig3.savefig(FIG / "01_entry_sources.png", dpi=120)
    sources
    return


if __name__ == "__main__":
    app.run()
