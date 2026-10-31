"""Dagster assets for the offline pipeline. Each asset writes Parquet under data/."""

from pathlib import Path
from typing import Any

import dagster as dg
import polars as pl

from rabbithole.config import REPO_ROOT, Settings, get_settings
from rabbithole.graph import communities as comm
from rabbithole.graph import pruned, report
from rabbithole.ingest import clickstream, hf_text, stats, subset, text
from rabbithole.ingest.dev_seeds import DEV_SEEDS


def processed(settings: Settings, name: str) -> Path:
    path = settings.data_dir / "processed" / settings.dataset_mode / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def clickstream_path(settings: Settings, month: str) -> Path:
    return settings.data_dir / "interim" / f"clickstream_{month}.parquet"


def feature_month(settings: Settings) -> str:
    """Graph and features come from the OLDEST month; later months are kept for LTR labels."""
    return clickstream.resolve_months(settings)[0]


def _write(df: pl.DataFrame, path: Path) -> dg.MaterializeResult[Any]:
    df.write_parquet(path)
    return dg.MaterializeResult(metadata={"rows": len(df), "path": str(path)})


@dg.asset(group_name="ingest")
def clickstream_months(context: dg.AssetExecutionContext) -> dg.MaterializeResult[Any]:
    """Download and parse the configured (or latest 3) clickstream months."""
    settings = get_settings()
    rows: dict[str, int] = {}
    for month in clickstream.resolve_months(settings):
        out = clickstream_path(settings, month)
        if not out.exists():
            clickstream.parse(clickstream.download(month, settings), out)
        rows[month] = pl.scan_parquet(out).select(pl.len()).collect().item()
        context.log.info(f"{month}: {rows[month]:,} rows")
    return dg.MaterializeResult(metadata={"rows_per_month": rows})


@dg.asset(group_name="ingest", deps=[clickstream_months])
def article_ids(context: dg.AssetExecutionContext) -> dg.MaterializeResult[Any]:
    settings = get_settings()
    cs = clickstream_path(settings, feature_month(settings))
    if settings.dataset_mode == "dev":
        titles = subset.dev_subset(cs, DEV_SEEDS, settings.dev_target_articles)
    else:
        titles = subset.full_subset(cs, settings.full_min_views)
    context.log.info(f"{settings.dataset_mode} subset: {len(titles):,} articles")
    return _write(subset.assign_ids(titles), processed(settings, "article_ids"))


@dg.asset(group_name="ingest", deps=[article_ids])
def articles() -> dg.MaterializeResult[Any]:
    settings = get_settings()
    ids = pl.read_parquet(processed(settings, "article_ids"))
    cs = clickstream_path(settings, feature_month(settings))
    return _write(stats.article_stats(cs, ids), processed(settings, "articles"))


@dg.asset(group_name="ingest", deps=[article_ids])
def edges() -> dg.MaterializeResult[Any]:
    settings = get_settings()
    ids = pl.read_parquet(processed(settings, "article_ids"))
    cs = clickstream_path(settings, feature_month(settings))
    return _write(stats.edges(cs, ids), processed(settings, "edges"))


@dg.asset(group_name="ingest", deps=[article_ids])
def article_text() -> dg.MaterializeResult[Any]:
    settings = get_settings()
    ids = pl.read_parquet(processed(settings, "article_ids"))
    if settings.dataset_mode == "dev":
        df = text.article_text_dev(ids, settings)
    else:
        df = hf_text.article_text_full(ids)
    return _write(df, processed(settings, "article_text"))


@dg.asset(group_name="ingest", deps=[article_text])
def chunks() -> dg.MaterializeResult[Any]:
    settings = get_settings()
    df = text.chunks(pl.read_parquet(processed(settings, "article_text")))
    return _write(df, processed(settings, "chunks"))


def report_path(settings: Settings, name: str) -> Path:
    suffix = "" if settings.dataset_mode == "dev" else f"_{settings.dataset_mode}"
    path = REPO_ROOT / "reports" / f"{name}{suffix}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@dg.asset(group_name="graph", deps=[edges, article_text])
def communities(context: dg.AssetExecutionContext) -> dg.MaterializeResult[Any]:
    """Leiden resolution sweep, final partition, labels, and the sanity-check reports."""
    settings = get_settings()
    e = pl.read_parquet(processed(settings, "edges"))
    texts = pl.read_parquet(processed(settings, "article_text")).sort("article_id")

    g = comm.undirected_graph(e, texts.height)
    vectors = comm.text_vectors(texts["lead_text"].to_list())
    metrics = comm.sweep(g, settings.leiden_resolutions, vectors)
    chosen = settings.leiden_resolution or comm.choose_resolution(metrics)
    context.log.info(f"Leiden resolution: {chosen:g}")

    membership = pl.DataFrame(
        {"article_id": texts["article_id"], "community_id": comm.leiden(g, chosen)}
    )
    table = comm.community_table(membership, comm.central_articles(e, membership), texts)

    metrics.write_parquet(processed(settings, "leiden_sweep"))
    membership.write_parquet(processed(settings, "article_communities"))
    table.write_parquet(processed(settings, "communities"))
    report_path(settings, "leiden_sweep").write_text(
        report.sweep_report(metrics, chosen, settings.dataset_mode)
    )
    report_path(settings, "communities_sample").write_text(
        report.sample_report(table, texts.select("article_id", "title"))
    )
    return dg.MaterializeResult(
        metadata={
            "resolution": chosen,
            "communities": table.height,
            "eligible": int(table["eligible"].sum()),
        }
    )


@dg.asset(group_name="graph", deps=[edges])
def edges_pruned() -> dg.MaterializeResult[Any]:
    settings = get_settings()
    e = pl.read_parquet(processed(settings, "edges"))
    return _write(
        pruned.prune(e, settings.prune_top_k, settings.hub_penalty),
        processed(settings, "edges_pruned"),
    )


def _check(passed: bool, **metadata: float | int | str) -> dg.AssetCheckResult:
    return dg.AssetCheckResult(passed=passed, metadata=metadata)


@dg.asset_check(asset=articles)
def articles_titles_and_views() -> dg.AssetCheckResult:
    df = pl.read_parquet(processed(get_settings(), "articles"))
    nulls = df["title"].null_count()
    no_views = df.filter(pl.col("views_in") <= 0).height
    return _check(nulls == 0 and no_views == 0, null_titles=nulls, zero_view_articles=no_views)


@dg.asset_check(asset=edges)
def edges_counts_and_probabilities() -> dg.AssetCheckResult:
    df = pl.read_parquet(processed(get_settings(), "edges"))
    bad_n = df.filter(pl.col("n") <= 0).height
    worst = df.group_by("src_id").agg(pl.col("p").sum()).select((pl.col("p") - 1).abs().max())
    max_dev = float(worst.item() or 0.0)
    return _check(bad_n == 0 and max_dev < 1e-6, nonpositive_n=bad_n, max_row_sum_error=max_dev)


@dg.asset_check(asset=article_text)
def article_text_match_rate() -> dg.AssetCheckResult:
    df = pl.read_parquet(processed(get_settings(), "article_text"))
    matched = df.filter(pl.col("lead_text").str.len_chars() > 0).height
    rate = matched / max(df.height, 1)
    return _check(rate >= 0.9, match_rate=round(rate, 4), matched=matched, total=df.height)


@dg.asset_check(asset=communities)
def every_article_in_one_community() -> dg.AssetCheckResult:
    settings = get_settings()
    ids = pl.read_parquet(processed(settings, "article_ids"))
    m = pl.read_parquet(processed(settings, "article_communities"))
    ok = m.height == ids.height and m["article_id"].n_unique() == ids.height
    return _check(ok, articles=ids.height, assigned=m["article_id"].n_unique())


@dg.asset_check(asset=edges_pruned)
def pruned_out_degree_and_costs() -> dg.AssetCheckResult:
    settings = get_settings()
    df = pl.read_parquet(processed(settings, "edges_pruned"))
    max_out = df.group_by("src_id").len().select(pl.col("len").max()).item()
    min_cost = df.select(pl.col("cost").min()).item()
    ok = max_out <= settings.prune_top_k and min_cost >= 0
    return _check(ok, max_out_degree=max_out, min_cost=min_cost)


defs = dg.Definitions(
    assets=[
        clickstream_months,
        article_ids,
        articles,
        edges,
        article_text,
        chunks,
        communities,
        edges_pruned,
    ],
    asset_checks=[
        articles_titles_and_views,
        edges_counts_and_probabilities,
        article_text_match_rate,
        every_article_in_one_community,
        pruned_out_degree_and_costs,
    ],
)
