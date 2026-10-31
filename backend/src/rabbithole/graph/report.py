"""Human-readable Markdown reports for community sanity checks."""

import random

import polars as pl


def _md_table(df: pl.DataFrame) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def sweep_report(metrics: pl.DataFrame, chosen: float, mode: str) -> str:
    return (
        f"# Leiden resolution sweep ({mode.upper()})\n\n"
        "Modularity is computed at resolution 1 for every row, so rows are comparable. "
        "Coherence = mean intra-community cosine similarity of article embeddings "
        "(communities with >= 8 articles); coherence_lift = coherence minus the mean cosine "
        "of random article pairs. score = modularity x coherence_lift.\n\n"
        f"{_md_table(metrics)}\n\n**Chosen resolution: {chosen:g}** "
        "(finest resolution within 5% of the best score, median community size in 20-200).\n"
    )


def sample_report(
    communities: pl.DataFrame, titles: pl.DataFrame, n: int = 20, seed: int = 7
) -> str:
    """`n` random eligible communities with their top 10 articles, for a human coherence check."""
    eligible = communities.filter("eligible").sort("community_id")
    picks = sorted(random.Random(seed).sample(range(eligible.height), min(n, eligible.height)))
    name = dict(zip(titles["article_id"].to_list(), titles["title"].to_list(), strict=True))
    out = [
        f"# Community sample ({len(picks)} of {eligible.height} eligible communities)\n",
        "Random sample (fixed seed). Articles are listed most central first. "
        "Each community should read as one coherent topic.\n",
    ]
    for i in picks:
        c = eligible.row(i, named=True)
        out.append(f"## {c['label']}\n")
        out.append(
            f"id {c['community_id']} · {c['size']} articles · "
            f"keywords: {', '.join(c['keywords'])}\n"
        )
        out.extend(
            f"{k + 1}. {name[a].replace('_', ' ')}" for k, a in enumerate(c["top_article_ids"])
        )
        out.append("")
    return "\n".join(out)
