"""`make eval`: offline evaluation of every system on future-navigation labels (DATA_AND_ML §6).

Writes reports/eval_<date>.md + figures and logs the same numbers to MLflow.
"""

import datetime as dt
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from rabbithole.config import REPO_ROOT, Settings
from rabbithole.eval import labels as lab
from rabbithole.eval import metrics as m
from rabbithole.recommend import ltr, ranker
from rabbithole.recommend.candidates import Candidates, candidates
from rabbithole.recommend.catalog import Catalog, load

log = logging.getLogger(__name__)

K_LIST = 10  # recall / MRR depth
K_SHOW = 5  # what a user sees: NDCG, novelty, diversity, serendipity, coverage
METRICS = ["ndcg@5", "recall@10", "mrr", "novelty", "diversity", "serendipity"]

System = Callable[[Candidates], list[int]]


@dataclass(frozen=True)
class SystemResult:
    name: str
    per_query: dict[str, np.ndarray]
    lists: list[list[int]]


def systems(catalog: Catalog, settings: Settings) -> dict[str, System]:
    rng = np.random.default_rng(0)
    sigma, lam = settings.goldilocks_sigma, settings.mmr_lambda

    def v1(**kw: object) -> System:
        return lambda c: ranker.heuristic(catalog, c, K_LIST, sigma=sigma, **kw)  # type: ignore[arg-type]

    systems_: dict[str, System] = {
        "Random": lambda c: ranker.random_ranker(c, K_LIST, rng),
        "Popularity": lambda c: ranker.popularity(catalog, c, K_LIST),
        "Nearest neighbour": lambda c: ranker.nearest(c, K_LIST),
        "Heuristic v1": v1(lam=None),
        "Heuristic v1 + MMR": v1(lam=lam),
        "v1 + MMR, no rh_score": v1(lam=lam, use_rh=False),
        "v1 + MMR, no bridge": v1(lam=lam, use_bridge=False),
        "v1 + MMR, surprise 0": v1(lam=lam, surprise=0.0),
        "v1 + MMR, surprise 1": v1(lam=lam, surprise=1.0),
    }
    if ltr.model_path(settings).exists():
        model = ltr.load_model(settings)

        def v2(**kw: object) -> System:
            return lambda c: ltr.rank(model, catalog, c, c.n_seeds, K_LIST, sigma=sigma, **kw)  # type: ignore[arg-type]

        systems_["LTR v2"] = v2(lam=None, band=False)
        systems_["LTR v2 + band + MMR"] = v2(lam=lam)
        systems_["LTR v2 + MMR, no band"] = v2(lam=lam, band=False)
    return systems_


def evaluate(catalog: Catalog, queries: list[lab.Query], settings: Settings) -> list[SystemResult]:
    pop_share = dict(
        zip(catalog.community_ids.tolist(), catalog.views / catalog.views.sum(), strict=True)
    )
    vec = dict(zip(catalog.community_ids.tolist(), catalog.centroids, strict=True))
    cands = [
        candidates(catalog, q.seeds, settings.shortlist_size, settings.max_hops) for q in queries
    ]

    def ids(c: Candidates, positions: list[int]) -> list[int]:
        return catalog.community_ids[c.idx[positions]].tolist()

    obvious = [ids(c, ranker.nearest(c, 10)) for c in cands]
    results: list[SystemResult] = []
    for name, system in systems(catalog, settings).items():
        lists = [ids(c, system(c)) for c in cands]
        rows = [
            {
                "ndcg@5": m.ndcg_at_k(r, q.relevance, K_SHOW),
                "recall@10": m.recall_at_k(r, q.relevance, K_LIST),
                "mrr": m.mrr(r, q.relevance),
                "novelty": m.novelty(r[:K_SHOW], pop_share),
                "diversity": m.diversity(r[:K_SHOW], vec),
                "serendipity": m.serendipity(r[:K_SHOW], q.relevance, o),
            }
            for r, q, o in zip(lists, queries, obvious, strict=True)
        ]
        per_query = {k: np.array([row[k] for row in rows]) for k in METRICS}
        results.append(SystemResult(name, per_query, [lst[:K_SHOW] for lst in lists]))
    return results


def summarise(
    results: list[SystemResult], n_catalog: int, n_boot: int
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Mean and 95% bootstrap CI per metric; coverage is bootstrapped over query resamples."""
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    for r in results:
        stats = {k: m.bootstrap_ci(v, n_boot) for k, v in r.per_query.items()}
        order = np.arange(len(r.lists))
        stats["coverage"] = m.bootstrap_ci(
            order,
            n_boot,
            stat=lambda ix, lists=r.lists: m.coverage([lists[i] for i in ix], n_catalog),
        )
        out[r.name] = stats
    return out


def _fmt(t: tuple[float, float, float]) -> str:
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def chart(summary: dict[str, dict[str, tuple[float, float, float]]], path: Path) -> None:
    """The money chart: accuracy (NDCG@5) against novelty, and against serendipity, with CIs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    panels = [
        ("novelty", "Novelty (mean -log2 popularity share; higher = less mainstream)"),
        ("serendipity", "Serendipity (relevant and not in nearest-neighbour top 10)"),
    ]
    for ax, (metric, xlabel) in zip(axes, panels, strict=True):
        points: list[tuple[float, float, str]] = []
        for name, s in summary.items():
            v1 = "v1" in name or "LTR" in name
            color, marker = ("#2a78d6", "o") if v1 else ("#52514e", "s")
            (x, xl, xh), (y, yl, yh) = s[metric], s["ndcg@5"]
            ax.errorbar(
                x,
                y,
                xerr=[[x - xl], [xh - x]],
                yerr=[[y - yl], [yh - y]],
                fmt=marker,
                color=color,
                ms=8,
                lw=1.5,
                capsize=3,
                mec="#fcfcfb",
                mew=1.5,
                label=("Heuristic v1 / LTR v2 variants" if v1 else "Baselines")
                if name in ("Random", "Heuristic v1")
                else None,
            )
            points.append((x, y, name))
        # Push labels apart vertically (top to bottom) so none overlap; thin leader lines.
        gap = 0.028
        dx = 0.03 * (max(p[0] for p in points) - min(p[0] for p in points))
        last = np.inf
        for x, y, name in sorted(points, key=lambda p: -p[1]):
            ly = min(y, last - gap)
            last = ly
            ax.annotate(
                name,
                (x, y),
                xytext=(x + dx, ly),
                textcoords="data",
                fontsize=8,
                color="#0b0b0b",
                ha="left",
                va="center",
                xycoords="data",
                annotation_clip=False,
            )
        ax.set_xlabel(xlabel)
        ax.grid(color="#e8e7e3", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes:
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + 0.25 * (hi - lo))  # room for labels on the right
    axes[0].set_ylabel("NDCG@5 on future-navigation labels")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Accuracy vs novelty and serendipity, all systems (95% bootstrap CIs)", x=0.01, ha="left"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(
    summary: dict[str, dict[str, tuple[float, float, float]]],
    results: list[SystemResult],
    meta: dict[str, object],
    figure: str,
) -> str:
    cols = [*METRICS, "coverage"]
    lines = [
        f"# Offline evaluation — {meta['date']} ({str(meta['mode']).upper()})\n",
        f"Feature month {meta['feature_month']}; labels from {meta['label_months']} "
        f"(3-step random walk, 1-hop mass excluded, graded 0-3). "
        f"{meta['queries']} test queries from {meta['test_communities']} held-out seed "
        f"communities; {meta['catalog']} eligible communities. "
        f"Mean [95% bootstrap CI, {meta['bootstrap']} resamples]. "
        "Relevant = grade >= 2. Top 5 shown to the user; recall/MRR use the top 10.\n",
        "| System | " + " | ".join(cols) + " |",
        "|---" * (len(cols) + 1) + "|",
    ]
    lines += [
        f"| {name} | " + " | ".join(_fmt(s[c]) for c in cols) + " |" for name, s in summary.items()
    ]
    by = {r.name: r for r in results}
    lines += [
        "",
        "## Paired bootstrap, NDCG@5 (same queries)\n",
        "| A vs B | mean diff [95% CI] | p |",
        "|---|---|---|",
    ]
    for a, b in [
        ("Heuristic v1 + MMR", "Nearest neighbour"),
        ("Heuristic v1 + MMR", "Popularity"),
        ("Heuristic v1 + MMR", "Random"),
        ("Heuristic v1 + MMR", "Heuristic v1"),
        ("Heuristic v1 + MMR", "v1 + MMR, no rh_score"),
        ("Heuristic v1 + MMR", "v1 + MMR, no bridge"),
        ("LTR v2", "Heuristic v1"),
        ("LTR v2 + band + MMR", "Heuristic v1 + MMR"),
        ("LTR v2 + band + MMR", "Nearest neighbour"),
    ]:
        if a not in by or b not in by:
            continue
        d, lo, hi, p = m.paired_bootstrap(by[a].per_query["ndcg@5"], by[b].per_query["ndcg@5"])
        lines.append(f"| {a} vs {b} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | {p:.3f} |")
    lines += ["", f"![accuracy vs novelty]({figure})", ""]
    return "\n".join(lines)


def log_mlflow(
    settings: Settings,
    summary: dict[str, dict[str, tuple[float, float, float]]],
    meta: dict[str, object],
    report_file: Path,
    figure_file: Path,
) -> None:
    """One run per system in the 'offline-eval' experiment; skipped if MLflow is down."""
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("offline-eval")
        for name, stats in summary.items():
            with mlflow.start_run(run_name=name):
                mlflow.log_params({**meta, "system": name})
                for metric, (mean, lo, hi) in stats.items():
                    key = metric.replace("@", "_at_")
                    mlflow.log_metrics({key: mean, f"{key}_ci_lo": lo, f"{key}_ci_hi": hi})
                mlflow.log_artifact(str(report_file))
                mlflow.log_artifact(str(figure_file))
    except Exception as exc:  # tracking is optional; the report on disk is the source of truth
        log.warning("MLflow logging skipped: %s", exc)


def run(
    settings: Settings, processed_dir: Path, feature_month: str, label_months: list[str]
) -> Path:
    catalog = load(processed_dir)
    t = lab.transition_matrix(
        pl.read_parquet(processed_dir / "label_edges.parquet"), len(catalog.titles)
    )
    test = lab.community_splits(catalog)["test"]
    queries = lab.sample_queries(catalog, t, test, settings.eval_queries)
    results = evaluate(catalog, queries, settings)
    summary = summarise(results, len(catalog.community_ids), settings.eval_bootstrap)

    date = dt.date.today().isoformat()
    suffix = "" if settings.dataset_mode == "dev" else f"_{settings.dataset_mode}"
    reports = REPO_ROOT / "reports"
    figure = reports / "figures" / f"eval_{date}{suffix}_accuracy_vs_novelty.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    chart(summary, figure)
    meta: dict[str, object] = {
        "date": date,
        "mode": settings.dataset_mode,
        "feature_month": feature_month,
        "label_months": ",".join(label_months),
        "queries": len(queries),
        "test_communities": len(test),
        "catalog": len(catalog.community_ids),
        "bootstrap": settings.eval_bootstrap,
        "surprise": 0.5,
        "sigma": settings.goldilocks_sigma,
        "mmr_lambda": settings.mmr_lambda,
    }
    out = reports / f"eval_{date}{suffix}.md"
    out.write_text(report(summary, results, meta, f"figures/{figure.name}"))
    log_mlflow(settings, summary, meta, out, figure)
    return out
