"""Trail evaluation (DATA_AND_ML §5, RAG evaluation): faithfulness via an LLM judge, link
validity, gentle -> deep ordering, and a hand-check sheet to calibrate the judge.

    rabbithole judge-trails --n 50          # costs money: generates and judges 50 trails
    rabbithole judge-trails --agreement     # after filling human_verdict in trail_check.csv

Writes reports/trail_judge.csv (every step), reports/trail_check.csv (first 30 trails, with an
empty human_verdict column) and reports/trail_eval.md.
"""

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau
from sqlalchemy import select
from sqlalchemy.orm import Session

from rabbithole.config import REPO_ROOT, Settings
from rabbithole.db.models import Article, Community
from rabbithole.db.session import get_engine
from rabbithole.embed.embeddings import load_model
from rabbithole.llm.provider import AnthropicProvider, Usage, structured_call
from rabbithole.rag.trail import BuiltTrail, build_trail, gentle_to_deep
from rabbithole.recommend import catalog as cat
from rabbithole.recommend import ranker
from rabbithole.recommend.candidates import bridge_path, candidates

MAX_WITHOUT_CONFIRMATION = 200
CHECK_TRAILS = 30
VERDICTS = ("supported", "partial", "unsupported")
# USD per million tokens (input, output), from the published price list; for the estimate only.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
}
# Rough tokens per trail, measured on the first runs: generation and judging.
EST_TOKENS = {"gen": (3500, 600), "judge": (2500, 400)}

JUDGE_SYSTEM = """You check whether short explanations are supported by source passages.
For each numbered step you get an explanation ("why") and the passage it claims to be based on.
verdict:
- supported: every factual claim in the why is stated in or directly implied by the passage.
- partial: the main claim is supported but some detail (a name, number, date) is not.
- unsupported: the main claim is not in the passage, or contradicts it.
Judge only factual support, not style. Mentions of the reader's interests are not claims."""

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer"},
                    "verdict": {"type": "string", "enum": list(VERDICTS)},
                    "reason": {"type": "string"},
                },
                "required": ["step", "verdict", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


def cost(usage: Usage) -> float:
    pin, pout = PRICES.get(usage.model, (0.0, 0.0))
    return (usage.input_tokens * pin + usage.output_tokens * pout) / 1e6


def estimate(n: int, settings: Settings) -> float:
    def one(model: str, tokens: tuple[int, int]) -> float:
        pin, pout = PRICES.get(model, (0.0, 0.0))
        return (tokens[0] * pin + tokens[1] * pout) / 1e6

    return n * (
        one(settings.llm_model, EST_TOKENS["gen"]) + one(settings.judge_model, EST_TOKENS["judge"])
    )


@dataclass(frozen=True)
class Query:
    seed: Article
    community: Community
    bridge_ids: list[int]


def sample_queries(db: Session, catalog: cat.Catalog, n: int, seed: int = 5) -> list[Query]:
    """Realistic trail requests: a random seed article, the recommender's top rabbit hole for
    it, and the bridge path to it, exactly what the UI would ask for."""
    rng = random.Random(seed)
    ids = list(range(len(catalog.titles)))
    out: list[Query] = []
    while len(out) < n:
        s = rng.choice(ids)
        cands = candidates(catalog, [s])
        if len(cands.idx) == 0:
            continue
        j = ranker.heuristic(catalog, cands, 1, surprise=rng.random())[0]
        community = db.get(Community, int(catalog.community_ids[cands.idx[j]]))
        article = db.get(Article, s)
        if community is not None and article is not None:
            out.append(Query(article, community, bridge_path(cands, j)))
    return out


def judge(client: Any, model: str, trail: BuiltTrail) -> tuple[list[dict[str, Any]], Usage]:
    passages = {c.chunk_id: c.text for c in trail.context}
    blocks = [
        f"Step {i}\nwhy: {st.why}\npassage: {passages.get(st.chunk_id or -1, '(none)')}"
        for i, st in enumerate(trail.steps)
    ]
    data, usage = structured_call(
        client, model, JUDGE_SYSTEM, "\n\n".join(blocks), JUDGE_SCHEMA, effort="medium"
    )
    by_step = {v["step"]: v for v in (data or {}).get("verdicts", [])}
    return [
        by_step.get(i, {"verdict": "unjudged", "reason": "judge returned nothing"})
        for i in range(len(trail.steps))
    ], usage


def run(settings: Settings, n: int, confirmed: bool) -> Path:
    if not (settings.anthropic_api_key and settings.anthropic_api_key.get_secret_value()):
        raise SystemExit("ANTHROPIC_API_KEY is not set: the judge needs the API (costs money).")
    est = estimate(n, settings)
    print(f"{n} trails with {settings.llm_model}, judged by {settings.judge_model}: ~${est:.2f}")
    if n > MAX_WITHOUT_CONFIRMATION and not confirmed:
        raise SystemExit(
            f"Refusing to judge more than {MAX_WITHOUT_CONFIRMATION} trails "
            "without --yes (cost guard)."
        )

    provider = AnthropicProvider(settings)
    encoder = load_model(settings.embedding_model)
    catalog = cat.load(settings.data_dir / "processed" / settings.dataset_mode)
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    taus: list[float] = []
    spend = 0.0
    with Session(get_engine()) as db:
        url_of = dict(db.execute(select(Article.id, Article.url)).tuples().all())
        for q_i, q in enumerate(sample_queries(db, catalog, n)):
            interest = q.seed.title.replace("_", " ")
            trail = build_trail(
                db, q.community, [interest], {q.seed.id}, q.bridge_ids, provider, encoder, settings
            )
            stats["trails"] += 1
            stats["fallback"] += trail.model == "template"
            spend += cost(trail.usage)
            if trail.model == "template":
                continue
            verdicts, usage = judge(provider.client, settings.judge_model, trail)
            spend += cost(usage)
            gentle = {
                a.id: r for r, a in enumerate(gentle_to_deep([s.article for s in trail.steps]))
            }
            tau: Any = kendalltau(
                range(len(trail.steps)), [gentle[s.article.id] for s in trail.steps]
            )
            taus.append(float(tau[0]))  # tau statistic
            passages = {c.chunk_id: c.text for c in trail.context}
            for i, (st, v) in enumerate(zip(trail.steps, verdicts, strict=True)):
                stats[v["verdict"]] += 1
                stats["links_ok"] += url_of.get(st.article.id) == st.article.url
                stats["steps"] += 1
                rows.append(
                    {
                        "trail": q_i,
                        "rabbit_hole": q.community.label,
                        "interest": interest,
                        "step": i,
                        "title": st.article.title,
                        "why": st.why,
                        "passage": passages.get(st.chunk_id or -1, ""),
                        "judge_verdict": v["verdict"],
                        "judge_reason": v["reason"],
                        "human_verdict": "",
                    }
                )

    reports = REPO_ROOT / "reports"
    write_csv(reports / "trail_judge.csv", rows)
    write_csv(reports / "trail_check.csv", [r for r in rows if r["trail"] < CHECK_TRAILS])
    steps = max(stats["steps"], 1)
    md = reports / "trail_eval.md"
    md.write_text(
        f"# Trail evaluation ({settings.dataset_mode.upper()})\n\n"
        f"Generator `{settings.llm_model}`, judge `{settings.judge_model}`. "
        f"{stats['trails']} trails requested, {stats['fallback']} fell back to the template "
        f"({stats['fallback'] / max(stats['trails'], 1):.0%}); "
        f"{stats['steps']} LLM steps judged.\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"| Link validity (URL = articles table) | {stats['links_ok'] / steps:.1%} |\n"
        f"| Faithfulness: supported | {stats['supported'] / steps:.1%} |\n"
        f"| Faithfulness: supported + partial | "
        f"{(stats['supported'] + stats['partial']) / steps:.1%} |\n"
        f"| Unsupported | {stats['unsupported'] / steps:.1%} |\n"
        f"| Kendall tau, trail order vs gentleness (mean) | "
        f"{np.mean(taus) if taus else float('nan'):.2f} |\n"
        f"| API spend for this run | ${spend:.2f} |\n\n"
        f"Hand-check sheet: `reports/trail_check.csv` (first {CHECK_TRAILS} trails). Fill "
        "`human_verdict` with supported / partial / unsupported, then run "
        "`rabbithole judge-trails --agreement`.\n"
    )
    return md


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    labels = set(a) | set(b)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in labels) / n**2
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def agreement(path: Path | None = None) -> str:
    """Judge vs human on the hand-checked rows (rows with an empty human_verdict are skipped)."""
    path = path or REPO_ROOT / "reports" / "trail_check.csv"
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["human_verdict"].strip()]
    if not rows:
        return "No human verdicts filled in yet."
    human = [r["human_verdict"].strip().lower() for r in rows]
    judged = [r["judge_verdict"] for r in rows]
    raw = sum(h == j for h, j in zip(human, judged, strict=True)) / len(rows)
    return (
        f"Judge vs human on {len(rows)} steps: agreement {raw:.1%}, "
        f"Cohen's kappa {cohen_kappa(human, judged):.2f}"
    )
