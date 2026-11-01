"""All endpoints (ARCHITECTURE §5). Ranking runs on the in-memory catalog; text, titles and
URLs come from Postgres."""

import hashlib
import time
from collections.abc import Iterable
from enum import Enum
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from rabbithole.api import schemas as s
from rabbithole.api.services import Services
from rabbithole.db.models import Article, Community, Feedback, RecommendationLog, Trail
from rabbithole.embed.search import ArticleHit, search_articles
from rabbithole.recommend import ranker
from rabbithole.recommend.candidates import bridge_path, candidates
from rabbithole.recommend.seeds import seeds_from_text
from rabbithole.scoring.labels import first_sentence

router = APIRouter()
HOLES: list[str | Enum] = ["rabbit holes"]


def services(request: Request) -> Services:
    return request.app.state.services


Svc = Annotated[Services, Depends(services)]


def _sha(parts: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(p.strip().lower() for p in parts)).encode()).hexdigest()


def _articles(db: Session, ids: Iterable[int]) -> dict[int, Article]:
    rows = db.scalars(select(Article).where(Article.id.in_(set(ids))))
    return {a.id: a for a in rows}


def _ref(a: Article) -> s.ArticleRef:
    return s.ArticleRef(id=a.id, title=a.title.replace("_", " "), url=a.url)


def _hit(h: ArticleHit) -> s.ArticleHitOut:
    return s.ArticleHitOut(
        id=h.article_id,
        title=h.title.replace("_", " "),
        url=h.url,
        community_id=h.community_id,
        similarity=round(h.similarity, 4),
    )


def _breakdown(features: dict[str, Any] | None) -> dict[str, s.FeatureValue]:
    return {
        k: s.FeatureValue(value=v["value"], z=v["z"])
        for k, v in (features or {}).items()
        if isinstance(v, dict)
    }


def _summary(c: Community, arts: dict[int, Article]) -> dict[str, Any]:
    return {
        "id": c.id,
        "label": c.label,
        "keywords": c.keywords,
        "size": c.size,
        "rh_score": c.rh_score,
        "rh_breakdown": _breakdown(c.rh_features),
        "top_articles": [_ref(arts[i]) for i in c.top_article_ids[:5] if i in arts],
    }


def _community(db: Session, community_id: int) -> Community:
    c = db.get(Community, community_id)
    if c is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"rabbit hole {community_id} not found")
    return c


def why(path: list[s.ArticleRef], rh_score: float | None) -> str:
    """Template explanation built only from graph facts and the score."""
    if len(path) >= 2:
        hops = len(path) - 1
        via = f" via {', '.join(a.title for a in path[1:-1])}" if hops > 1 else ""
        text = (
            f"Readers of {path[0].title} reach {path[-1].title} in {hops} "
            f"click{'s' if hops > 1 else ''}{via}."
        )
    else:
        text = "Close in meaning to your interests, though readers rarely click across directly."
    if rh_score is not None:
        text += f" Deeper than {rh_score:.0f}% of rabbit holes."
    return text


@router.get("/health", response_model=s.Health, tags=["meta"])
def health(svc: Svc) -> s.Health:
    db = svc.database_ok()
    n = len(svc.catalog.community_ids)
    return s.Health(
        status="ok" if db and n > 0 else "degraded",
        database=db,
        model_loaded=True,  # load_services fails startup if the encoder cannot load
        catalog_loaded=n > 0,
        communities=n,
    )


@router.post("/seeds/search", response_model=list[s.ArticleHitOut], tags=["seeds"])
def seed_search(body: s.SeedSearchIn, svc: Svc) -> list[s.ArticleHitOut]:
    """Articles matching a free-text interest (autocomplete)."""
    with svc.engine.connect() as conn:
        hits = search_articles(conn, body.query, svc.settings, k=body.k, model=svc.encoder)
    return [_hit(h) for h in hits]


@router.post("/recommend", response_model=s.RecommendOut, tags=["recommend"])
def recommend(body: s.RecommendIn, svc: Svc) -> s.RecommendOut:
    """Heuristic v1: Goldilocks band x rabbit-holeness x bridge strength, MMR-diversified."""
    start = time.perf_counter()
    settings, catalog = svc.settings, svc.catalog
    with Session(svc.engine) as db:
        seeds = seeds_from_text(db.connection(), body.interests, settings, svc.encoder)
    if not seeds:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "none of the interests matched an article closely enough; try other words",
        )
    cands = candidates(
        catalog, [h.article_id for h in seeds], settings.shortlist_size, settings.max_hops
    )
    picks = ranker.heuristic(
        catalog,
        cands,
        body.k,
        surprise=body.surprise,
        sigma=settings.goldilocks_sigma,
        lam=settings.mmr_lambda,
    )
    ids = [int(catalog.community_ids[cands.idx[j]]) for j in picks]
    paths = {j: bridge_path(cands, j, settings.bridge_max_hops) for j in picks}

    with Session(svc.engine) as db:
        comms = {c.id: c for c in db.scalars(select(Community).where(Community.id.in_(ids)))}
        wanted = {a for p in paths.values() for a in p}
        wanted |= {a for c in comms.values() for a in c.top_article_ids[:5]}
        arts = _articles(db, wanted)
        out: list[s.Recommendation] = []
        for j, cid in zip(picks, ids, strict=True):
            c = comms[cid]
            path = [_ref(arts[a]) for a in paths[j]]
            out.append(
                s.Recommendation(**_summary(c, arts), bridge_path=path, why=why(path, c.rh_score))
            )
        db.add(
            RecommendationLog(
                interests_hash=_sha(body.interests),
                n_interests=len(body.interests),
                surprise=body.surprise,
                community_ids=ids,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        )
        db.commit()
    return s.RecommendOut(seeds=[_hit(h) for h in seeds], rabbit_holes=out)


@router.get("/rabbit-holes/{community_id}", response_model=s.RabbitHoleDetail, tags=HOLES)
def rabbit_hole(community_id: int, svc: Svc) -> s.RabbitHoleDetail:
    with Session(svc.engine) as db:
        c = _community(db, community_id)
        arts = _articles(db, [c.central_article_id, *c.top_article_ids[:5]])
        central = arts[c.central_article_id]
    return s.RabbitHoleDetail(
        **_summary(c, arts),
        central_article=_ref(central),
        about=first_sentence(central.lead_text, max_chars=400),
        umap_x=c.umap_x,
        umap_y=c.umap_y,
    )


TEMPLATE_MODEL = "template"


@router.post("/rabbit-holes/{community_id}/trail", response_model=s.TrailOut, tags=HOLES)
def trail(community_id: int, body: s.TrailIn, request: Request, svc: Svc) -> s.TrailOut:
    """Reading trail, gentle to deep. Phase 6 serves the template trail (most-read first,
    each step explained by its own opening sentence); Phase 7 swaps in RAG + LLM."""
    client = request.client.host if request.client else "unknown"
    if not request.app.state.trail_limiter.allow(client):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "too many trail requests")
    seed_key = _sha(body.seed_titles)
    with Session(svc.engine) as db:
        c = _community(db, community_id)
        cached = db.scalars(
            select(Trail.payload).where(
                Trail.community_id == community_id,
                Trail.seed_key == seed_key,
                Trail.model == TEMPLATE_MODEL,
            )
        ).first()
        if cached is not None:
            return s.TrailOut.model_validate(cached)
        arts = _articles(db, c.top_article_ids)
        seeds = {t.strip().lower().replace(" ", "_") for t in body.seed_titles}
        ordered = sorted(
            (a for a in arts.values() if a.title.lower() not in seeds), key=lambda a: -a.views
        )
        steps = [
            s.TrailStep(**_ref(a).model_dump(), why=first_sentence(a.lead_text, max_chars=200))
            for a in ordered[:7]
        ]
        out = s.TrailOut(rabbit_hole_id=community_id, model=TEMPLATE_MODEL, steps=steps)
        db.add(
            Trail(
                community_id=community_id,
                seed_key=seed_key,
                model=TEMPLATE_MODEL,
                payload=out.model_dump(),
            )
        )
        db.commit()
    return out


@router.get("/map", response_model=list[s.MapPoint], tags=["map"])
def map_points(svc: Svc) -> list[s.MapPoint]:
    """Every eligible rabbit hole with its map position (gzip-compressed when large)."""
    with Session(svc.engine) as db:
        rows = db.scalars(select(Community).where(Community.eligible)).all()
    return [
        s.MapPoint(id=c.id, label=c.label, size=c.size, rh_score=c.rh_score, x=c.umap_x, y=c.umap_y)
        for c in rows
    ]


@router.post("/feedback", status_code=status.HTTP_201_CREATED, tags=["feedback"])
def feedback(body: s.FeedbackIn, svc: Svc) -> dict[str, str]:
    with Session(svc.engine) as db:
        db.add(Feedback(**body.model_dump()))
        db.commit()
    return {"status": "stored"}
