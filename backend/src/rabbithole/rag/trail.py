"""Grounded reading trail (DATA_AND_ML §5): retrieve -> order -> generate -> validate.

Whatever the LLM returns, only article ids from the retrieved candidate set survive, and
titles and URLs are attached here from the articles table.
"""

from dataclasses import dataclass

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from rabbithole.config import Settings
from rabbithole.db.models import Article, Community
from rabbithole.embed.embeddings import Encoder, embed_query
from rabbithole.llm.provider import (
    CandidateArticle,
    LLMProvider,
    RetrievedChunk,
    TemplateProvider,
    TrailDraft,
    TrailRequest,
    Usage,
)

N_CANDIDATES = 30
N_CHUNKS = 12
PER_ARTICLE = 2
MIN_STEPS, MAX_STEPS = 5, 7


@dataclass(frozen=True)
class Step:
    article: Article
    why: str
    chunk_id: int | None


@dataclass(frozen=True)
class BuiltTrail:
    intro: str
    steps: list[Step]
    model: str  # provider that produced the steps ("template" after a fallback)
    usage: Usage
    fallback_reason: str | None
    context: list[RetrievedChunk]


def candidates(
    db: Session, community: Community, exclude: set[int], bridge_ids: list[int]
) -> list[Article]:
    """Top articles by centrality (community.top_article_ids), then by views, plus bridge
    articles; seeds excluded. At most N_CANDIDATES + bridge."""
    ids = [i for i in community.top_article_ids if i not in exclude]
    by_views = db.scalars(
        select(Article.id)
        .where(Article.community_id == community.id)
        .order_by(Article.views.desc())
        .limit(N_CANDIDATES + len(exclude))
    )
    ids += [i for i in by_views if i not in exclude and i not in ids]
    ids = ids[:N_CANDIDATES] + [i for i in bridge_ids if i not in exclude and i not in ids]
    rows = {a.id: a for a in db.scalars(select(Article).where(Article.id.in_(ids)))}
    return [rows[i] for i in ids if i in rows]


def gentle_to_deep(articles: list[Article]) -> list[Article]:
    """Gentler = more views and more arrivals from search (an entry point people look up);
    deeper = less viewed and mostly reached by clicking around."""
    if len(articles) < 2:
        return articles

    def z(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() or 1.0)

    views = np.log1p([a.views for a in articles])
    search = np.array([a.search_share or 0.0 for a in articles])
    order = np.argsort(-(z(views) + z(search)), kind="stable")
    return [articles[i] for i in order]


def retrieve(db: Session, query_vec: np.ndarray, articles: list[Article]) -> list[RetrievedChunk]:
    """Top N_CHUNKS chunks by cosine within the candidate articles, max PER_ARTICLE each."""
    titles = {a.id: a.title for a in articles}
    rows = db.execute(
        text(
            "SELECT id, article_id, text FROM chunks WHERE article_id = ANY(:ids) "
            "ORDER BY embedding <=> CAST(:q AS vector) LIMIT :n"
        ),
        {"ids": list(titles), "q": str(query_vec.tolist()), "n": N_CHUNKS * 4},
    )
    per: dict[int, int] = {}
    out: list[RetrievedChunk] = []
    for cid, aid, body in rows:
        if per.get(aid, 0) < PER_ARTICLE and len(out) < N_CHUNKS:
            per[aid] = per.get(aid, 0) + 1
            out.append(RetrievedChunk(int(cid), int(aid), titles[aid], body))
    return out


def validate(
    draft: TrailDraft, allowed: dict[int, Article], context: list[RetrievedChunk]
) -> list[Step]:
    """Keep steps whose article is a candidate and whose chunk (if any) belongs to it;
    drop duplicates; cap at MAX_STEPS. Anything invented is silently dropped."""
    chunk_owner = {c.chunk_id: c.article_id for c in context}
    seen: set[int] = set()
    steps: list[Step] = []
    for st in draft.steps:
        ok_article = st.article_id in allowed and st.article_id not in seen
        ok_chunk = st.chunk_id is None or chunk_owner.get(st.chunk_id) == st.article_id
        if ok_article and ok_chunk and st.why.strip():
            seen.add(st.article_id)
            steps.append(Step(allowed[st.article_id], st.why.strip(), st.chunk_id))
    return steps[:MAX_STEPS]


def build_trail(
    db: Session,
    community: Community,
    interests: list[str],
    seed_ids: set[int],
    bridge_ids: list[int],
    provider: LLMProvider,
    encoder: Encoder,
    settings: Settings,
) -> BuiltTrail:
    ordered = gentle_to_deep(candidates(db, community, seed_ids, bridge_ids))
    template = TemplateProvider()

    def as_request(arts: list[Article]) -> TrailRequest:
        return TrailRequest(
            community.label,
            interests,
            [CandidateArticle(a.id, a.title, a.lead_text or "") for a in arts],
        )

    def fallback(reason: str | None, usage: Usage, ctx: list[RetrievedChunk]) -> BuiltTrail:
        # Bridge articles help the LLM connect topics; the template lists members only.
        members = [a for a in ordered if a.community_id == community.id]
        draft, _ = template.generate_trail(as_request(members), [])
        steps = validate(draft, {a.id: a for a in members}, [])
        return BuiltTrail(draft.intro, steps, template.name, usage, reason, ctx)

    if provider.name == template.name:
        return fallback(None, Usage(template.name), [])

    query = " ".join([*interests, community.label])
    context = retrieve(db, embed_query(query, settings, encoder), ordered)
    with_chunks = {c.article_id for c in context}
    offered = [a for a in ordered if a.id in with_chunks]  # keep the gentle -> deep order
    draft, usage = provider.generate_trail(as_request(offered), context)
    steps = validate(draft, {a.id: a for a in offered}, context)
    if len(steps) < MIN_STEPS:
        return fallback(f"only {len(steps)} valid steps from {provider.name}", usage, context)
    return BuiltTrail(draft.intro, steps, provider.name, usage, None, context)
