"""Semantic article search over pgvector: seeds for the recommender and /seeds/search."""

from dataclasses import dataclass

import numpy as np
from sqlalchemy import Connection, text

from rabbithole.config import Settings
from rabbithole.embed.embeddings import Encoder, embed_query


@dataclass(frozen=True)
class ArticleHit:
    article_id: int
    title: str
    url: str
    community_id: int
    similarity: float


def nearest_articles(
    conn: Connection, vector: np.ndarray, k: int = 10, min_similarity: float = 0.0
) -> list[ArticleHit]:
    """Top-k articles by cosine similarity (HNSW index), filtered by a similarity floor."""
    rows = conn.execute(
        text(
            "SELECT id, title, url, community_id, 1 - (embedding <=> CAST(:q AS vector)) AS sim "
            "FROM articles ORDER BY embedding <=> CAST(:q AS vector) LIMIT :k"
        ),
        {"q": str(vector.tolist()), "k": k},
    )
    return [
        ArticleHit(int(i), t, u, int(c), float(s)) for i, t, u, c, s in rows if s >= min_similarity
    ]


def search_articles(
    conn: Connection,
    query: str,
    settings: Settings,
    k: int = 10,
    min_similarity: float = 0.0,
    model: Encoder | None = None,
) -> list[ArticleHit]:
    """Free-text query -> nearest articles. Empty or blank queries return nothing."""
    if not query.strip():
        return []
    return nearest_articles(conn, embed_query(query, settings, model), k, min_similarity)
