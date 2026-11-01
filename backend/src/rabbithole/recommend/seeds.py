"""Seeds from free text (DATA_AND_ML §4.1): top articles per interest above a similarity floor."""

from sqlalchemy import Connection

from rabbithole.config import Settings
from rabbithole.embed.embeddings import Encoder
from rabbithole.embed.search import ArticleHit, search_articles


def seeds_from_text(
    conn: Connection, interests: list[str], settings: Settings, model: Encoder | None = None
) -> list[ArticleHit]:
    """Up to `seeds_per_interest` hits per interest, deduplicated, in interest order."""
    seen: set[int] = set()
    seeds: list[ArticleHit] = []
    for interest in interests:
        hits = search_articles(
            conn,
            interest,
            settings,
            k=settings.seeds_per_interest,
            min_similarity=settings.seed_min_similarity,
            model=model,
        )
        for hit in hits:
            if hit.article_id not in seen:
                seen.add(hit.article_id)
                seeds.append(hit)
    return seeds
