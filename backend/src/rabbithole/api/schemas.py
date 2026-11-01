"""Request and response models (ARCHITECTURE §5). Every URL in a response comes from the
articles table, never from free text."""

from typing import Literal

from pydantic import BaseModel, Field


class ArticleRef(BaseModel):
    id: int
    title: str
    url: str


class ArticleHitOut(ArticleRef):
    community_id: int
    similarity: float
    x: float | None = None  # map position (filled by /recommend so seeds can glow on the map)
    y: float | None = None


class SeedSearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    k: int = Field(default=8, ge=1, le=20)


class RecommendIn(BaseModel):
    interests: list[str] = Field(min_length=1, max_length=5)
    surprise: float = Field(default=0.5, ge=0, le=1)
    k: int = Field(default=5, ge=1, le=10)
    session_id: str | None = Field(default=None, max_length=64)


class FeatureValue(BaseModel):
    value: float
    z: float


class RabbitHoleSummary(BaseModel):
    id: int
    label: str
    keywords: list[str]
    size: int
    rh_score: float | None
    rh_breakdown: dict[str, FeatureValue]
    top_articles: list[ArticleRef]


class Recommendation(RabbitHoleSummary):
    hook: str  # opening sentence of the community's central article
    x: float
    y: float
    bridge_path: list[ArticleRef]  # seed -> ... -> entry article; empty if none within 4 hops
    why: str


class RecommendOut(BaseModel):
    seeds: list[ArticleHitOut]
    rabbit_holes: list[Recommendation]


class RabbitHoleDetail(RabbitHoleSummary):
    central_article: ArticleRef
    about: str
    umap_x: float
    umap_y: float


class TrailIn(BaseModel):
    interests: list[str] = Field(default_factory=list, max_length=5)
    seed_titles: list[str] = Field(default_factory=list, max_length=10)  # left out of the trail
    bridge_article_ids: list[int] = Field(default_factory=list, max_length=10)


class TrailStep(ArticleRef):
    why: str


class TrailOut(BaseModel):
    rabbit_hole_id: int
    model: str  # "template" when no LLM is configured or the LLM output failed validation
    intro: str
    steps: list[TrailStep]


class MapPoint(BaseModel):
    id: int
    label: str
    size: int
    rh_score: float | None
    x: float
    y: float


class FeedbackIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    community_id: int
    seed_titles: list[str] = Field(default_factory=list, max_length=10)
    surprise: float = Field(ge=0, le=1)
    rating: Literal["up", "down", "obsessed"]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    model_loaded: bool
    catalog_loaded: bool
    communities: int
