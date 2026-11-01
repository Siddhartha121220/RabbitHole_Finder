"""LLM providers for trails (ARCHITECTURE §6).

The model only ever sees article ids and chunk ids from our own tables and must answer with
ids chosen from them. URLs are never shown to it and never taken from it.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from rabbithole.config import Settings
from rabbithole.scoring.labels import first_sentence

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateArticle:
    id: int
    title: str
    lead: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    article_id: int
    title: str
    text: str


@dataclass(frozen=True)
class TrailRequest:
    rabbit_hole: str  # community label
    interests: list[str]
    articles: list[CandidateArticle]  # suggested order: gentle -> deep


class DraftStep(BaseModel):
    article_id: int
    chunk_id: int | None = None  # the chunk the "why" is based on (None for template)
    why: str = Field(max_length=400)


class TrailDraft(BaseModel):
    intro: str = Field(default="", max_length=600)
    steps: list[DraftStep]


@dataclass
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    extra: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())


class LLMProvider(Protocol):
    name: str

    def generate_trail(
        self, request: TrailRequest, context: list[RetrievedChunk]
    ) -> tuple[TrailDraft, Usage]: ...


class TemplateProvider:
    """No LLM: the suggested gentle -> deep order, each step explained by its own lead."""

    name = "template"

    def generate_trail(
        self, request: TrailRequest, context: list[RetrievedChunk]
    ) -> tuple[TrailDraft, Usage]:
        steps = [
            DraftStep(article_id=a.id, why=first_sentence(a.lead, max_chars=200) or a.title)
            for a in request.articles[:7]
        ]
        intro = f"A gentle way into {request.rabbit_hole}, most-read pages first."
        return TrailDraft(intro=intro, steps=steps), Usage(self.name)


TRAIL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intro": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "article_id": {"type": "integer"},
                    "chunk_id": {"type": "integer"},
                    "why": {"type": "string"},
                },
                "required": ["article_id", "chunk_id", "why"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["intro", "steps"],
    "additionalProperties": False,
}

TRAIL_SYSTEM = """You design reading trails through Wikipedia for curious people.
You get a topic cluster, the reader's interests, candidate articles (in a suggested
gentle-to-deep order) and numbered text passages from those articles.

Return 5 to 7 steps, ordered from easy entry points to deeper material. For each step:
- article_id: one of the candidate article ids, each used at most once.
- chunk_id: the passage your explanation is based on; it must belong to that article.
- why: one or two sentences (under 40 words) saying why this article is worth reading next,
  using only facts stated in that passage. Connect to the reader's interests where the
  passage allows; never invent facts, dates or names that are not in the passage.
intro: one sentence introducing the trail. Do not write URLs anywhere."""

# Models that accept server-side refusal fallbacks ("default" routing); Haiku does not.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}


def structured_call(
    client: Any,
    model: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    max_tokens: int = 4000,
    effort: str = "low",
) -> tuple[dict[str, Any] | None, Usage]:
    """One Messages call constrained to a JSON schema. Returns (None, usage) on a refusal
    or unparseable output so callers can fall back instead of crashing."""
    kwargs: dict[str, Any] = {}
    if model in FALLBACK_MODELS:
        kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    response = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        **kwargs,
    )
    usage = Usage(
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        extra={"request_id": getattr(response, "_request_id", None)},
    )
    if response.stop_reason in ("refusal", "max_tokens"):
        log.warning("structured call stopped: %s", response.stop_reason)
        return None, usage
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        return json.loads(text), usage
    except json.JSONDecodeError:
        log.warning("structured call returned invalid JSON")
        return None, usage


def trail_prompt(request: TrailRequest, context: list[RetrievedChunk]) -> str:
    """Ids and text only. Titles are given so the model can reason, URLs are not."""
    articles = "\n".join(f"[article {a.id}] {a.title.replace('_', ' ')}" for a in request.articles)
    passages = "\n\n".join(
        f"[chunk {c.chunk_id} | article {c.article_id}]\n{c.text}" for c in context
    )
    return (
        f"Topic cluster: {request.rabbit_hole}\n"
        f"Reader's interests: {', '.join(request.interests) or '(none given)'}\n\n"
        f"Candidate articles, gentle -> deep:\n{articles}\n\nPassages:\n{passages}"
    )


class AnthropicProvider:
    """Claude via the official SDK: JSON-schema output, validated again with Pydantic."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        if client is None:
            import anthropic

            key = settings.anthropic_api_key
            client = anthropic.Anthropic(
                api_key=key.get_secret_value() if key else None, timeout=30.0, max_retries=1
            )
        self.client = client
        self.name = settings.llm_model

    def generate_trail(
        self, request: TrailRequest, context: list[RetrievedChunk]
    ) -> tuple[TrailDraft, Usage]:
        data, usage = structured_call(
            self.client, self.name, TRAIL_SYSTEM, trail_prompt(request, context), TRAIL_SCHEMA
        )
        try:
            return TrailDraft.model_validate(data or {"steps": []}), usage
        except ValidationError:
            log.warning("trail draft failed validation")
            return TrailDraft(steps=[]), usage


def make_provider(settings: Settings) -> LLMProvider:
    """Anthropic if a key is configured, otherwise the template trail."""
    if settings.anthropic_api_key and settings.anthropic_api_key.get_secret_value():
        return AnthropicProvider(settings)
    return TemplateProvider()
