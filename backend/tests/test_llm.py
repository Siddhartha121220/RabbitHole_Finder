"""AnthropicProvider against a fake SDK client: request shape, parsing, refusals. No network."""

import json
from types import SimpleNamespace
from typing import Any

from pydantic import SecretStr

from rabbithole.config import get_settings
from rabbithole.llm.provider import (
    AnthropicProvider,
    CandidateArticle,
    RetrievedChunk,
    TemplateProvider,
    TrailRequest,
    make_provider,
)


class FakeMessages:
    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.text, self.stop_reason = text, stop_reason
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="text", text=self.text)],
            usage=SimpleNamespace(input_tokens=1200, output_tokens=300),
        )


def fake_client(text: str, stop_reason: str = "end_turn") -> Any:
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(text, stop_reason)))


REQUEST = TrailRequest(
    "Chess", ["board games"], [CandidateArticle(1, "Rook_(chess)", "The rook is a piece.")]
)
CONTEXT = [RetrievedChunk(10, 1, "Rook_(chess)", "The rook moves in straight lines.")]


def provider(model: str, client: Any) -> AnthropicProvider:
    return AnthropicProvider(get_settings().model_copy(update={"llm_model": model}), client)


def test_parses_structured_trail_and_reports_tokens() -> None:
    payload = {"intro": "Start here.", "steps": [{"article_id": 1, "chunk_id": 10, "why": "W"}]}
    client = fake_client(json.dumps(payload))
    draft, usage = provider("claude-haiku-5-5", client).generate_trail(REQUEST, CONTEXT)
    assert draft.steps[0].article_id == 1 and draft.steps[0].chunk_id == 10
    assert (usage.input_tokens, usage.output_tokens) == (1200, 300)
    call = client.beta.messages.calls[0]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert "fallbacks" not in call  # Haiku has no server-side refusal fallback
    assert "[chunk 10 | article 1]" in call["messages"][0]["content"]
    assert "http" not in call["messages"][0]["content"]  # the model never sees URLs


def test_fallbacks_requested_for_opus() -> None:
    client = fake_client(json.dumps({"intro": "", "steps": []}))
    provider("claude-opus-5-5", client).generate_trail(REQUEST, CONTEXT)
    call = client.beta.messages.calls[0]
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]


def test_refusal_or_garbage_gives_empty_draft() -> None:
    for client in (fake_client("", "refusal"), fake_client("not json")):
        draft, _ = provider("claude-haiku-5-5", client).generate_trail(REQUEST, CONTEXT)
        assert draft.steps == []


def test_no_key_means_template_provider() -> None:
    settings = get_settings().model_copy(update={"anthropic_api_key": None})
    assert isinstance(make_provider(settings), TemplateProvider)


def test_cost_guard_and_kappa(tmp_path: Any) -> None:
    import pytest

    from rabbithole.eval import trail_judge

    settings = get_settings().model_copy(update={"anthropic_api_key": SecretStr("sk-test")})
    with pytest.raises(SystemExit, match="cost guard"):
        trail_judge.run(settings, 201, confirmed=False)
    no_key = settings.model_copy(update={"anthropic_api_key": None})
    with pytest.raises(SystemExit, match="not set"):
        trail_judge.run(no_key, 10, confirmed=False)
    assert trail_judge.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert trail_judge.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    sheet = tmp_path / "check.csv"
    sheet.write_text(
        "judge_verdict,human_verdict\nsupported,supported\npartial,unsupported\nsupported,\n"
    )
    assert "agreement 50.0%" in trail_judge.agreement(sheet)
