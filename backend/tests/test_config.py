import pytest

from rabbithole.config import Settings


def test_months_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLICKSTREAM_MONTHS", "2026-07, 2026-08,")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    s = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    assert s.months == ["2026-07", "2026-08"]
    assert s.dataset_mode == "dev"


def test_bad_dataset_mode_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATASET_MODE", "huge")
    with pytest.raises(ValueError):
        Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
