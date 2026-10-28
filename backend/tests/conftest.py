from pathlib import Path

import pytest

from rabbithole.ingest.clickstream import parse

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def tiny_clickstream(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("cs") / "clickstream.parquet"
    parse(FIXTURES / "clickstream_tiny.tsv", out)
    return out
