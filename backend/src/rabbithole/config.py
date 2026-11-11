"""Runtime settings, read from environment variables (and `.env` if present)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    database_url: str = "postgresql+psycopg://rabbit:rabbit@localhost:5432/rabbithole"
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-haiku-5-5"
    judge_model: str = "claude-opus-5-5"  # LLM-as-judge for trail faithfulness
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384  # must match the model; checked when embedding
    # bge models expect this instruction on short queries (not on documents)
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embed_batch_size: int = 64
    embed_shard_rows: int = 2048
    # Empty = CPU. ["cuda:0"] = one GPU (fp16). ["cuda:0", "cuda:1"] = one process per GPU.
    embed_devices: list[str] = []
    max_chunks_per_article: int = 8  # caps FULL storage; retrieval only needs the best few
    dataset_mode: Literal["dev", "full"] = "dev"
    clickstream_months: str = ""
    mlflow_tracking_uri: str = "http://localhost:5000"
    wiki_user_agent: str = "RabbitHoleFinder/0.1 (student project; https://github.com/Siddhartha121220/RabbitHole_Finder)"
    data_dir: Path = REPO_ROOT / "data"
    dev_target_articles: int = 5000
    full_min_views: int = 1000
    max_download_bytes: int = 2 * 1024**3
    leiden_resolutions: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0]
    leiden_resolution: float | None = None  # set to override the sweep's choice
    prune_top_k: int = 10
    hub_penalty: float = 0.1
    # Rabbit-holeness weights (DATA_AND_ML §3.3): equal, expected depth counted double.
    rh_weights: dict[str, float] = {
        "expected_depth": 2.0,
        "stickiness": 1.0,
        "onward_ratio": 1.0,
        "breadth": 1.0,
        "content_richness": 1.0,
        "discovery_share": 1.0,
    }

    # Recommender (DATA_AND_ML §4) and offline evaluation (§6).
    seeds_per_interest: int = 3
    seed_min_similarity: float = 0.6
    shortlist_size: int = 200
    max_hops: int = 6
    bridge_max_hops: int = 4
    goldilocks_sigma: float = 0.12
    mmr_lambda: float = 0.7
    eval_queries: int = 500
    # API (ARCHITECTURE §5).
    frontend_origin: str = "http://localhost:3000"
    trail_rate_per_minute: int = 10
    eval_bootstrap: int = 1000

    @property
    def months(self) -> list[str]:
        """Clickstream months as a list; empty means "discover the latest"."""
        return [m.strip() for m in self.clickstream_months.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
