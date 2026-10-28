"""Runtime settings, read from environment variables (and `.env` if present)."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    database_url: str = "postgresql+psycopg://rabbit:rabbit@localhost:5432/rabbithole"
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-haiku-5-5"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    dataset_mode: Literal["dev", "full"] = "dev"
    clickstream_months: str = ""
    mlflow_tracking_uri: str = "http://localhost:5000"
    wiki_user_agent: str = "RabbitHoleFinder/0.1 (student project; contact: <email>)"

    @property
    def months(self) -> list[str]:
        """Clickstream months as a list; empty means "discover the latest"."""
        return [m.strip() for m in self.clickstream_months.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
