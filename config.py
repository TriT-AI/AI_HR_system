# config.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Tuple

from dotenv import load_dotenv, find_dotenv
from pydantic import Field,SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# 1) Load .env into process env if present (searches up the directory tree)
#    This is helpful for tools/IDEs; environment variables still override .env.
#    Use override=False so real env vars are not replaced by .env values.
load_dotenv(find_dotenv(), override=False)


def _env_chain() -> Tuple[str, ...]:
    """
    Return an ordered tuple of dotenv files to load via Pydantic.
    .env.<ENV> can override .env if ENV is set (e.g., development, staging, production).
    """
    env = os.getenv("ENV")
    return (".env", f".env.{env}") if env else (".env",)


class Settings(BaseSettings):
    # Azure Document Intelligence
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT: str = Field(..., description="Doc Intelligence endpoint")
    AZURE_DOCUMENT_INTELLIGENCE_KEY: str = Field(..., description="Doc Intelligence API key")

    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str = Field(..., description="Azure OpenAI endpoint")
    AZURE_OPENAI_API_KEY: SecretStr = Field(..., description="Azure OpenAI API key")
    AZURE_OPENAI_DEPLOYMENT: str = Field(..., description="Deployment name (e.g., gpt-4.1-mini)")
    AZURE_OPENAI_API_VERSION: str = Field("2025-04-01-preview", description="AOAI API version")

    # Database
    DUCKDB_PATH: str = Field("hr_resume_system.db", description="DuckDB file path")

    # Misc
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    ENV: Optional[str] = Field(default=None, description="Current environment name")

    # Load order: environment variables take precedence over dotenv files
    # You can pass multiple env files; later entries override earlier ones.
    model_config = SettingsConfigDict(
        env_file=_env_chain(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


