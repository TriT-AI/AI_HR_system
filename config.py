# config.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Tuple

from dotenv import load_dotenv, find_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(find_dotenv(), override=False)

def _env_chain() -> Tuple[str, ...]:
    env = os.getenv("ENV")
    return (".env", f".env.{env}") if env else (".env",)

class Settings(BaseSettings):
    # Azure Document Intelligence
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT: str = Field(default=..., description="Doc Intelligence endpoint")
    AZURE_DOCUMENT_INTELLIGENCE_KEY: str = Field(default=..., description="Doc Intelligence API key")

    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str = Field(default=..., description="Azure OpenAI endpoint")
    AZURE_OPENAI_API_KEY: SecretStr = Field(default=..., description="Azure OpenAI API key")
    AZURE_OPENAI_DEPLOYMENT: str = Field(default=..., description="Deployment name (e.g., gpt-4.1-mini)")
    AZURE_OPENAI_API_VERSION: str = Field(default="2025-04-01-preview", description="AOAI API version")

    # Google Drive Configuration
    GOOGLE_DRIVE_API_KEY: str = Field(default="", description="Google Drive API key")
    GOOGLE_DRIVE_FOLDER_ID: str = Field(default="", description="Google Drive folder ID to monitor")

    # Database
    DUCKDB_PATH: str = Field(default="hr_resume_system.db", description="DuckDB file path")

    # Misc
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    ENV: Optional[str] = Field(default=None, description="Current environment name")

    model_config = SettingsConfigDict(
        env_file=_env_chain(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()
