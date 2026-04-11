"""
Application configuration.

All settings are loaded from environment variables (or .env file) and
validated at startup via pydantic-settings.  A missing required var will
crash the process immediately with a clear error — no silent defaults
for secrets.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Extra env vars won't cause a validation error.
        extra="ignore",
    )

    # ── Supabase ──────────────────────────────────────────────
    supabase_url: str = Field(description="Supabase project URL")
    supabase_key: str = Field(description="Supabase anon or service-role key")

    # ── Gemini AI ─────────────────────────────────────────────
    gemini_api_key: str = Field(description="Google Gemini API key")
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model ID (e.g. gemini-2.0-flash, gemini-2.5-pro)",
    )
    gemini_summary_style: str = Field(
        default="detailed",
        description="Default summary style: short, detailed, or reel",
    )
    gemini_max_retries: int = Field(default=3, ge=1, le=10)
    gemini_base_delay: float = Field(
        default=1.0,
        description="Base delay in seconds for exponential backoff",
    )

    # ── App ───────────────────────────────────────────────────
    app_name: str = "Today in History"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── CORS ──────────────────────────────────────────────────
    # Comma-separated origins allowed to call the API.
    # "*" for development; restrict in production.
    cors_origins: str = "*"

    # ── Scheduler ─────────────────────────────────────────────
    # Hour (UTC) at which the daily fetch job runs.
    scheduler_hour_utc: int = Field(default=0, ge=0, le=23)
    scheduler_minute_utc: int = Field(default=0, ge=0, le=59)
    scheduler_enabled: bool = Field(
        default=True,
        description="Set to false to disable the daily scheduler (useful in dev/test)",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton for the process)."""
    return Settings()
