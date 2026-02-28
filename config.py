from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    higgsfield_api_key: str = ""
    google_cloud_tts_key: str = ""
    sync_so_api_key: str = ""
    youtube_client_id: str = ""
    youtube_client_secret: str = ""

    database_url: str = "sqlite+aiosqlite:///./frameshift.db"
    database_public_url: str = ""  # Railway Postgres public URL (fallback)
    host: str = "0.0.0.0"
    port: int = 8000

    knowledge_dir: str = "knowledge"
    output_dir: str = "output"

    model: str = "claude-sonnet-4-6"

    # Auth
    secret_key: str = "change-me-in-production"

    # Google OAuth
    google_oauth_client_id: str = ""
    google_oauth_client_secret: str = ""
    google_oauth_redirect_uri: str = "http://localhost:8000/auth/google/callback"

    # SMTP
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "noreply@frameshift.ai"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
