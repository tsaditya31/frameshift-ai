from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    higgsfield_api_key: str = ""
    google_cloud_tts_key: str = ""
    sync_so_api_key: str = ""
    youtube_client_id: str = ""
    youtube_client_secret: str = ""

    database_url: str = "sqlite+aiosqlite:///./clawd.db"
    host: str = "0.0.0.0"
    port: int = 8000

    knowledge_dir: str = "knowledge"
    output_dir: str = "output"

    model: str = "claude-sonnet-4-6"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
