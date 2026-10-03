from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "financial-doc-fraud-detection"
    environment: str = "dev"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    backend_url: str = "http://localhost:8000"
    openai_api_key: str = ""
    llm_provider: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    storage_root: str = "data"
    max_upload_size_mb: float = 10.0
    pdf_scale: float = 2.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache

def get_settings() -> Settings:
    return Settings()


settings = get_settings()
