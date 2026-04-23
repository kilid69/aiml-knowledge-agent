from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file.

    Field names are case-insensitive and map directly to env vars:
      QDRANT_HOST -> qdrant_host
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # qdrant section
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "aiml_knowledge"

    # lmstudio section
    lm_studio_url: str = "http://localhost:1234/v1"
    lm_studio_chat_model: str | None = None
    lm_studio_embed_model: str | None = None

# Instantiating Settings() reads env vars + .env once. Import elsewhere.
settings = Settings()
