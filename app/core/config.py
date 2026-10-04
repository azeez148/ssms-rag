from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "rag-platform"
    app_env: str = "development"
    log_level: str = "INFO"

    rag_database_url: str = "postgresql+psycopg://rag:rag@localhost:5433/rag"

    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384

    # 'auto' = Claude when ANTHROPIC_API_KEY is set, else the offline extractive
    # provider. Others: 'anthropic', 'openai', 'gemini', 'qwen', 'ollama',
    # 'openai_compatible', 'extractive', 'stub' (see app/generation/factory.py).
    llm_provider: str = "auto"
    # Empty = the selected provider's default model.
    llm_model: str = ""
    # Overrides the provider's endpoint; required for 'openai_compatible'.
    llm_base_url: str = ""
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.2
    llm_timeout_seconds: float = 120.0

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    qwen_api_key: SecretStr | None = None
    # For 'openai_compatible' endpoints that need a key (Groq, OpenRouter, ...).
    llm_api_key: SecretStr | None = None

    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 4096

    reranker_provider: str = "cross_encoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidate_multiplier: int = 4

    max_history_turns: int = 3

    guardrails_similarity_threshold: float = 0.35
    max_context_tokens: int = 1500
    agentic_max_retries: int = 2

    # Comma-separated. Defaults to the dev Angular console; production sets
    # this to the real frontend origin(s) via an env var, never by editing code.
    cors_allowed_origins: str = "http://localhost:4200"

    # The only directory /admin/ingest may load source configs from. Defaults to
    # the project's config/ folder (next to app/), independent of the process's
    # working directory.
    source_config_dir: str = str(Path(__file__).resolve().parents[2] / "config")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
