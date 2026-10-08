"""
Application Configuration Module for Stage 1 Baseline Multi-Agent System.

Uses pydantic-settings to manage environment variables cleanly.
"""

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """System settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Provider configuration
    llm_provider: str = Field(
        default="gemini",
        description="Active LLM provider identifier (e.g., 'gemini', 'mock')"
    )
    llm_api_key: str = Field(
        default="",
        description="API Key for the active provider. If empty, the system runs in unconfigured mode."
    )
    llm_model: str = Field(
        default="gemini-3.1-flash-lite",
        description="Model name to be used by the active provider"
    )

    # Knowledge and retrieval configuration
    embedding_provider: str = Field(default="gemini", description="Embedding provider identifier (gemini or mock)")
    embedding_api_key: str = Field(default="", description="Optional embedding API key; falls back to LLM_API_KEY for Gemini")
    embedding_model: str = Field(default="gemini-embedding-001", description="Embedding model name")
    knowledge_database_url: str = Field(default="", description="Optional PostgreSQL connection URL for the pgvector knowledge store")
    knowledge_embedding_dimensions: int = Field(default=768, ge=1, description="Vector dimensions for the configured knowledge repository")

    # Agent / Pipeline configuration
    evaluator_pass_threshold: int = Field(
        default=80,
        ge=0,
        le=100,
        description="Minimum score (0-100) required for the Evaluator to award PASS status"
    )

    # Server / Network configuration
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173",
        description="Comma-separated list of allowed CORS origins"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse the comma-separated CORS origins into a list of strings."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_api_key_configured(self) -> bool:
        """Check whether the necessary API key has been supplied."""
        if self.llm_provider.lower() == "mock":
            return True
        return bool(self.llm_api_key and self.llm_api_key.strip())


# Global settings singleton
settings = Settings()
