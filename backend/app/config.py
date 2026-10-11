"""
Application Configuration Module for Stage 1 Baseline Multi-Agent System.

Uses pydantic-settings to manage environment variables cleanly.
"""

from pathlib import Path
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """System settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env", str(_BASE_DIR / ".env")),
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
    knowledge_chunk_size: int = Field(default=1200, ge=100, description="Target character chunk size for document chunking")
    knowledge_chunk_overlap: int = Field(default=120, ge=0, description="Overlap in characters between consecutive chunks")
    embedding_batch_size: int = Field(default=64, ge=1, le=100, description="Batch size for embedding API calls")
    embedding_max_concurrency: int = Field(default=5, ge=1, le=20, description="Max concurrent batches for embedding generation")
    retrieval_top_k: int = Field(default=10, ge=1, le=50, description="Default number of top chunks returned by hybrid retrieval")
    retrieval_budget_per_run: int = Field(default=30, ge=1, description="Maximum total retrieved chunks budget per execution run")
    retrieval_rrf_k: int = Field(default=60, ge=1, description="Reciprocal Rank Fusion smoothing parameter k")

    # Agent / Pipeline configuration
    evaluator_pass_threshold: int = Field(
        default=80,
        ge=0,
        le=100,
        description="Minimum score (0-100) required for the Evaluator to award PASS status"
    )
    max_replanning_rounds: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum number of targeted replanning rounds if evaluation or verification fails"
    )

    # Phase 7: Strategy Resource Budgets
    budget_direct_max_llm_calls: int = Field(default=1, ge=1, description="Max LLM calls for DIRECT strategy")
    budget_direct_max_tokens: int = Field(default=3000, ge=1, description="Max tokens for DIRECT strategy")
    budget_direct_max_wall_time_seconds: float = Field(default=15.0, ge=0.001, description="Max wall time in seconds for DIRECT strategy")

    budget_single_agent_max_llm_calls: int = Field(default=3, ge=1, description="Max LLM calls for SINGLE_AGENT strategy")
    budget_single_agent_max_tokens: int = Field(default=35000, ge=1, description="Max tokens for SINGLE_AGENT strategy")
    budget_single_agent_max_wall_time_seconds: float = Field(default=45.0, ge=0.001, description="Max wall time in seconds for SINGLE_AGENT strategy")

    budget_single_agent_rag_max_llm_calls: int = Field(default=4, ge=1, description="Max LLM calls for SINGLE_AGENT_RAG strategy")
    budget_single_agent_rag_max_tokens: int = Field(default=18000, ge=1, description="Max tokens for SINGLE_AGENT_RAG strategy")
    budget_single_agent_rag_max_wall_time_seconds: float = Field(default=60.0, ge=0.001, description="Max wall time in seconds for SINGLE_AGENT_RAG strategy")

    budget_multi_agent_max_llm_calls: int = Field(default=35, ge=1, description="Max LLM calls for MULTI_AGENT strategy")
    budget_multi_agent_max_tokens: int = Field(default=120000, ge=1000, description="Max tokens for MULTI_AGENT strategy")
    budget_multi_agent_max_wall_time_seconds: float = Field(default=240.0, ge=1.0, description="Max wall time in seconds for MULTI_AGENT strategy")
    enable_multi_domain_rag_promotion: bool = Field(default=False, description="Whether to promote multi-article or multi-domain document RAG queries to MULTI_AGENT")

    # Reliability and Persistence configuration
    max_concurrent_runs: int = Field(default=5, ge=1, le=50, description="Maximum concurrent pipeline execution runs allowed")
    run_timeout_seconds: float = Field(default=180.0, ge=1.0, description="Global execution timeout in seconds per pipeline run")
    database_url: str = Field(default="", description="Optional PostgreSQL connection URL for runs and metrics persistence")

    # Web search and page fetch configuration
    tavily_api_key: str = Field(default="", description="API key for Tavily search provider")
    web_search_provider: str = Field(default="tavily", description="Web search provider (tavily, mock)")
    web_max_results_default: int = Field(default=5, ge=1, le=20, description="Default max search results")
    fetch_page_max_bytes: int = Field(default=2 * 1024 * 1024, ge=1024, description="Max allowed bytes per fetched page (2MB)")
    fetch_page_timeout_seconds: float = Field(default=10.0, ge=1.0, description="Timeout in seconds for page fetching")
    fetch_page_max_redirects: int = Field(default=5, ge=0, le=10, description="Max allowed redirects")
    fetch_page_user_agent: str = Field(default="AdaptiveAgentBot/1.0 (+http://localhost:8000; web-research)", description="User-Agent header for fetching")
    fetch_page_domain_allowlist: str = Field(default="", description="Comma-separated allowed domains, or empty for all public domains")
    fetch_page_domain_denylist: str = Field(default="", description="Comma-separated disallowed domains")
    fetch_page_respect_robots_txt: bool = Field(default=True, description="Whether to check and enforce robots.txt")
    web_cache_mode: str = Field(default="live", description="Caching mode: 'live', 'record', or 'replay'")
    web_cache_dir: str = Field(default="data/web_cache", description="Directory path for cached search and page fixtures")
    web_cache_ttl_seconds: float = Field(default=86400.0, ge=0.0, description="Cache entry time-to-live in seconds (0 = never expire)")

    # Server / Network configuration
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173",
        description="Comma-separated list of allowed CORS origins"
    )

    @property
    def fetch_page_domain_allowlist_set(self) -> set[str]:
        return {d.strip().lower() for d in self.fetch_page_domain_allowlist.split(",") if d.strip()}

    @property
    def fetch_page_domain_denylist_set(self) -> set[str]:
        return {d.strip().lower() for d in self.fetch_page_domain_denylist.split(",") if d.strip()}

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
