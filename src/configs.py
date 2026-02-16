
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EnvFlag = Literal["development", "staging", "production"]


def _resolve_env_override_files(app_env: str) -> list[Path]:
    if app_env == "development":
        return [
            PROJECT_ROOT / ".env.development",
            PROJECT_ROOT / ".env.develop",
        ]
    if app_env == "staging":
        return [PROJECT_ROOT / ".env.staging"]
    return []


def _load_env_files() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    app_env = os.environ.get("APP_ENV", "development")
    for env_file in _resolve_env_override_files(app_env):
        load_dotenv(env_file)


def _get_env_files() -> list[Path]:
    app_env = os.environ.get("APP_ENV", "development")
    if app_env == "production":
        return [PROJECT_ROOT / ".env"]
    return [PROJECT_ROOT / ".env", *_resolve_env_override_files(app_env)]


# Pre-load env so APP_ENV and overrides are available
_load_env_files()


class APIConfig(BaseSettings):

    # OpenAI keys
    openai_api_key_1: str | None = Field(default=None, alias="OPENAI_API_KEY_1")
    openai_api_key_2: str | None = Field(default=None, alias="OPENAI_API_KEY_2")
    openai_api_key_3: str | None = Field(default=None, alias="OPENAI_API_KEY_3")
    openai_api_key_4: str | None = Field(default=None, alias="OPENAI_API_KEY_4")
    openai_api_key_5: str | None = Field(default=None, alias="OPENAI_API_KEY_5")

    # Gemini (Google) keys
    gemini_api_key_1: str | None = Field(default=None, alias="GEMINI_API_KEY_1")
    gemini_api_key_2: str | None = Field(default=None, alias="GEMINI_API_KEY_2")
    gemini_api_key_3: str | None = Field(default=None, alias="GEMINI_API_KEY_3")
    gemini_api_key_4: str | None = Field(default=None, alias="GEMINI_API_KEY_4")
    gemini_api_key_5: str | None = Field(default=None, alias="GEMINI_API_KEY_5")

    # Anthropic keys
    anthropic_api_key_1: str | None = Field(default=None, alias="ANTHROPIC_API_KEY_1")
    anthropic_api_key_2: str | None = Field(default=None, alias="ANTHROPIC_API_KEY_2")
    anthropic_api_key_3: str | None = Field(default=None, alias="ANTHROPIC_API_KEY_3")
    anthropic_api_key_4: str | None = Field(default=None, alias="ANTHROPIC_API_KEY_4")
    anthropic_api_key_5: str | None = Field(default=None, alias="ANTHROPIC_API_KEY_5")

    @property
    def openai_api_keys(self) -> list[str]:
        return self._collect_keys("openai_api_key")

    @property
    def gemini_api_keys(self) -> list[str]:
        return self._collect_keys("gemini_api_key")

    @property
    def anthropic_api_keys(self) -> list[str]:
        return self._collect_keys("anthropic_api_key")

    def _collect_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        for i in range(1, 6):
            key = getattr(self, f"{prefix}_{i}", None)
            if key and key.strip():
                keys.append(key.strip())
        return keys

    model_config = SettingsConfigDict(
        env_file=_get_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class AppConfig(BaseSettings):

    # Flag: development | staging | production
    app_env: EnvFlag = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    api_port: int = Field(default=8000, alias="API_PORT")

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="", alias="NEO4J_PASSWORD")

    # MongoDB
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017", alias="MONGODB_URI"
    )
    mongodb_db_name: str = Field(default="opsly", alias="MONGODB_DB_NAME")

    # Milvus
    enable_milvus: bool = Field(default=False, alias="ENABLE_MILVUS")
    milvus_host: str = Field(default="localhost", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    milvus_collection_prefix: str = Field(
        default="opsly", alias="MILVUS_COLLECTION_PREFIX"
    )

    # MCP Agent URLs
    orchestrator_agent_url: str = Field(
        default="http://localhost:8001", alias="ORCHESTRATOR_AGENT_URL"
    )
    indexer_agent_url: str = Field(
        default="http://localhost:8002", alias="INDEXER_AGENT_URL"
    )
    graph_query_agent_url: str = Field(
        default="http://localhost:8003", alias="GRAPH_QUERY_AGENT_URL"
    )
    code_analyst_agent_url: str = Field(
        default="http://localhost:8004", alias="CODE_ANALYST_AGENT_URL"
    )
    retrieval_memory_agent_url: str = Field(
        default="http://localhost:8005", alias="RETRIEVAL_MEMORY_AGENT_URL"
    )

    # FastAPI repo
    fastapi_repo_url: str = Field(
        default="https://github.com/tiangolo/fastapi.git",
        alias="FASTAPI_REPO_URL",
    )
    repo_workspace: str = Field(
        default="/tmp/opsly/repos", alias="REPO_WORKSPACE"
    )

    # LLM routing
    llm_default_provider: str = Field(
        default="openai", alias="LLM_DEFAULT_PROVIDER"
    )
    llm_fallback_order: str = Field(
        default="openai,anthropic,gemini", alias="LLM_FALLBACK_ORDER"
    )

    # Embedding
    embedding_default_provider: str = Field(
        default="local", alias="EMBEDDING_DEFAULT_PROVIDER"
    )
    embedding_model_code: str = Field(
        default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL_CODE"
    )
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")

    # Agent timeouts and retries
    agent_timeout_seconds: float = Field(
        default=30.0, alias="AGENT_TIMEOUT_SECONDS"
    )
    agent_max_retries: int = Field(default=2, alias="AGENT_MAX_RETRIES")

    # API auth
    auth_user_id: str = Field(default="tanzir", alias="AUTH_USER_ID")
    auth_password: str = Field(default="khan", alias="AUTH_PASSWORD")
    auth_jwt_secret: str = Field(
        default="opsly-dev-secret-change-me",
        alias="AUTH_JWT_SECRET",
    )
    auth_jwt_exp_minutes: int = Field(default=60, alias="AUTH_JWT_EXP_MINUTES")

    model_config = SettingsConfigDict(
        env_file=_get_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_app_config() -> AppConfig:
    return AppConfig()


@lru_cache
def get_api_config() -> APIConfig:
    return APIConfig()
