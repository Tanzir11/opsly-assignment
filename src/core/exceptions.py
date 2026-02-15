
from __future__ import annotations

from typing import Any


class OpslyError(Exception):

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


# ─── Agent Errors ───────────────────────────────────────────────────────────


class AgentError(OpslyError):

    def __init__(
        self,
        message: str,
        agent: str = "unknown",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.agent = agent
        super().__init__(message, details)


class AgentTimeoutError(AgentError):
    pass


class AgentConnectionError(AgentError):
    pass


class AgentToolError(AgentError):

    def __init__(
        self,
        message: str,
        agent: str = "unknown",
        tool: str = "unknown",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.tool = tool
        super().__init__(message, agent, details)


class AgentRoutingError(AgentError):
    pass


# ─── Database Errors ────────────────────────────────────────────────────────


class DatabaseError(OpslyError):
    pass


class Neo4jError(DatabaseError):
    pass


class MongoDBError(DatabaseError):
    pass


class MilvusError(DatabaseError):
    pass


# ─── Indexing Errors ────────────────────────────────────────────────────────


class IndexingError(OpslyError):
    pass


class RepositoryCloneError(IndexingError):
    pass


class ASTParsingError(IndexingError):
    pass


class EntityExtractionError(IndexingError):
    pass


# ─── LLM / Embedding Errors ────────────────────────────────────────────────


class LLMError(OpslyError):
    pass


class LLMProviderError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class EmbeddingError(OpslyError):
    pass


# ─── Validation Errors ──────────────────────────────────────────────────────


class ValidationError(OpslyError):
    pass


class QuerySafetyError(ValidationError):
    pass
