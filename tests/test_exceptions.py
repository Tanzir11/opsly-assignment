
from __future__ import annotations

import pytest

from src.core.exceptions import (
    AgentConnectionError,
    AgentError,
    AgentRoutingError,
    AgentTimeoutError,
    AgentToolError,
    ASTParsingError,
    DatabaseError,
    EmbeddingError,
    EntityExtractionError,
    IndexingError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    MilvusError,
    MongoDBError,
    Neo4jError,
    OpslyError,
    QuerySafetyError,
    RepositoryCloneError,
    ValidationError,
)


class TestExceptionHierarchy:

    def test_base_exception(self) -> None:
        exc = OpslyError("test error", {"key": "value"})
        assert exc.message == "test error"
        assert exc.details == {"key": "value"}
        assert str(exc) == "test error"

    def test_agent_error(self) -> None:
        exc = AgentError("agent failed", agent="indexer")
        assert exc.agent == "indexer"
        assert isinstance(exc, OpslyError)

    def test_agent_timeout(self) -> None:
        exc = AgentTimeoutError("timed out", agent="graph_query")
        assert isinstance(exc, AgentError)
        assert isinstance(exc, OpslyError)

    def test_agent_tool_error(self) -> None:
        exc = AgentToolError("tool failed", agent="indexer", tool="index_repository")
        assert exc.tool == "index_repository"
        assert exc.agent == "indexer"

    def test_database_errors(self) -> None:
        assert isinstance(Neo4jError("neo4j"), DatabaseError)
        assert isinstance(MongoDBError("mongo"), DatabaseError)
        assert isinstance(MilvusError("milvus"), DatabaseError)
        assert isinstance(DatabaseError("db"), OpslyError)

    def test_indexing_errors(self) -> None:
        assert isinstance(RepositoryCloneError("clone"), IndexingError)
        assert isinstance(ASTParsingError("parse"), IndexingError)
        assert isinstance(EntityExtractionError("extract"), IndexingError)
        assert isinstance(IndexingError("index"), OpslyError)

    def test_llm_errors(self) -> None:
        assert isinstance(LLMProviderError("provider"), LLMError)
        assert isinstance(LLMRateLimitError("rate"), LLMError)
        assert isinstance(LLMError("llm"), OpslyError)

    def test_validation_errors(self) -> None:
        assert isinstance(QuerySafetyError("unsafe"), ValidationError)
        assert isinstance(ValidationError("invalid"), OpslyError)

    def test_embedding_error(self) -> None:
        exc = EmbeddingError("embed failed")
        assert isinstance(exc, OpslyError)
