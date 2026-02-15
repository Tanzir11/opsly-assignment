
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ─── Enums ──────────────────────────────────────────────────────────────────


class QueryIntent(str, Enum):

    FACTUAL_LOOKUP = "factual_lookup"
    RELATIONSHIP_QUERY = "relationship_query"
    DEEP_ANALYSIS = "deep_analysis"
    CODE_EXPLANATION = "code_explanation"
    PATTERN_SEARCH = "pattern_search"
    INDEXING_OPERATION = "indexing_operation"
    GENERAL = "general"


class AgentName(str, Enum):

    ORCHESTRATOR = "orchestrator"
    INDEXER = "indexer"
    GRAPH_QUERY = "graph_query"
    CODE_ANALYST = "code_analyst"
    RETRIEVAL_MEMORY = "retrieval_memory"


class IndexJobStatus(str, Enum):

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EntityType(str, Enum):

    MODULE = "Module"
    CLASS = "Class"
    FUNCTION = "Function"
    METHOD = "Method"
    PARAMETER = "Parameter"
    DECORATOR = "Decorator"
    IMPORT = "Import"
    DOCSTRING = "Docstring"
    FILE = "File"


# ─── Chat Models ────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):

    session_id: str | None = Field(default=None, description="Session ID; generated if absent")
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    stream: bool = Field(default=False, description="Enable SSE streaming")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class ChatResponse(BaseModel):

    session_id: str
    message_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    response: str
    agents_used: list[str] = Field(default_factory=list)
    sources: list[SourceReference] | None = None
    correlation_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SourceReference(BaseModel):

    file_path: str
    entity_name: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    snippet: str | None = None


# Fix forward reference
ChatResponse.model_rebuild()


# ─── Index Models ───────────────────────────────────────────────────────────


class IndexRequest(BaseModel):

    repo_url: str | None = Field(default=None, description="Repository URL; uses default if absent")
    incremental: bool = Field(default=False, description="Incremental indexing by git diff")


class IndexStatusResponse(BaseModel):

    job_id: str
    status: IndexJobStatus
    repo: str
    progress: float = 0.0
    files_processed: int = 0
    files_total: int = 0
    entities_extracted: int = 0
    errors: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None


# ─── Agent Communication Models ─────────────────────────────────────────────


class AgentRequest(BaseModel):

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    session_id: str = ""
    correlation_id: str = ""


class AgentResponse(BaseModel):

    agent: AgentName
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0


class QueryAnalysis(BaseModel):

    intent: QueryIntent
    entities: list[str] = Field(default_factory=list)
    target_agents: list[AgentName] = Field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0


# ─── Graph Models ───────────────────────────────────────────────────────────


class CodeEntity(BaseModel):

    entity_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    entity_type: EntityType
    name: str
    file_path: str = ""
    signature: str = ""
    start_line: int | None = None
    end_line: int | None = None
    doc: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class CodeRelationship(BaseModel):

    source_id: str
    target_id: str
    relationship_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphStatistics(BaseModel):

    total_nodes: int = 0
    total_relationships: int = 0
    nodes_by_type: dict[str, int] = Field(default_factory=dict)
    relationships_by_type: dict[str, int] = Field(default_factory=dict)
    indexed_files: int = 0
    indexed_repos: list[str] = Field(default_factory=list)


# ─── Health Models ──────────────────────────────────────────────────────────


class AgentHealth(BaseModel):

    agent: str
    healthy: bool
    url: str
    latency_ms: float | None = None
    error: str | None = None


class SystemHealth(BaseModel):

    healthy: bool
    agents: list[AgentHealth] = Field(default_factory=list)
    databases: dict[str, bool] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ─── Conversation Memory Models ─────────────────────────────────────────────


class ConversationMessage(BaseModel):

    session_id: str
    message_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingDecision(BaseModel):

    session_id: str
    query: str
    selected_agents: list[str]
    reasoning_summary: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
