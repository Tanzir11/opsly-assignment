
from __future__ import annotations

import pytest

from src.core.models import (
    AgentName,
    AgentResponse,
    ChatRequest,
    ChatResponse,
    CodeEntity,
    CodeRelationship,
    ConversationMessage,
    EntityType,
    GraphStatistics,
    IndexJobStatus,
    IndexRequest,
    IndexStatusResponse,
    QueryAnalysis,
    QueryIntent,
    RoutingDecision,
    SourceReference,
    SystemHealth,
)


class TestChatModels:

    def test_chat_request_required_message(self) -> None:
        req = ChatRequest(message="Hello")
        assert req.message == "Hello"
        assert req.session_id is None
        assert req.stream is False

    def test_chat_request_with_session(self) -> None:
        req = ChatRequest(session_id="sess123", message="Hello", stream=True)
        assert req.session_id == "sess123"
        assert req.stream is True

    def test_chat_response_auto_id(self) -> None:
        resp = ChatResponse(session_id="sess123", response="Hi")
        assert resp.message_id is not None
        assert len(resp.message_id) > 0

    def test_source_reference(self) -> None:
        ref = SourceReference(
            file_path="fastapi/routing.py",
            entity_name="APIRouter",
            start_line=10,
            end_line=50,
        )
        assert ref.file_path == "fastapi/routing.py"


class TestIndexModels:

    def test_index_request_defaults(self) -> None:
        req = IndexRequest()
        assert req.repo_url is None
        assert req.incremental is False

    def test_index_status_response(self) -> None:
        resp = IndexStatusResponse(
            job_id="abc123",
            status=IndexJobStatus.RUNNING,
            repo="fastapi",
            progress=0.5,
            files_processed=50,
            files_total=100,
        )
        assert resp.progress == 0.5
        assert resp.status == IndexJobStatus.RUNNING


class TestCodeEntityModels:

    def test_code_entity_auto_id(self) -> None:
        entity = CodeEntity(
            entity_type=EntityType.CLASS,
            name="FastAPI",
            file_path="fastapi/applications.py",
        )
        assert entity.entity_id is not None
        assert len(entity.entity_id) > 0

    def test_code_relationship(self) -> None:
        rel = CodeRelationship(
            source_id="abc",
            target_id="def",
            relationship_type="INHERITS_FROM",
        )
        assert rel.relationship_type == "INHERITS_FROM"


class TestQueryAnalysis:

    def test_query_analysis(self) -> None:
        analysis = QueryAnalysis(
            intent=QueryIntent.FACTUAL_LOOKUP,
            entities=["FastAPI"],
            target_agents=[AgentName.GRAPH_QUERY],
            confidence=0.9,
        )
        assert analysis.intent == QueryIntent.FACTUAL_LOOKUP
        assert analysis.confidence == 0.9

    def test_agent_response(self) -> None:
        resp = AgentResponse(
            agent=AgentName.GRAPH_QUERY,
            tool_name="find_entity",
            success=True,
            result={"entities": []},
        )
        assert resp.success is True
        assert resp.agent == AgentName.GRAPH_QUERY


class TestEnums:

    def test_query_intent_values(self) -> None:
        assert QueryIntent.FACTUAL_LOOKUP.value == "factual_lookup"
        assert QueryIntent.DEEP_ANALYSIS.value == "deep_analysis"

    def test_agent_name_values(self) -> None:
        assert AgentName.ORCHESTRATOR.value == "orchestrator"
        assert AgentName.RETRIEVAL_MEMORY.value == "retrieval_memory"

    def test_entity_type_values(self) -> None:
        assert EntityType.CLASS.value == "Class"
        assert EntityType.FUNCTION.value == "Function"
        assert EntityType.METHOD.value == "Method"

    def test_index_job_status(self) -> None:
        assert IndexJobStatus.PENDING.value == "pending"
        assert IndexJobStatus.COMPLETED.value == "completed"
