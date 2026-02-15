
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.models import AgentName, QueryIntent
from src.modules.orchestrator.planner import QueryPlanner


class TestQueryPlanner:

    def setup_method(self) -> None:
        mock_llm = MagicMock()
        self.mock_generate = AsyncMock()
        mock_llm.generate = self.mock_generate
        self.planner = QueryPlanner(mock_llm)

    @pytest.mark.asyncio
    async def test_analyze_query_uses_llm_intent_entities_and_agents(self) -> None:
        self.mock_generate.return_value = {
            "content": (
                '{"intent":"deep_analysis","entities":["FastAPI","APIRouter"],'
                '"target_agents":["graph_query","code_analyst"],"confidence":0.91,'
                '"reasoning":"Needs dependency + implementation analysis"}'
            )
        }

        result = await self.planner.analyze_query(
            "Explain how FastAPI and APIRouter interact."
        )

        assert result.intent == QueryIntent.DEEP_ANALYSIS
        assert result.entities == ["FastAPI", "APIRouter"]
        assert result.target_agents == [
            AgentName.GRAPH_QUERY,
            AgentName.CODE_ANALYST,
            AgentName.RETRIEVAL_MEMORY,
        ]
        assert result.confidence == 0.91
        assert "Needs dependency + implementation analysis" in result.reasoning

    @pytest.mark.asyncio
    async def test_analyze_query_falls_back_when_llm_returns_invalid_payload(self) -> None:
        self.mock_generate.return_value = {"content": "not-json"}

        result = await self.planner.analyze_query("Hello there")

        assert result.intent == QueryIntent.GENERAL
        assert result.entities == []
        assert result.target_agents == [AgentName.GRAPH_QUERY, AgentName.RETRIEVAL_MEMORY]
        assert result.confidence == 0.3

    @pytest.mark.asyncio
    async def test_analyze_query_parses_embedded_json(self) -> None:
        self.mock_generate.return_value = {
            "content": (
                "Plan result:\n"
                '{"intent":"code_explanation","entities":["add_middleware"],'
                '"target_agents":["code_analyst","retrieval_memory"],"confidence":0.82,'
                '"reasoning":"Implementation-focused request"}\n'
                "done"
            )
        }

        result = await self.planner.analyze_query("Show add_middleware implementation.")

        assert result.intent == QueryIntent.CODE_EXPLANATION
        assert result.entities == ["add_middleware"]
        assert result.target_agents == [
            AgentName.CODE_ANALYST,
            AgentName.RETRIEVAL_MEMORY,
            AgentName.GRAPH_QUERY,
        ]
        assert result.confidence == 0.82

    @pytest.mark.asyncio
    async def test_analyze_query_uses_policy_when_target_agents_invalid(self) -> None:
        self.mock_generate.return_value = {
            "content": (
                '{"intent":"relationship_query","entities":["Depends"],'
                '"target_agents":["not-an-agent"],"confidence":0.74,'
                '"reasoning":"Relationship lookup"}'
            )
        }

        result = await self.planner.analyze_query("What depends on Depends?")

        assert result.intent == QueryIntent.RELATIONSHIP_QUERY
        assert result.target_agents == [AgentName.GRAPH_QUERY, AgentName.RETRIEVAL_MEMORY]
