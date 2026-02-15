
from __future__ import annotations

import json
from typing import Any

from src.core.llm_router import LLMRouter
from src.core.logging import get_logger
from src.core.models import AgentName, QueryAnalysis, QueryIntent

logger = get_logger(__name__)


# ─── Agent routing policy fallback ───────────────────────────────────────────

ROUTING_POLICY: dict[QueryIntent, list[AgentName]] = {
    QueryIntent.FACTUAL_LOOKUP: [AgentName.GRAPH_QUERY, AgentName.RETRIEVAL_MEMORY],
    QueryIntent.RELATIONSHIP_QUERY: [AgentName.GRAPH_QUERY, AgentName.RETRIEVAL_MEMORY],
    QueryIntent.DEEP_ANALYSIS: [
        AgentName.GRAPH_QUERY,
        AgentName.CODE_ANALYST,
        AgentName.RETRIEVAL_MEMORY,
    ],
    QueryIntent.CODE_EXPLANATION: [
        AgentName.GRAPH_QUERY,
        AgentName.CODE_ANALYST,
        AgentName.RETRIEVAL_MEMORY,
    ],
    QueryIntent.PATTERN_SEARCH: [AgentName.CODE_ANALYST, AgentName.GRAPH_QUERY],
    QueryIntent.INDEXING_OPERATION: [AgentName.INDEXER],
    QueryIntent.GENERAL: [AgentName.GRAPH_QUERY, AgentName.RETRIEVAL_MEMORY],
}


class QueryPlanner:

    def __init__(self, llm_router: LLMRouter) -> None:
        self.llm = llm_router

    async def analyze_query(
        self,
        message: str,
        session_id: str = "",
        conversation_context: list[dict[str, Any]] | None = None,
    ) -> QueryAnalysis:
        llm_analysis: dict[str, Any] | None = None
        try:
            llm_analysis = await self._llm_plan(message, conversation_context)
        except Exception as exc:
            logger.warning("LLM planning failed, using default fallback: %s", exc)

        intent = self._resolve_intent(llm_analysis.get("intent") if llm_analysis else None)
        entities = self._normalize_entities(llm_analysis.get("entities") if llm_analysis else None)
        target_agents = self._resolve_target_agents(llm_analysis, intent)
        confidence = self._normalize_confidence(
            llm_analysis.get("confidence") if llm_analysis else None
        )

        llm_reasoning = ""
        if llm_analysis:
            llm_reasoning = str(llm_analysis.get("reasoning", "")).strip()
        if not llm_reasoning:
            llm_reasoning = "LLM planner fallback applied."

        reasoning = (
            f"{llm_reasoning} | "
            f"Intent: {intent.value} | "
            f"Entities: {', '.join(entities) if entities else 'none'} | "
            f"Agents: {', '.join(agent.value for agent in target_agents)}"
        )

        return QueryAnalysis(
            intent=intent,
            entities=entities,
            target_agents=target_agents,
            reasoning=reasoning,
            confidence=confidence,
        )

    @staticmethod
    def _resolve_intent(raw_intent: Any) -> QueryIntent:
        if isinstance(raw_intent, str):
            intent_text = raw_intent.strip().lower()
            try:
                return QueryIntent(intent_text)
            except ValueError:
                return QueryIntent.GENERAL
        return QueryIntent.GENERAL

    @staticmethod
    def _normalize_entities(raw_entities: Any) -> list[str]:
        if not isinstance(raw_entities, list):
            return []

        seen: set[str] = set()
        normalized: list[str] = []
        for item in raw_entities:
            if not isinstance(item, str):
                continue
            entity = item.strip()
            if not entity:
                continue
            if entity in seen:
                continue
            seen.add(entity)
            normalized.append(entity)
        return normalized

    def _resolve_target_agents(
        self,
        llm_analysis: dict[str, Any] | None,
        intent: QueryIntent,
    ) -> list[AgentName]:
        policy_agents = ROUTING_POLICY.get(intent, ROUTING_POLICY[QueryIntent.GENERAL])

        if llm_analysis and isinstance(llm_analysis.get("target_agents"), list):
            parsed_agents: list[AgentName] = []
            seen: set[AgentName] = set()
            for raw in llm_analysis["target_agents"]:
                if not isinstance(raw, str):
                    continue
                try:
                    agent = AgentName(raw.strip())
                except ValueError:
                    continue
                if agent in seen:
                    continue
                seen.add(agent)
                parsed_agents.append(agent)
            if parsed_agents:
                for agent in policy_agents:
                    if agent not in seen:
                        parsed_agents.append(agent)
                return parsed_agents

        return policy_agents

    @staticmethod
    def _normalize_confidence(raw_confidence: Any) -> float:
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            return 0.3
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any] | None:
        payload = text.strip()
        if not payload:
            return None

        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return parsed
            return None
        except json.JSONDecodeError:
            start = payload.find("{")
            end = payload.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                parsed = json.loads(payload[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
                return None
            except json.JSONDecodeError:
                return None

    async def _llm_plan(
        self,
        message: str,
        conversation_context: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        context_str = ""
        if conversation_context:
            recent = conversation_context[-4:]
            context_str = "\n".join(
                f"{item.get('role', 'user')}: {item.get('content', '')[:200]}"
                for item in recent
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a query planner for a multi-agent code analysis system. "
                    "Analyze the user request and return ONLY valid JSON.\n\n"
                    "Allowed intents:\n"
                    "- factual_lookup\n"
                    "- relationship_query\n"
                    "- deep_analysis\n"
                    "- code_explanation\n"
                    "- pattern_search\n"
                    "- indexing_operation\n"
                    "- general\n\n"
                    "Allowed target_agents:\n"
                    "- graph_query\n"
                    "- code_analyst\n"
                    "- retrieval_memory\n"
                    "- indexer\n\n"
                    "Return this exact JSON schema:\n"
                    "{\n"
                    '  "intent": "<allowed intent>",\n'
                    '  "entities": ["EntityName", "function_name"],\n'
                    '  "target_agents": ["graph_query", "retrieval_memory"],\n'
                    '  "confidence": 0.0,\n'
                    '  "reasoning": "short reason"\n'
                    "}\n\n"
                    "Rules:\n"
                    "- entities should include explicit code entities if present.\n"
                    "- confidence must be a float in the 0.0-1.0 range.\n"
                    "- choose indexer only for indexing requests."
                ),
            }
        ]

        if context_str:
            messages.append(
                {
                    "role": "user",
                    "content": f"Conversation context:\n{context_str}\n\nNew query: {message}",
                }
            )
        else:
            messages.append({"role": "user", "content": message})

        result = await self.llm.generate(messages, task="analyze_query")
        return self._parse_json_object(str(result.get("content", "")))
