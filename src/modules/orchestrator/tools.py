
from __future__ import annotations

import time
from typing import Any

from src.core.llm_router import LLMRouter
from src.core.logging import get_logger
from src.core.mcp_client import MCPClient
from src.core.models import AgentName, AgentResponse, QueryIntent
from src.modules.orchestrator.planner import QueryPlanner
from src.modules.orchestrator.synthesizer import ResponseSynthesizer

logger = get_logger(__name__)

# ─── Tool mapping: intent + agent -> tool name and arguments builder ────────


def _looks_like_class_name(name: str) -> bool:
    trimmed = name.strip()
    if not trimmed:
        return False
    return trimmed[0].isupper() and "_" not in trimmed


def _infer_entity_type(entities: list[str], query: str) -> str:
    candidate = entities[0] if entities else query.split()[0] if query.split() else ""
    return "class" if _looks_like_class_name(candidate) else "function"


def _build_graph_find_entity_args(entities: list[str], query: str) -> dict[str, Any]:
    name = entities[0] if entities else query[:50]
    args: dict[str, Any] = {"name": name}
    if _looks_like_class_name(name):
        args["entity_type"] = "Class"
    elif "_" in name:
        args["entity_type"] = "Function"
    return args


AGENT_TOOL_DISPATCH: dict[tuple[str, AgentName], dict[str, Any]] = {
    # Graph Query tools
    ("factual_lookup", AgentName.GRAPH_QUERY): {
        "tool": "find_entity",
        "arg_builder": _build_graph_find_entity_args,
    },
    ("relationship_query", AgentName.GRAPH_QUERY): {
        "tool": "find_entity",
        "arg_builder": _build_graph_find_entity_args,
    },
    ("deep_analysis", AgentName.GRAPH_QUERY): {
        "tool": "find_entity",
        "arg_builder": _build_graph_find_entity_args,
    },
    ("code_explanation", AgentName.GRAPH_QUERY): {
        "tool": "find_entity",
        "arg_builder": _build_graph_find_entity_args,
    },
    ("general", AgentName.GRAPH_QUERY): {
        "tool": "find_entity",
        "arg_builder": _build_graph_find_entity_args,
    },
    # Code Analyst tools
    ("code_explanation", AgentName.CODE_ANALYST): {
        "tool": "explain_implementation",
        "arg_builder": lambda entities, query: {
            "name": entities[0] if entities else query[:50],
            "entity_type": _infer_entity_type(entities, query),
        },
    },
    ("deep_analysis", AgentName.CODE_ANALYST): {
        "tool": "explain_implementation",
        "arg_builder": lambda entities, query: {
            "name": entities[0] if entities else query[:50],
            "entity_type": _infer_entity_type(entities, query),
        },
    },
    ("pattern_search", AgentName.CODE_ANALYST): {
        "tool": "find_patterns",
        "arg_builder": lambda entities, query: {},
    },
    # Retrieval Memory tools
    ("factual_lookup", AgentName.RETRIEVAL_MEMORY): {
        "tool": "semantic_search_chunks",
        "arg_builder": lambda entities, query: {"query": query, "top_k": 5},
    },
    ("relationship_query", AgentName.RETRIEVAL_MEMORY): {
        "tool": "semantic_search_chunks",
        "arg_builder": lambda entities, query: {"query": query, "top_k": 5},
    },
    ("deep_analysis", AgentName.RETRIEVAL_MEMORY): {
        "tool": "semantic_search_chunks",
        "arg_builder": lambda entities, query: {"query": query, "top_k": 10},
    },
    ("code_explanation", AgentName.RETRIEVAL_MEMORY): {
        "tool": "semantic_search_chunks",
        "arg_builder": lambda entities, query: {"query": query, "top_k": 5},
    },
    ("general", AgentName.RETRIEVAL_MEMORY): {
        "tool": "semantic_search_chunks",
        "arg_builder": lambda entities, query: {"query": query, "top_k": 5},
    },
    # Indexer tools
    ("indexing_operation", AgentName.INDEXER): {
        "tool": "index_repository",
        "arg_builder": lambda entities, query: {},
    },
}


class OrchestratorTools:

    def __init__(
        self,
        mcp_client: MCPClient,
        llm_api_keys: dict[str, list[str]] | None = None,
    ) -> None:
        self.mcp = mcp_client
        self.llm = LLMRouter(api_keys=llm_api_keys or {})
        self.planner = QueryPlanner(self.llm)
        self.synthesizer = ResponseSynthesizer(self.llm)

    async def analyze_query(
        self,
        message: str,
        session_id: str = "",
    ) -> dict[str, Any]:
        # Get conversation context if session exists
        context = None
        if session_id:
            try:
                ctx_resp = await self.mcp.call_tool(
                    AgentName.RETRIEVAL_MEMORY,
                    "get_relevant_memory",
                    {"session_id": session_id, "num_turns": 3},
                )
                if ctx_resp.success:
                    context = ctx_resp.result.get("conversation_context", [])
            except Exception:
                pass

        analysis = await self.planner.analyze_query(message, session_id, context)
        return analysis.model_dump()

    async def route_to_agents(
        self,
        message: str,
        intent: str,
        entities: list[str],
        target_agents: list[str],
        session_id: str = "",
    ) -> dict[str, Any]:
        start = time.monotonic()

        # Build calls for each agent
        calls: list[tuple[AgentName, str, dict[str, Any]]] = []

        for agent_str in target_agents:
            try:
                agent = AgentName(agent_str)
            except ValueError:
                continue

            dispatch_key = (intent, agent)
            dispatch = AGENT_TOOL_DISPATCH.get(dispatch_key)

            if dispatch:
                tool_name = dispatch["tool"]
                arg_builder = dispatch["arg_builder"]
                arguments = arg_builder(entities, message)
                calls.append((agent, tool_name, arguments))
            else:
                # Default fallback tools per agent
                if agent == AgentName.GRAPH_QUERY:
                    name = entities[0] if entities else message[:50]
                    calls.append((agent, "find_entity", {"name": name}))
                elif agent == AgentName.CODE_ANALYST:
                    name = entities[0] if entities else message[:50]
                    calls.append((agent, "explain_implementation", {"name": name}))
                elif agent == AgentName.RETRIEVAL_MEMORY:
                    calls.append((agent, "semantic_search_chunks", {"query": message, "top_k": 5}))

        # Execute calls in parallel
        responses = await self.mcp.call_tools_parallel(calls)

        latency_ms = (time.monotonic() - start) * 1000

        return {
            "responses": [
                {
                    "agent": r.agent.value,
                    "tool": r.tool_name,
                    "success": r.success,
                    "result": r.result,
                    "error": r.error,
                    "latency_ms": r.latency_ms,
                }
                for r in responses
            ],
            "total_latency_ms": round(latency_ms, 2),
            "agents_called": len(calls),
        }

    async def get_conversation_context(
        self,
        session_id: str,
        num_turns: int = 5,
    ) -> dict[str, Any]:
        try:
            response = await self.mcp.call_tool(
                AgentName.RETRIEVAL_MEMORY,
                "get_relevant_memory",
                {"session_id": session_id, "num_turns": num_turns},
            )
            if response.success:
                return response.result
            return {"error": response.error, "context": []}
        except Exception as exc:
            return {"error": str(exc), "context": []}

    async def synthesize_response(
        self,
        query: str,
        agent_responses: list[dict[str, Any]],
        session_id: str = "",
    ) -> dict[str, Any]:
        # Convert dicts back to AgentResponse objects
        responses = []
        for r in agent_responses:
            try:
                agent = AgentName(r.get("agent", "orchestrator"))
            except ValueError:
                agent = AgentName.ORCHESTRATOR
            responses.append(
                AgentResponse(
                    agent=agent,
                    tool_name=r.get("tool", "unknown"),
                    success=r.get("success", False),
                    result=r.get("result"),
                    error=r.get("error"),
                    latency_ms=r.get("latency_ms", 0),
                )
            )

        # Get conversation context
        context = None
        if session_id:
            try:
                ctx_result = await self.get_conversation_context(session_id, 3)
                context = ctx_result.get("conversation_context", [])
            except Exception:
                pass

        return await self.synthesizer.synthesize(query, responses, context)

    async def process_message(
        self,
        session_id: str,
        message: str,
    ) -> dict[str, Any]:
        start = time.monotonic()

        # Step 1: Store user message
        try:
            await self.mcp.call_tool(
                AgentName.RETRIEVAL_MEMORY,
                "store_conversation_turn",
                {
                    "session_id": session_id,
                    "role": "user",
                    "content": message,
                },
            )
        except Exception as exc:
            logger.warning("Failed to store user message: %s", exc)

        # Step 2: Analyze query
        analysis = await self.analyze_query(message, session_id)

        # Step 3: Route to agents
        routing_result = await self.route_to_agents(
            message=message,
            intent=analysis["intent"],
            entities=analysis["entities"],
            target_agents=[a.value if hasattr(a, "value") else a for a in analysis["target_agents"]],
            session_id=session_id,
        )

        # Step 4: Synthesize response
        synthesis = await self.synthesize_response(
            query=message,
            agent_responses=routing_result["responses"],
            session_id=session_id,
        )

        # Step 5: Store assistant response
        try:
            await self.mcp.call_tool(
                AgentName.RETRIEVAL_MEMORY,
                "store_conversation_turn",
                {
                    "session_id": session_id,
                    "role": "assistant",
                    "content": synthesis.get("response", "")[:5000],
                },
            )
        except Exception as exc:
            logger.warning("Failed to store assistant response: %s", exc)

        total_latency_ms = (time.monotonic() - start) * 1000

        return {
            "response": synthesis.get("response", ""),
            "agents_used": synthesis.get("agents_used", []),
            "sources": synthesis.get("sources", []),
            "analysis": {
                "intent": analysis["intent"],
                "entities": analysis["entities"],
                "confidence": analysis["confidence"],
            },
            "total_latency_ms": round(total_latency_ms, 2),
        }
