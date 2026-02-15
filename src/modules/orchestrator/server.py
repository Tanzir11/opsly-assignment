
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.configs import get_api_config, get_app_config
from src.core.logging import setup_logging
from src.core.mcp_client import MCPClient
from src.core.models import AgentName
from src.modules.orchestrator.tools import OrchestratorTools

logger = setup_logging("orchestrator-agent")
config = get_app_config()
api_config = get_api_config()

mcp_client: MCPClient | None = None
tools: OrchestratorTools | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global mcp_client, tools

    agent_urls = {
        AgentName.INDEXER: config.indexer_agent_url,
        AgentName.GRAPH_QUERY: config.graph_query_agent_url,
        AgentName.CODE_ANALYST: config.code_analyst_agent_url,
        AgentName.RETRIEVAL_MEMORY: config.retrieval_memory_agent_url,
    }

    mcp_client = MCPClient(
        agent_urls=agent_urls,
        timeout=config.agent_timeout_seconds,
        max_retries=config.agent_max_retries,
    )

    # Build LLM API keys
    llm_api_keys: dict[str, list[str]] = {}
    if api_config.openai_api_keys:
        llm_api_keys["openai"] = api_config.openai_api_keys
    if api_config.anthropic_api_keys:
        llm_api_keys["anthropic"] = api_config.anthropic_api_keys
    if api_config.gemini_api_keys:
        llm_api_keys["gemini"] = api_config.gemini_api_keys

    tools = OrchestratorTools(
        mcp_client=mcp_client,
        llm_api_keys=llm_api_keys,
    )

    logger.info("Orchestrator agent started on port 8001")
    yield
    if mcp_client:
        await mcp_client.close()


app = FastAPI(title="Orchestrator Agent", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    agent_health = {}
    if mcp_client:
        for agent in [AgentName.INDEXER, AgentName.GRAPH_QUERY,
                      AgentName.CODE_ANALYST, AgentName.RETRIEVAL_MEMORY]:
            result = await mcp_client.health_check(agent)
            agent_health[agent.value] = result["healthy"]

    return {
        "agent": "orchestrator",
        "healthy": True,
        "downstream_agents": agent_health,
    }


@app.post("/mcp/call_tool")
async def call_tool(request: dict[str, Any]) -> JSONResponse:
    if tools is None:
        return JSONResponse(
            {"success": False, "error": "Agent not initialized"}, status_code=503
        )

    tool_name = request.get("tool_name", "")
    arguments = request.get("arguments", {})

    tool_map = {
        "analyze_query": tools.analyze_query,
        "route_to_agents": tools.route_to_agents,
        "get_conversation_context": tools.get_conversation_context,
        "synthesize_response": tools.synthesize_response,
        "process_message": tools.process_message,
    }

    handler = tool_map.get(tool_name)
    if not handler:
        return JSONResponse(
            {"success": False, "error": f"Unknown tool: {tool_name}"}, status_code=400
        )

    try:
        result = await handler(**arguments)
        return JSONResponse({"success": True, "result": result})
    except Exception as exc:
        logger.error("Tool %s failed: %s", tool_name, exc, exc_info=True)
        return JSONResponse(
            {"success": False, "error": str(exc)}, status_code=500
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
