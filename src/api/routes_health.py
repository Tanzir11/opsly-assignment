
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.core.logging import get_logger
from src.core.models import AgentHealth, AgentName, SystemHealth

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/agents/health")
async def agents_health(request: Request) -> JSONResponse:
    mcp_client = request.app.state.mcp_client
    agents: list[AgentHealth] = []
    db_health: dict[str, bool] = {}

    # Check all agents
    for agent in [
        AgentName.ORCHESTRATOR,
        AgentName.INDEXER,
        AgentName.GRAPH_QUERY,
        AgentName.CODE_ANALYST,
        AgentName.RETRIEVAL_MEMORY,
    ]:
        url = mcp_client.agent_urls.get(agent, "")
        result = await mcp_client.health_check(agent)
        agents.append(
            AgentHealth(
                agent=agent.value,
                healthy=result.get("healthy", False),
                url=url,
                latency_ms=result.get("latency_ms"),
                error=result.get("error"),
            )
        )

    # Check database dependencies via orchestrator health
    try:
        orch_health = await mcp_client.call_tool(
            AgentName.ORCHESTRATOR,
            "analyze_query",
            {"message": "health check ping", "session_id": "__health__"},
        )
        # If orchestrator responds, its DB deps are likely fine
        db_health["orchestrator_reachable"] = orch_health.success
    except Exception:
        db_health["orchestrator_reachable"] = False

    all_healthy = all(a.healthy for a in agents)

    system = SystemHealth(
        healthy=all_healthy,
        agents=agents,
        databases=db_health,
    )

    status_code = 200 if all_healthy else 503
    return JSONResponse(system.model_dump(mode="json"), status_code=status_code)
