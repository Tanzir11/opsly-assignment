
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.configs import get_app_config
from src.core.logging import setup_logging
from src.db.neo4j import Neo4jConnection, Neo4jRepository
from src.modules.graph_query.tools import GraphQueryTools

logger = setup_logging("graph-query-agent")
config = get_app_config()

neo4j_conn = Neo4jConnection(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
tools: GraphQueryTools | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global tools
    await neo4j_conn.connect()
    neo4j_repo = Neo4jRepository(neo4j_conn)
    tools = GraphQueryTools(neo4j_repo=neo4j_repo)
    logger.info("Graph Query agent started on port 8003")
    yield
    await neo4j_conn.close()


app = FastAPI(title="Graph Query Agent", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "agent": "graph_query",
        "healthy": True,
        "neo4j": await neo4j_conn.health_check(),
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
        "find_entity": tools.find_entity,
        "get_dependencies": tools.get_dependencies,
        "get_dependents": tools.get_dependents,
        "trace_imports": tools.trace_imports,
        "find_related": tools.find_related,
        "execute_query": tools.execute_query,
        "get_statistics": tools.get_statistics,
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
    uvicorn.run(app, host="0.0.0.0", port=8003)
