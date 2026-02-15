
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.configs import get_api_config, get_app_config
from src.core.logging import setup_logging
from src.db.neo4j import Neo4jConnection, Neo4jRepository
from src.modules.code_analyst.tools import CodeAnalystTools

logger = setup_logging("code-analyst-agent")
config = get_app_config()
api_config = get_api_config()

neo4j_conn = Neo4jConnection(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
tools: CodeAnalystTools | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global tools
    await neo4j_conn.connect()
    neo4j_repo = Neo4jRepository(neo4j_conn)

    # Build API keys for LLM router
    llm_api_keys: dict[str, list[str]] = {}
    if api_config.openai_api_keys:
        llm_api_keys["openai"] = api_config.openai_api_keys
    if api_config.anthropic_api_keys:
        llm_api_keys["anthropic"] = api_config.anthropic_api_keys
    if api_config.gemini_api_keys:
        llm_api_keys["gemini"] = api_config.gemini_api_keys

    tools = CodeAnalystTools(
        neo4j_repo=neo4j_repo,
        repo_workspace=config.repo_workspace,
        llm_api_keys=llm_api_keys,
    )
    logger.info("Code Analyst agent started on port 8004")
    yield
    await neo4j_conn.close()


app = FastAPI(title="Code Analyst Agent", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "agent": "code_analyst",
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
        "analyze_function": tools.analyze_function,
        "analyze_class": tools.analyze_class,
        "find_patterns": tools.find_patterns,
        "get_code_snippet": tools.get_code_snippet,
        "explain_implementation": tools.explain_implementation,
        "compare_implementations": tools.compare_implementations,
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
    uvicorn.run(app, host="0.0.0.0", port=8004)
