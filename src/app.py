
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes_chat import router as chat_router
from src.api.routes_graph import router as graph_router
from src.api.routes_health import router as health_router
from src.api.routes_index import router as index_router
from src.api.ws_chat import router as ws_router
from src.configs import get_app_config
from src.core.logging import (
    correlation_id_var,
    generate_correlation_id,
    session_id_var,
    setup_logging,
)
from src.core.mcp_client import MCPClient
from src.core.models import AgentName

config = get_app_config()
logger = setup_logging("api-gateway", level=config.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Build agent URL map (orchestrator is the primary target for gateway)
    agent_urls = {
        AgentName.ORCHESTRATOR: config.orchestrator_agent_url,
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

    # Attach to app state so route handlers can access it
    app.state.mcp_client = mcp_client
    app.state.config = config

    logger.info(
        "API Gateway started on port %d, env=%s",
        config.api_port,
        config.app_env,
    )

    yield

    # Cleanup
    await mcp_client.close()
    logger.info("API Gateway shut down")


app = FastAPI(
    title="Opsly - FastAPI Codebase Chat Agent",
    description=(
        "Multi-agent MCP system for answering questions about the FastAPI codebase. "
        "Routes queries through specialized agents (orchestrator, indexer, graph query, "
        "code analyst, retrieval-memory) for comprehensive code analysis."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ─── CORS Middleware ────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Correlation ID Middleware ──────────────────────────────────────────────

@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next: object) -> Response:
    # Correlation ID: from header or generate
    corr_id = request.headers.get("X-Correlation-ID", generate_correlation_id())
    correlation_id_var.set(corr_id)

    # Session ID: from header or query param
    sess_id = (
        request.headers.get("X-Session-ID", "")
        or request.query_params.get("session_id", "")
    )
    if sess_id:
        session_id_var.set(sess_id)

    response = await call_next(request)  # type: ignore[operator]
    response.headers["X-Correlation-ID"] = corr_id
    return response


# ─── Include routers ────────────────────────────────────────────────────────

app.include_router(chat_router)
app.include_router(index_router)
app.include_router(health_router)
app.include_router(graph_router)
app.include_router(ws_router)


# ─── Root endpoint ──────────────────────────────────────────────────────────

@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "Opsly API Gateway",
        "version": "0.1.0",
        "docs": "/docs",
        "environment": config.app_env,
    }


@app.get("/health")
async def gateway_health() -> dict[str, object]:
    return {
        "service": "api-gateway",
        "healthy": True,
        "environment": config.app_env,
    }


# ─── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=config.api_port,
        reload=config.debug,
        log_level=config.log_level.lower(),
    )
