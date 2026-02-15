
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.configs import get_api_config, get_app_config
from src.core.logging import setup_logging
from src.db.milvus import MilvusConnection, MilvusRepository
from src.db.mongodb import MongoDBConnection, MongoDBRepository
from src.db.neo4j import Neo4jConnection, Neo4jRepository
from src.modules.indexer.tools import IndexerTools

logger = setup_logging("indexer-agent")
config = get_app_config()
api_config = get_api_config()

# Shared state
neo4j_conn = Neo4jConnection(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
mongo_conn = MongoDBConnection(config.mongodb_uri, config.mongodb_db_name)
milvus_conn = (
    MilvusConnection(config.milvus_host, config.milvus_port)
    if config.enable_milvus
    else None
)
tools: IndexerTools | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global tools
    await neo4j_conn.connect()
    await mongo_conn.connect()
    if milvus_conn is not None:
        await milvus_conn.connect()

    neo4j_repo = Neo4jRepository(neo4j_conn)
    mongo_repo = MongoDBRepository(mongo_conn)
    milvus_repo = (
        MilvusRepository(milvus_conn, dimension=config.embedding_dimension)
        if milvus_conn is not None
        else None
    )

    embedding_api_keys: dict[str, str] = {}
    if api_config.openai_api_keys:
        embedding_api_keys["openai"] = api_config.openai_api_keys[0]

    tools = IndexerTools(
        neo4j_repo=neo4j_repo,
        mongo_repo=mongo_repo,
        milvus_repo=milvus_repo,
        repo_workspace=config.repo_workspace,
        default_repo_url=config.fastapi_repo_url,
        embedding_api_keys=embedding_api_keys,
        embedding_default_provider=config.embedding_default_provider,
    )
    logger.info("Indexer agent started on port 8002")
    yield
    await neo4j_conn.close()
    await mongo_conn.close()
    if milvus_conn is not None:
        await milvus_conn.close()


app = FastAPI(title="Indexer Agent", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "agent": "indexer",
        "healthy": True,
        "neo4j": await neo4j_conn.health_check(),
        "mongodb": await mongo_conn.health_check(),
        "milvus_enabled": config.enable_milvus,
        "milvus": await milvus_conn.health_check() if milvus_conn is not None else False,
    }


@app.post("/mcp/call_tool")
async def call_tool(request: dict[str, Any]) -> JSONResponse:
    if tools is None:
        return JSONResponse(
            {"success": False, "error": "Agent not initialized"},
            status_code=503,
        )

    tool_name = request.get("tool_name", "")
    arguments = request.get("arguments", {})

    tool_map = {
        "index_repository": tools.index_repository,
        "index_file": tools.index_file,
        "parse_python_ast": tools.parse_python_ast,
        "extract_entities": tools.extract_entities,
        "get_index_status": tools.get_index_status,
    }

    handler = tool_map.get(tool_name)
    if not handler:
        return JSONResponse(
            {"success": False, "error": f"Unknown tool: {tool_name}"},
            status_code=400,
        )

    try:
        result = await handler(**arguments)
        return JSONResponse({"success": True, "result": result})
    except Exception as exc:
        logger.error("Tool %s failed: %s", tool_name, exc, exc_info=True)
        return JSONResponse(
            {"success": False, "error": str(exc)},
            status_code=500,
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
