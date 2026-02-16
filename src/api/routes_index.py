
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from src.core.auth import require_jwt_user
from src.core.logging import get_logger
from src.core.models import AgentName, IndexRequest

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["indexing"])


@router.post("/index")
async def start_indexing(
    request: Request,
    body: IndexRequest,
    _: str = Depends(require_jwt_user),
) -> JSONResponse:
    mcp_client = request.app.state.mcp_client

    try:
        repo_url = body.repo_url.strip()
        if "github.com" not in repo_url:
            return JSONResponse(
                {"error": "repo_url must be a valid GitHub repository URL"},
                status_code=400,
            )

        response = await mcp_client.call_tool(
            AgentName.INDEXER,
            "index_repository",
            {
                "repo_url": repo_url,
                "incremental": body.incremental,
            },
        )

        if response.success:
            return JSONResponse(response.result, status_code=202)
        else:
            return JSONResponse(
                {"error": response.error}, status_code=500
            )

    except Exception as exc:
        logger.error("Index endpoint error: %s", exc, exc_info=True)
        return JSONResponse(
            {"error": f"Failed to start indexing: {exc}"},
            status_code=500,
        )


@router.get("/index/status/{job_id}")
async def get_index_status(
    request: Request,
    job_id: str,
    _: str = Depends(require_jwt_user),
) -> JSONResponse:
    mcp_client = request.app.state.mcp_client

    try:
        response = await mcp_client.call_tool(
            AgentName.INDEXER,
            "get_index_status",
            {"job_id": job_id},
        )

        if response.success:
            return JSONResponse(response.result)
        else:
            return JSONResponse(
                {"error": response.error}, status_code=500
            )

    except Exception as exc:
        logger.error("Index status error: %s", exc, exc_info=True)
        return JSONResponse(
            {"error": f"Failed to get status: {exc}"},
            status_code=500,
        )
