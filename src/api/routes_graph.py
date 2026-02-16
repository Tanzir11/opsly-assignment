
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from src.core.auth import require_jwt_user
from src.core.logging import get_logger
from src.core.models import AgentName

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["graph"])


@router.get("/graph/statistics")
async def graph_statistics(
    request: Request,
    _: str = Depends(require_jwt_user),
) -> JSONResponse:
    mcp_client = request.app.state.mcp_client

    try:
        # Get graph statistics from graph query agent
        response = await mcp_client.call_tool(
            AgentName.GRAPH_QUERY,
            "get_statistics",
            {},
        )

        if response.success and response.result:
            return JSONResponse(response.result)
        return JSONResponse(
            {"error": response.error or "Failed to load graph statistics"},
            status_code=500,
        )

    except Exception as exc:
        logger.error("Graph statistics error: %s", exc, exc_info=True)
        return JSONResponse(
            {"error": f"Failed to get statistics: {exc}"},
            status_code=500,
        )
