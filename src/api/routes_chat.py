
from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from src.core.auth import require_jwt_user
from src.core.logging import correlation_id_var, get_logger, session_id_var
from src.core.models import AgentName, ChatRequest, ChatResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=None)
async def chat(
    request: Request,
    body: ChatRequest,
    _: str = Depends(require_jwt_user),
) -> Any:
    # Generate session ID if not provided
    session_id = body.session_id or uuid.uuid4().hex[:16]
    session_id_var.set(session_id)

    # Get the MCP client from app state
    mcp_client = request.app.state.mcp_client

    if body.stream:
        return EventSourceResponse(
            _stream_response(mcp_client, session_id, body.message)
        )

    # Standard response
    try:
        response = await mcp_client.call_tool(
            AgentName.ORCHESTRATOR,
            "process_message",
            {
                "session_id": session_id,
                "message": body.message,
            },
        )

        if response.success and response.result:
            result = response.result
            chat_resp = ChatResponse(
                session_id=session_id,
                response=result.get("response", ""),
                agents_used=result.get("agents_used", []),
                sources=None,
                correlation_id=correlation_id_var.get(""),
            )
            return JSONResponse(chat_resp.model_dump(mode="json"))
        else:
            return JSONResponse(
                {
                    "session_id": session_id,
                    "response": f"Sorry, I couldn't process your request: {response.error}",
                    "agents_used": [],
                    "error": response.error,
                },
                status_code=500,
            )

    except Exception as exc:
        logger.error("Chat endpoint error: %s", exc, exc_info=True)
        return JSONResponse(
            {
                "session_id": session_id,
                "response": "An internal error occurred. Please try again.",
                "error": str(exc),
            },
            status_code=500,
        )


async def _stream_response(
    mcp_client: Any,
    session_id: str,
    message: str,
) -> Any:
    try:
        # Send initial event
        yield {
            "event": "start",
            "data": json.dumps({"session_id": session_id, "status": "processing"}),
        }

        # Step 1: Analyze query
        yield {
            "event": "status",
            "data": json.dumps({"step": "analyzing_query"}),
        }

        analysis_resp = await mcp_client.call_tool(
            AgentName.ORCHESTRATOR,
            "analyze_query",
            {"message": message, "session_id": session_id},
        )

        if analysis_resp.success:
            analysis = analysis_resp.result
            yield {
                "event": "analysis",
                "data": json.dumps(analysis),
            }
        else:
            yield {
                "event": "error",
                "data": json.dumps({"error": analysis_resp.error}),
            }
            return

        # Step 2: Route to agents
        yield {
            "event": "status",
            "data": json.dumps({"step": "routing_to_agents", "agents": analysis.get("target_agents", [])}),
        }

        routing_resp = await mcp_client.call_tool(
            AgentName.ORCHESTRATOR,
            "route_to_agents",
            {
                "message": message,
                "intent": analysis["intent"],
                "entities": analysis["entities"],
                "target_agents": [a.value if hasattr(a, "value") else a for a in analysis["target_agents"]],
                "session_id": session_id,
            },
        )

        if routing_resp.success:
            yield {
                "event": "agent_results",
                "data": json.dumps({"agents_called": routing_resp.result.get("agents_called", 0)}),
            }
        else:
            yield {
                "event": "error",
                "data": json.dumps({"error": routing_resp.error}),
            }
            return

        # Step 3: Synthesize
        yield {
            "event": "status",
            "data": json.dumps({"step": "synthesizing"}),
        }

        synth_resp = await mcp_client.call_tool(
            AgentName.ORCHESTRATOR,
            "synthesize_response",
            {
                "query": message,
                "agent_responses": routing_resp.result.get("responses", []),
                "session_id": session_id,
            },
        )

        if synth_resp.success:
            result = synth_resp.result
            yield {
                "event": "response",
                "data": json.dumps({
                    "session_id": session_id,
                    "response": result.get("response", ""),
                    "agents_used": result.get("agents_used", []),
                    "sources": result.get("sources", []),
                }),
            }
        else:
            yield {
                "event": "error",
                "data": json.dumps({"error": synth_resp.error}),
            }

        yield {"event": "done", "data": json.dumps({"status": "complete"})}

    except Exception as exc:
        yield {
            "event": "error",
            "data": json.dumps({"error": str(exc)}),
        }
