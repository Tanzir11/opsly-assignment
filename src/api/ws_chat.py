
from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.core.auth import get_jwt_subject_from_authorization
from src.core.logging import get_logger, session_id_var
from src.core.models import AgentName

logger = get_logger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    authorization = websocket.headers.get("authorization")
    if not authorization:
        query_token = websocket.query_params.get("token")
        if query_token:
            authorization = f"Bearer {query_token}"

    secret = websocket.app.state.config.auth_jwt_secret
    try:
        get_jwt_subject_from_authorization(authorization, secret)
    except ValueError:
        await websocket.close(code=1008, reason="Unauthorized")
        return

    await websocket.accept()
    mcp_client = websocket.app.state.mcp_client

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = data.get("message", "")
            session_id = data.get("session_id") or uuid.uuid4().hex[:16]
            session_id_var.set(session_id)

            if not message:
                await websocket.send_json({
                    "type": "error",
                    "error": "Message is required",
                })
                continue

            await _process_ws_message(websocket, mcp_client, session_id, message)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc, exc_info=True)
        try:
            await websocket.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass


async def _process_ws_message(
    websocket: WebSocket,
    mcp_client: Any,
    session_id: str,
    message: str,
) -> None:
    try:
        # Start
        await websocket.send_json({
            "type": "start",
            "session_id": session_id,
        })

        # Store user message
        try:
            await mcp_client.call_tool(
                AgentName.RETRIEVAL_MEMORY,
                "store_conversation_turn",
                {"session_id": session_id, "role": "user", "content": message},
            )
        except Exception:
            pass

        # Analyze query
        await websocket.send_json({"type": "status", "step": "analyzing_query"})

        analysis_resp = await mcp_client.call_tool(
            AgentName.ORCHESTRATOR,
            "analyze_query",
            {"message": message, "session_id": session_id},
        )

        if not analysis_resp.success:
            await websocket.send_json({
                "type": "error",
                "error": analysis_resp.error or "Query analysis failed",
            })
            return

        analysis = analysis_resp.result
        await websocket.send_json({
            "type": "analysis",
            "data": analysis,
        })

        # Route to agents
        target_agents = [
            a.value if hasattr(a, "value") else a
            for a in analysis.get("target_agents", [])
        ]
        await websocket.send_json({
            "type": "status",
            "step": "routing_to_agents",
            "agents": target_agents,
        })

        routing_resp = await mcp_client.call_tool(
            AgentName.ORCHESTRATOR,
            "route_to_agents",
            {
                "message": message,
                "intent": analysis["intent"],
                "entities": analysis["entities"],
                "target_agents": target_agents,
                "session_id": session_id,
            },
        )

        if not routing_resp.success:
            await websocket.send_json({
                "type": "error",
                "error": routing_resp.error or "Agent routing failed",
            })
            return

        # Send individual agent results
        for resp in routing_resp.result.get("responses", []):
            await websocket.send_json({
                "type": "agent_result",
                "agent": resp.get("agent", ""),
                "tool": resp.get("tool", ""),
                "success": resp.get("success", False),
            })

        # Synthesize
        await websocket.send_json({"type": "status", "step": "synthesizing"})

        synth_resp = await mcp_client.call_tool(
            AgentName.ORCHESTRATOR,
            "synthesize_response",
            {
                "query": message,
                "agent_responses": routing_resp.result.get("responses", []),
                "session_id": session_id,
            },
        )

        if synth_resp.success and synth_resp.result:
            result = synth_resp.result
            await websocket.send_json({
                "type": "response",
                "data": {
                    "session_id": session_id,
                    "response": result.get("response", ""),
                    "agents_used": result.get("agents_used", []),
                    "sources": result.get("sources", []),
                },
            })

            # Store assistant response
            try:
                await mcp_client.call_tool(
                    AgentName.RETRIEVAL_MEMORY,
                    "store_conversation_turn",
                    {
                        "session_id": session_id,
                        "role": "assistant",
                        "content": result.get("response", "")[:5000],
                    },
                )
            except Exception:
                pass
        else:
            await websocket.send_json({
                "type": "error",
                "error": synth_resp.error or "Synthesis failed",
            })

        await websocket.send_json({"type": "done"})

    except Exception as exc:
        logger.error("WS processing error: %s", exc, exc_info=True)
        await websocket.send_json({"type": "error", "error": str(exc)})
