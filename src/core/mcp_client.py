
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from src.core.exceptions import (
    AgentConnectionError,
    AgentTimeoutError,
    AgentToolError,
)
from src.core.logging import correlation_id_var, get_logger
from src.core.models import AgentName, AgentResponse

logger = get_logger(__name__)


class MCPClient:

    def __init__(
        self,
        agent_urls: dict[AgentName, str] | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        retry_delay: float = 0.5,
    ) -> None:
        self.agent_urls = agent_urls or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=5.0),
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def call_tool(
        self,
        agent: AgentName,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> AgentResponse:
        base_url = self.agent_urls.get(agent)
        if not base_url:
            raise AgentConnectionError(
                f"No URL configured for agent {agent.value}",
                agent=agent.value,
            )

        url = f"{base_url}/mcp/call_tool"
        payload = {
            "tool_name": tool_name,
            "arguments": arguments or {},
            "correlation_id": correlation_id_var.get(""),
        }

        effective_timeout = timeout or self.timeout
        start = time.monotonic()
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                client = await self._get_client()
                response = await client.post(
                    url,
                    json=payload,
                    timeout=effective_timeout,
                )

                latency_ms = (time.monotonic() - start) * 1000

                if response.status_code == 200:
                    data = response.json()
                    return AgentResponse(
                        agent=agent,
                        tool_name=tool_name,
                        success=data.get("success", True),
                        result=data.get("result"),
                        error=data.get("error"),
                        latency_ms=round(latency_ms, 2),
                    )
                else:
                    error_detail = response.text[:500]
                    raise AgentToolError(
                        f"Agent {agent.value} returned {response.status_code}: {error_detail}",
                        agent=agent.value,
                        tool=tool_name,
                    )

            except httpx.TimeoutException as exc:
                latency_ms = (time.monotonic() - start) * 1000
                last_error = AgentTimeoutError(
                    f"Agent {agent.value} timed out after {latency_ms:.0f}ms",
                    agent=agent.value,
                )
            except httpx.ConnectError as exc:
                last_error = AgentConnectionError(
                    f"Cannot connect to agent {agent.value} at {base_url}: {exc}",
                    agent=agent.value,
                )
            except AgentToolError:
                raise
            except Exception as exc:
                last_error = exc

            # Exponential backoff before retry
            if attempt < self.max_retries:
                delay = self.retry_delay * (2**attempt)
                logger.warning(
                    "Retrying agent %s tool %s (attempt %d/%d) after %.1fs",
                    agent.value,
                    tool_name,
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        latency_ms = (time.monotonic() - start) * 1000
        if isinstance(last_error, (AgentConnectionError, AgentTimeoutError)):
            raise last_error

        raise AgentConnectionError(
            f"Agent {agent.value} unreachable after {self.max_retries + 1} attempts: {last_error}",
            agent=agent.value,
        )

    async def health_check(self, agent: AgentName) -> dict[str, Any]:
        base_url = self.agent_urls.get(agent)
        if not base_url:
            return {"healthy": False, "error": f"No URL for {agent.value}"}

        start = time.monotonic()
        try:
            client = await self._get_client()
            resp = await client.get(f"{base_url}/health", timeout=5.0)
            latency_ms = (time.monotonic() - start) * 1000
            return {
                "healthy": resp.status_code == 200,
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            return {
                "healthy": False,
                "latency_ms": round(latency_ms, 2),
                "error": str(exc),
            }

    async def call_tools_parallel(
        self,
        calls: list[tuple[AgentName, str, dict[str, Any]]],
        timeout: float | None = None,
    ) -> list[AgentResponse]:
        tasks = [
            self.call_tool(agent, tool_name, arguments, timeout)
            for agent, tool_name, arguments in calls
        ]
        results: list[AgentResponse] = []
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
            except Exception as exc:
                # Return error response instead of propagating
                results.append(
                    AgentResponse(
                        agent=AgentName.ORCHESTRATOR,
                        tool_name="unknown",
                        success=False,
                        error=str(exc),
                    )
                )
        return results
