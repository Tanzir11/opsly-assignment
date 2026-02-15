
from __future__ import annotations

from typing import Any

from src.core.llm_router import LLMRouter
from src.core.logging import get_logger
from src.core.models import AgentResponse, SourceReference

logger = get_logger(__name__)


class ResponseSynthesizer:

    def __init__(self, llm_router: LLMRouter) -> None:
        self.llm = llm_router

    async def synthesize(
        self,
        query: str,
        agent_responses: list[AgentResponse],
        conversation_context: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        # Filter successful responses
        successful = [r for r in agent_responses if r.success and r.result]
        failed = [r for r in agent_responses if not r.success]

        if not successful:
            return self._build_fallback_response(query, failed)

        # Try LLM synthesis
        try:
            synthesized = await self._llm_synthesize(
                query, successful, conversation_context
            )
            return synthesized
        except Exception as exc:
            logger.warning("LLM synthesis failed, using template: %s", exc)
            return self._template_synthesize(query, successful, failed)

    async def _llm_synthesize(
        self,
        query: str,
        responses: list[AgentResponse],
        conversation_context: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        # Build agent outputs summary
        agent_sections: list[str] = []
        sources: list[SourceReference] = []
        agents_used: list[str] = []

        for resp in responses:
            agent_name = resp.agent.value
            agents_used.append(agent_name)
            result = resp.result

            # Format result for LLM
            if isinstance(result, dict):
                section = f"### {agent_name} ({resp.tool_name})\n"
                section += self._format_result(result)
                agent_sections.append(section)

                # Extract source references
                self._extract_sources(result, sources)
            else:
                agent_sections.append(f"### {agent_name}\n{str(result)[:2000]}")

        agent_data = "\n\n".join(agent_sections)

        # Build context
        context_str = ""
        if conversation_context:
            recent = conversation_context[-3:]
            context_str = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')[:200]}"
                for m in recent
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert code assistant that synthesizes information from "
                    "multiple analysis agents to answer questions about the FastAPI codebase. "
                    "Provide clear, accurate, and well-structured responses. "
                    "Include specific file references and code examples when available. "
                    "Be concise but thorough."
                ),
            },
        ]

        if context_str:
            messages.append({
                "role": "user",
                "content": f"Previous conversation:\n{context_str}",
            })

        messages.append({
            "role": "user",
            "content": (
                f"User Question: {query}\n\n"
                f"Agent Analysis Results:\n{agent_data[:6000]}\n\n"
                "Synthesize a comprehensive answer using all available information. "
                "Reference specific code entities, files, and relationships where relevant."
            ),
        })

        result = await self.llm.generate(messages, task="response_synthesis")

        return {
            "response": result["content"],
            "agents_used": agents_used,
            "sources": [s.model_dump() for s in sources],
            "synthesis_model": result["model"],
            "synthesis_latency_ms": result["latency_ms"],
        }

    def _template_synthesize(
        self,
        query: str,
        successful: list[AgentResponse],
        failed: list[AgentResponse],
    ) -> dict[str, Any]:
        parts: list[str] = []
        agents_used: list[str] = []
        sources: list[SourceReference] = []

        parts.append(f"**Query**: {query}\n")

        for resp in successful:
            agent_name = resp.agent.value
            agents_used.append(agent_name)
            result = resp.result

            parts.append(f"\n**{agent_name}** ({resp.tool_name}):")
            if isinstance(result, dict):
                formatted = self._format_result(result)
                parts.append(formatted)
                self._extract_sources(result, sources)
            else:
                parts.append(str(result)[:1000])

        if failed:
            parts.append("\n*Note: Some agents could not be reached:*")
            for resp in failed:
                parts.append(f"- {resp.agent.value}: {resp.error or 'Unknown error'}")

        return {
            "response": "\n".join(parts),
            "agents_used": agents_used,
            "sources": [s.model_dump() for s in sources],
        }

    @staticmethod
    def _build_fallback_response(
        query: str, failed: list[AgentResponse]
    ) -> dict[str, Any]:
        error_msgs = [
            f"{r.agent.value}: {r.error}" for r in failed if r.error
        ]
        return {
            "response": (
                f"I'm sorry, I couldn't process your query: \"{query}\"\n\n"
                f"The following agents encountered errors:\n"
                + "\n".join(f"- {e}" for e in error_msgs)
                + "\n\nPlease try again or rephrase your question."
            ),
            "agents_used": [],
            "sources": [],
            "errors": error_msgs,
        }

    @staticmethod
    def _format_result(result: dict[str, Any], depth: int = 0) -> str:
        lines: list[str] = []
        indent = "  " * depth

        for key, value in result.items():
            if key in ("source", "body_source", "embedding"):
                continue  # Skip raw source and embeddings
            if isinstance(value, dict):
                lines.append(f"{indent}- **{key}**:")
                lines.append(ResponseSynthesizer._format_result(value, depth + 1))
            elif isinstance(value, list):
                if len(value) == 0:
                    lines.append(f"{indent}- **{key}**: (empty)")
                elif len(value) <= 5:
                    items = ", ".join(str(v)[:100] for v in value[:5])
                    lines.append(f"{indent}- **{key}**: {items}")
                else:
                    lines.append(f"{indent}- **{key}**: ({len(value)} items)")
                    for item in value[:3]:
                        lines.append(f"{indent}  - {str(item)[:150]}")
                    lines.append(f"{indent}  - ... and {len(value) - 3} more")
            else:
                val_str = str(value)[:300]
                lines.append(f"{indent}- **{key}**: {val_str}")

        return "\n".join(lines)

    @staticmethod
    def _extract_sources(
        result: dict[str, Any], sources: list[SourceReference]
    ) -> None:
        # Look for file_path references
        if "file_path" in result and result["file_path"]:
            sources.append(
                SourceReference(
                    file_path=result["file_path"],
                    entity_name=result.get("name"),
                    start_line=result.get("start_line"),
                    end_line=result.get("end_line"),
                )
            )

        # Look for entities with file paths
        for key in ("entities", "dependencies", "dependents", "results"):
            if key in result and isinstance(result[key], list):
                for item in result[key][:5]:
                    if isinstance(item, dict) and "file_path" in item:
                        sources.append(
                            SourceReference(
                                file_path=item["file_path"],
                                entity_name=item.get("name"),
                                start_line=item.get("start_line"),
                                end_line=item.get("end_line"),
                            )
                        )
