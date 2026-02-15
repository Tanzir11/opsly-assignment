
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.core.exceptions import LLMProviderError, LLMRateLimitError
from src.core.logging import get_logger
from src.core.llm_client import LLMClient

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelCard:

    provider: str
    model: str

    @property
    def litellm_model(self) -> str:
        return f"{self.provider}/{self.model}"


MODEL_CARDS: dict[str, ModelCard] = {
    # OpenAI
    "openai_gpt_52_codex": ModelCard(provider="openai", model="gpt-5.2-codex"),
    "openai_gpt_52": ModelCard(provider="openai", model="gpt-5.2"),
    # Anthropic
    "anthropic_sonnet_45": ModelCard(
        provider="anthropic", model="claude-sonnet-4-5-20250929"
    ),
    "anthropic_haiku_45": ModelCard(
        provider="anthropic", model="claude-haiku-4-5-20251001"
    ),
    # Gemini
    "gemini_3_flash": ModelCard(provider="gemini", model="gemini-3-flash-preview"),
    "gemini_25_flash_lite": ModelCard(
        provider="gemini", model="gemini-2.5-flash-lite"
    ),
}


def _model(card_key: str) -> str:
    return MODEL_CARDS[card_key].litellm_model


@dataclass
class LLMRoute:

    model: str  # litellm model string, e.g. "openai/gpt-4o-mini", "anthropic/claude-3-haiku"
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout: float = 30.0


@dataclass
class TaskRouteConfig:

    primary: LLMRoute
    fallbacks: list[LLMRoute] = field(default_factory=list)


# ─── Default routing table ──────────────────────────────────────────────────

DEFAULT_ROUTES: dict[str, TaskRouteConfig] = {
    "analyze_query": TaskRouteConfig(
        primary=LLMRoute(
            model=_model("gemini_25_flash_lite"),
            temperature=0.2,
            max_tokens=512,
            timeout=12,
        ),
        fallbacks=[
            LLMRoute(
                model=_model("anthropic_haiku_45"), temperature=0.2, max_tokens=512
            ),
            LLMRoute(model=_model("openai_gpt_52"), temperature=0.2, max_tokens=512),
        ],
    ),
    "code_explanation": TaskRouteConfig(
        primary=LLMRoute(
            model=_model("openai_gpt_52_codex"),
            temperature=0.3,
            max_tokens=4096,
            timeout=60,
        ),
        fallbacks=[
            LLMRoute(
                model=_model("anthropic_sonnet_45"),
                temperature=0.3,
                max_tokens=4096,
            ),
            LLMRoute(model=_model("gemini_3_flash"), temperature=1.0, max_tokens=4096),
        ],
    ),
    "response_synthesis": TaskRouteConfig(
        primary=LLMRoute(
            model=_model("openai_gpt_52"),
            temperature=0.3,
            max_tokens=4096,
            timeout=45,
        ),
        fallbacks=[
            LLMRoute(
                model=_model("anthropic_sonnet_45"),
                temperature=0.3,
                max_tokens=4096,
            ),
            LLMRoute(model=_model("gemini_3_flash"), temperature=1.0, max_tokens=4096),
        ],
    ),
    "pattern_analysis": TaskRouteConfig(
        primary=LLMRoute(
            model=_model("anthropic_sonnet_45"),
            temperature=0.2,
            max_tokens=4096,
            timeout=45,
        ),
        fallbacks=[
            LLMRoute(
                model=_model("openai_gpt_52_codex"),
                temperature=0.2,
                max_tokens=4096,
            ),
            LLMRoute(model=_model("gemini_3_flash"), temperature=1.0, max_tokens=4096),
        ],
    ),
    "fallback": TaskRouteConfig(
        primary=LLMRoute(
            model=_model("anthropic_haiku_45"), temperature=0.3, max_tokens=2048
        ),
        fallbacks=[
            LLMRoute(model=_model("openai_gpt_52"), temperature=0.3, max_tokens=2048),
            LLMRoute(
                model=_model("gemini_25_flash_lite"),
                temperature=0.3,
                max_tokens=2048,
            ),
        ],
    ),
}


class LLMRouter:

    def __init__(
        self,
        routes: dict[str, TaskRouteConfig] | None = None,
        api_keys: dict[str, list[str]] | None = None,
    ) -> None:
        self.routes = routes or DEFAULT_ROUTES
        self.client = LLMClient(api_keys=api_keys or {})

    def _extract_provider(self, model_string: str) -> str:
        if "/" in model_string:
            return model_string.split("/")[0]
        return "openai"  # Default provider

    async def generate(
        self,
        messages: list[dict[str, str]],
        task: str = "fallback",
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        route_config = self.routes.get(task, self.routes.get("fallback"))
        if not route_config:
            raise LLMProviderError("No route configured for task and no fallback available")

        candidates = [route_config.primary, *route_config.fallbacks]
        last_error: Exception | None = None

        for route in candidates:
            try:
                return await self._call_provider(route, messages, tools, **kwargs)
            except LLMRateLimitError:
                logger.warning("Rate limited on %s, trying next fallback", route.model)
                last_error = LLMRateLimitError(f"Rate limited: {route.model}")
                continue
            except Exception as exc:
                logger.warning(
                    "LLM call failed for %s: %s, trying next fallback",
                    route.model,
                    str(exc),
                )
                last_error = exc
                continue

        raise LLMProviderError(
            f"All LLM providers failed for task '{task}': {last_error}"
        )

    async def _call_provider(
        self,
        route: LLMRoute,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        provider = self._extract_provider(route.model)

        start = time.monotonic()
        try:
            response = await self.client.complete(
                model=route.model,
                messages=messages,
                temperature=route.temperature,
                max_tokens=route.max_tokens,
                timeout=route.timeout,
                tools=tools,
                **kwargs,
            )
        except Exception as exc:
            error_msg = str(exc).lower()
            if "rate" in error_msg or "429" in error_msg:
                raise LLMRateLimitError(str(exc)) from exc
            raise LLMProviderError(f"Provider {provider} error: {exc}") from exc

        latency_ms = (time.monotonic() - start) * 1000

        content = ""
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content or ""

        return {
            "content": content,
            "model": route.model,
            "provider": provider,
            "latency_ms": round(latency_ms, 2),
            "usage": dict(response.usage) if response.usage else {},
            "raw_message": response.choices[0].message if response.choices else None,
        }
