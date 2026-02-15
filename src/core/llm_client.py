
from __future__ import annotations

from typing import Any

from litellm import acompletion


class LLMClient:

    def __init__(self, api_keys: dict[str, list[str]] | None = None) -> None:
        self.api_keys = api_keys or {}
        self._key_index: dict[str, int] = {}

    def _extract_provider(self, model: str) -> str:
        if "/" in model:
            return model.split("/")[0]
        return "openai"

    def _get_api_key(self, provider: str) -> str | None:
        keys = self.api_keys.get(provider, [])
        if not keys:
            return None
        idx = self._key_index.get(provider, 0)
        key = keys[idx % len(keys)]
        self._key_index[provider] = idx + 1
        return key

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: float = 30.0,
        tools: list[dict[str, Any]] | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        provider = self._extract_provider(model)
        resolved_key = api_key or self._get_api_key(provider)

        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            **kwargs,
        }
        if resolved_key:
            call_kwargs["api_key"] = resolved_key
        if tools:
            call_kwargs["tools"] = tools

        return await acompletion(**call_kwargs)
