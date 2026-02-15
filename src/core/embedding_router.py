
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.core.exceptions import EmbeddingError
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingRoute:

    provider: str  # "local" | "openai" | "gemini"
    model: str  # e.g. "all-MiniLM-L6-v2", "text-embedding-3-small"
    dimension: int = 384
    batch_size: int = 64


@dataclass
class EmbeddingTaskConfig:

    primary: EmbeddingRoute
    fallbacks: list[EmbeddingRoute] = field(default_factory=list)


# ─── Default routing table ──────────────────────────────────────────────────

def _build_default_routes(default_provider: str) -> dict[str, EmbeddingTaskConfig]:
    provider = default_provider.strip().lower()

    if provider == "openai":
        openai_primary = EmbeddingRoute(
            provider="openai",
            model="text-embedding-3-small",
            dimension=1536,
            batch_size=100,
        )
        return {
            "code_indexing": EmbeddingTaskConfig(primary=openai_primary),
            "query_embedding": EmbeddingTaskConfig(primary=openai_primary),
            "conversation_memory": EmbeddingTaskConfig(primary=openai_primary),
        }

    local_primary = EmbeddingRoute(
        provider="local",
        model="all-MiniLM-L6-v2",
        dimension=384,
        batch_size=64,
    )
    return {
        "code_indexing": EmbeddingTaskConfig(primary=local_primary),
        "query_embedding": EmbeddingTaskConfig(
            primary=EmbeddingRoute(
                provider="local",
                model="all-MiniLM-L6-v2",
                dimension=384,
                batch_size=16,
            )
        ),
        "conversation_memory": EmbeddingTaskConfig(
            primary=EmbeddingRoute(
                provider="local",
                model="all-MiniLM-L6-v2",
                dimension=384,
                batch_size=32,
            )
        ),
    }


class EmbeddingRouter:

    def __init__(
        self,
        routes: dict[str, EmbeddingTaskConfig] | None = None,
        api_keys: dict[str, str] | None = None,
        default_provider: str = "local",
    ) -> None:
        self.routes = routes or _build_default_routes(default_provider)
        self.api_keys = api_keys or {}
        self._local_models: dict[str, Any] = {}  # Cache loaded local models

    def _get_local_model(self, model_name: str) -> Any:
        if model_name not in self._local_models:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading local embedding model: %s", model_name)
                self._local_models[model_name] = SentenceTransformer(model_name)
            except ImportError:
                raise EmbeddingError(
                    "sentence-transformers not installed; "
                    "run: pip install sentence-transformers"
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to load local model {model_name}: {exc}"
                ) from exc
        return self._local_models[model_name]

    async def embed(
        self,
        texts: list[str],
        task: str = "query_embedding",
    ) -> dict[str, Any]:
        route_config = self.routes.get(task)
        if not route_config:
            route_config = self.routes.get("query_embedding")
        if not route_config:
            raise EmbeddingError(f"No embedding route configured for task '{task}'")

        candidates = [route_config.primary, *route_config.fallbacks]
        last_error: Exception | None = None

        for route in candidates:
            try:
                return await self._generate_embeddings(route, texts)
            except Exception as exc:
                logger.warning(
                    "Embedding failed for %s/%s: %s",
                    route.provider,
                    route.model,
                    str(exc),
                )
                last_error = exc
                continue

        raise EmbeddingError(
            f"All embedding providers failed for task '{task}': {last_error}"
        )

    async def _generate_embeddings(
        self,
        route: EmbeddingRoute,
        texts: list[str],
    ) -> dict[str, Any]:
        start = time.monotonic()

        if route.provider == "local":
            embeddings = await self._local_embed(route, texts)
        elif route.provider == "openai":
            embeddings = await self._openai_embed(route, texts)
        else:
            raise EmbeddingError(f"Unknown embedding provider: {route.provider}")

        latency_ms = (time.monotonic() - start) * 1000

        return {
            "embeddings": embeddings,
            "model": route.model,
            "provider": route.provider,
            "dimension": route.dimension,
            "latency_ms": round(latency_ms, 2),
        }

    async def _local_embed(
        self, route: EmbeddingRoute, texts: list[str]
    ) -> list[list[float]]:
        model = self._get_local_model(route.model)
        # sentence-transformers encode is sync; run in thread for async compat
        import asyncio

        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, batch_size=route.batch_size, show_progress_bar=False),
        )
        if isinstance(vectors, np.ndarray):
            return vectors.tolist()
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]

    async def _openai_embed(
        self, route: EmbeddingRoute, texts: list[str]
    ) -> list[list[float]]:
        from litellm import aembedding

        api_key = self.api_keys.get("openai")
        kwargs: dict[str, Any] = {
            "model": f"openai/{route.model}",
            "input": texts,
        }
        if api_key:
            kwargs["api_key"] = api_key

        response = await aembedding(**kwargs)
        return [item["embedding"] for item in response.data]

    def get_dimension(self, task: str = "query_embedding") -> int:
        route_config = self.routes.get(task)
        if route_config:
            return route_config.primary.dimension
        return 384  # Default
