
from __future__ import annotations

from typing import Any

from src.core.embedding_router import EmbeddingRouter
from src.core.logging import get_logger
from src.db.milvus import MilvusRepository
from src.db.mongodb import MongoDBRepository
from src.modules.retrieval_memory.retrieval import RetrievalService

logger = get_logger(__name__)


class RetrievalMemoryTools:

    def __init__(
        self,
        mongo_repo: MongoDBRepository,
        milvus_repo: MilvusRepository | None,
        embedding_api_keys: dict[str, str] | None = None,
        embedding_default_provider: str = "local",
    ) -> None:
        self.mongo = mongo_repo
        self.milvus = milvus_repo
        self.embedding_router = EmbeddingRouter(
            api_keys=embedding_api_keys or {},
            default_provider=embedding_default_provider,
        )
        self.vector_search_enabled = milvus_repo is not None
        self.retrieval = RetrievalService(
            milvus_repo=milvus_repo,
            mongo_repo=mongo_repo,
            embedding_router=self.embedding_router,
        )

    async def store_conversation_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        correlation_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self.retrieval.store_conversation_turn(
            session_id=session_id,
            role=role,
            content=content,
            correlation_id=correlation_id,
            metadata=metadata,
        )

    async def get_relevant_memory(
        self,
        session_id: str,
        query: str | None = None,
        num_turns: int = 5,
    ) -> dict[str, Any]:
        if query:
            return await self.retrieval.hybrid_search(
                query=query,
                session_id=session_id,
                top_k=5,
            )
        else:
            context = await self.retrieval.get_conversation_context(
                session_id, num_turns
            )
            return {
                "session_id": session_id,
                "conversation_context": context,
                "context_turns": len(context),
            }

    async def semantic_search_chunks(
        self,
        query: str,
        top_k: int = 10,
        repo: str | None = None,
        entity_type: str | None = None,
    ) -> dict[str, Any]:
        filters: dict[str, str] = {}
        if repo:
            filters["repo"] = repo
        if entity_type:
            filters["entity_type"] = entity_type

        results = await self.retrieval.semantic_search(
            query=query,
            top_k=top_k,
            filters=filters if filters else None,
        )

        return {
            "query": query,
            "count": len(results),
            "results": results,
            "vector_search_enabled": self.vector_search_enabled,
        }

    async def upsert_agent_cache(
        self,
        agent: str,
        input_data: dict[str, Any],
        output: Any,
        ttl_seconds: int = 3600,
    ) -> dict[str, Any]:
        await self.retrieval.cache_response(agent, input_data, output, ttl_seconds)
        return {"cached": True, "agent": agent, "ttl_seconds": ttl_seconds}

    async def get_agent_cache(
        self,
        agent: str,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        cached = await self.retrieval.get_cached_response(agent, input_data)
        if cached is not None:
            return {"hit": True, "output": cached}
        return {"hit": False, "output": None}
