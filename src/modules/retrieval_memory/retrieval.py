
from __future__ import annotations

from typing import Any

from src.core.embedding_router import EmbeddingRouter
from src.core.logging import get_logger
from src.core.models import ConversationMessage
from src.db.milvus import MilvusRepository
from src.db.mongodb import MongoDBRepository

logger = get_logger(__name__)


class RetrievalService:

    def __init__(
        self,
        milvus_repo: MilvusRepository | None,
        mongo_repo: MongoDBRepository,
        embedding_router: EmbeddingRouter,
    ) -> None:
        self.milvus = milvus_repo
        self.mongo = mongo_repo
        self.embedding_router = embedding_router

    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        if self.milvus is None:
            logger.info("Milvus disabled; returning empty semantic results")
            return []

        # Embed the query
        embed_result = await self.embedding_router.embed([query], task="query_embedding")
        query_embedding = embed_result["embeddings"][0]

        # Search Milvus
        results = await self.milvus.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

        return results

    async def get_conversation_context(
        self,
        session_id: str,
        num_turns: int = 5,
    ) -> list[dict[str, Any]]:
        return await self.mongo.get_recent_context(session_id, num_turns)

    async def store_conversation_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        correlation_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Ensure session exists
        await self.mongo.create_session(session_id)

        # Store message
        message = ConversationMessage(
            session_id=session_id,
            role=role,
            content=content,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )
        await self.mongo.store_message(message)

        return {
            "session_id": session_id,
            "message_id": message.message_id,
            "role": role,
            "stored": True,
        }

    async def get_cached_response(
        self,
        agent: str,
        input_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        cached = await self.mongo.get_agent_cache(agent, input_data)
        if cached:
            return cached.get("output")
        return None

    async def cache_response(
        self,
        agent: str,
        input_data: dict[str, Any],
        output: Any,
        ttl_seconds: int = 3600,
    ) -> None:
        await self.mongo.upsert_agent_cache(agent, input_data, output, ttl_seconds)

    async def hybrid_search(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int = 10,
        filters: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        # Semantic search
        semantic_results = await self.semantic_search(query, top_k, filters)
        result["semantic_results"] = semantic_results
        result["semantic_count"] = len(semantic_results)

        # Conversation context
        if session_id:
            context = await self.get_conversation_context(session_id)
            result["conversation_context"] = context
            result["context_turns"] = len(context)
        else:
            result["conversation_context"] = []
            result["context_turns"] = 0

        return result
