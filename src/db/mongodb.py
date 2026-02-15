
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from src.core.exceptions import MongoDBError
from src.core.logging import get_logger
from src.core.models import ConversationMessage, RoutingDecision

logger = get_logger(__name__)


class MongoDBConnection:

    def __init__(self, uri: str, db_name: str = "opsly") -> None:
        self.uri = uri
        self.db_name = db_name
        self._client: AsyncIOMotorClient | None = None  # type: ignore[type-arg]
        self._db: AsyncIOMotorDatabase | None = None  # type: ignore[type-arg]

    async def connect(self) -> None:
        try:
            self._client = AsyncIOMotorClient(self.uri)
            self._db = self._client[self.db_name]
            # Ping to verify
            await self._client.admin.command("ping")
            logger.info("Connected to MongoDB at %s, db=%s", self.uri, self.db_name)
        except Exception as exc:
            raise MongoDBError(f"Failed to connect to MongoDB: {exc}") from exc

    async def close(self) -> None:
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")

    @property
    def db(self) -> AsyncIOMotorDatabase:  # type: ignore[type-arg]
        if self._db is None:
            raise MongoDBError("MongoDB not initialized; call connect() first")
        return self._db

    async def health_check(self) -> bool:
        try:
            if self._client:
                await self._client.admin.command("ping")
                return True
            return False
        except Exception:
            return False


class MongoDBRepository:

    def __init__(self, connection: MongoDBConnection) -> None:
        self.conn = connection

    # ─── Sessions ───────────────────────────────────────────────────────

    async def create_session(
        self, session_id: str, user_meta: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        doc = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc),
            "user_meta": user_meta or {},
        }
        await self.conn.db.sessions.update_one(
            {"session_id": session_id}, {"$setOnInsert": doc}, upsert=True
        )
        return doc

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        return await self.conn.db.sessions.find_one(
            {"session_id": session_id}, {"_id": 0}
        )

    # ─── Messages ───────────────────────────────────────────────────────

    async def store_message(self, message: ConversationMessage) -> None:
        doc = message.model_dump()
        doc["timestamp"] = message.timestamp
        await self.conn.db.messages.insert_one(doc)

    async def get_messages(
        self,
        session_id: str,
        limit: int = 50,
        before: datetime | None = None,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"session_id": session_id}
        if before:
            query["timestamp"] = {"$lt": before}

        cursor = (
            self.conn.db.messages.find(query, {"_id": 0})
            .sort("timestamp", -1)
            .limit(limit)
        )
        messages = await cursor.to_list(length=limit)
        messages.reverse()  # Return in chronological order
        return messages

    async def get_recent_context(
        self, session_id: str, num_turns: int = 5
    ) -> list[dict[str, Any]]:
        return await self.get_messages(session_id, limit=num_turns * 2)

    # ─── Routing Decisions ──────────────────────────────────────────────

    async def store_routing_decision(self, decision: RoutingDecision) -> None:
        await self.conn.db.routing_decisions.insert_one(decision.model_dump())

    async def get_routing_history(
        self, session_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        cursor = (
            self.conn.db.routing_decisions.find(
                {"session_id": session_id}, {"_id": 0}
            )
            .sort("timestamp", -1)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    # ─── Agent Cache ────────────────────────────────────────────────────

    @staticmethod
    def _compute_cache_key(agent: str, input_data: dict[str, Any]) -> str:
        normalized = json.dumps(input_data, sort_keys=True, default=str)
        return hashlib.sha256(f"{agent}:{normalized}".encode()).hexdigest()[:32]

    async def get_agent_cache(
        self, agent: str, input_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        cache_key = self._compute_cache_key(agent, input_data)
        doc = await self.conn.db.agent_cache.find_one(
            {
                "cache_key": cache_key,
                "ttl_expires_at": {"$gt": datetime.now(timezone.utc)},
            },
            {"_id": 0},
        )
        if doc:
            logger.info("Cache hit for agent=%s key=%s", agent, cache_key[:8])
        return doc

    async def upsert_agent_cache(
        self,
        agent: str,
        input_data: dict[str, Any],
        output: Any,
        ttl_seconds: int = 3600,
    ) -> None:
        cache_key = self._compute_cache_key(agent, input_data)
        doc = {
            "cache_key": cache_key,
            "agent": agent,
            "input_hash": json.dumps(input_data, sort_keys=True, default=str)[:500],
            "output": output,
            "ttl_expires_at": datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
            "updated_at": datetime.now(timezone.utc),
        }
        await self.conn.db.agent_cache.update_one(
            {"cache_key": cache_key}, {"$set": doc}, upsert=True
        )

    # ─── Index Jobs ─────────────────────────────────────────────────────

    async def create_index_job(self, job_id: str, repo: str) -> dict[str, Any]:
        doc = {
            "job_id": job_id,
            "status": "pending",
            "repo": repo,
            "progress": 0.0,
            "files_processed": 0,
            "files_total": 0,
            "entities_extracted": 0,
            "errors": [],
            "started_at": datetime.now(timezone.utc),
            "finished_at": None,
        }
        await self.conn.db.index_jobs.insert_one(doc)
        return doc

    async def update_index_job(
        self, job_id: str, updates: dict[str, Any]
    ) -> None:
        await self.conn.db.index_jobs.update_one(
            {"job_id": job_id}, {"$set": updates}
        )

    async def get_index_job(self, job_id: str) -> dict[str, Any] | None:
        return await self.conn.db.index_jobs.find_one(
            {"job_id": job_id}, {"_id": 0}
        )
