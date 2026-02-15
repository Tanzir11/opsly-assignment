
from __future__ import annotations

from src.core.logging import get_logger
from src.db.mongodb import MongoDBConnection
from src.db.neo4j import Neo4jConnection

logger = get_logger(__name__)


async def bootstrap_neo4j_schema(conn: Neo4jConnection) -> None:
    constraints = [
        # Unique constraints
        "CREATE CONSTRAINT IF NOT EXISTS FOR (f:File) REQUIRE (f.repo, f.path) IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Class) REQUIRE c.entity_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (fn:Function) REQUIRE fn.entity_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Method) REQUIRE m.entity_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (ij:IndexJob) REQUIRE ij.job_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (mod:Module) REQUIRE mod.entity_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Parameter) REQUIRE p.entity_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Decorator) REQUIRE d.entity_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (imp:Import) REQUIRE imp.entity_id IS UNIQUE",
    ]

    indexes = [
        # Secondary indexes for lookups
        "CREATE INDEX IF NOT EXISTS FOR (c:Class) ON (c.name)",
        "CREATE INDEX IF NOT EXISTS FOR (fn:Function) ON (fn.name)",
        "CREATE INDEX IF NOT EXISTS FOR (m:Method) ON (m.name)",
        "CREATE INDEX IF NOT EXISTS FOR (ch:Chunk) ON (ch.file_path)",
        "CREATE INDEX IF NOT EXISTS FOR (imp:Import) ON (imp.module)",
        "CREATE INDEX IF NOT EXISTS FOR (f:File) ON (f.repo)",
    ]

    async with conn.session() as session:
        for stmt in constraints:
            try:
                await session.run(stmt)
                logger.debug("Neo4j schema: %s", stmt[:80])
            except Exception as exc:
                logger.warning("Neo4j constraint skipped: %s - %s", stmt[:60], exc)

        for stmt in indexes:
            try:
                await session.run(stmt)
                logger.debug("Neo4j schema: %s", stmt[:80])
            except Exception as exc:
                logger.warning("Neo4j index skipped: %s - %s", stmt[:60], exc)

    logger.info("Neo4j schema bootstrap complete")


async def bootstrap_mongodb_indexes(conn: MongoDBConnection) -> None:
    db = conn.db

    # Sessions
    await db.sessions.create_index("session_id", unique=True)

    # Messages
    await db.messages.create_index([("session_id", 1), ("timestamp", 1)])
    await db.messages.create_index("message_id", unique=True)

    # Routing decisions
    await db.routing_decisions.create_index([("session_id", 1), ("timestamp", -1)])

    # Agent cache - unique key + TTL
    await db.agent_cache.create_index("cache_key", unique=True)
    await db.agent_cache.create_index("ttl_expires_at", expireAfterSeconds=0)

    # Index jobs
    await db.index_jobs.create_index("job_id", unique=True)

    logger.info("MongoDB index bootstrap complete")


async def bootstrap_all(
    neo4j_conn: Neo4jConnection,
    mongo_conn: MongoDBConnection,
) -> None:
    await bootstrap_neo4j_schema(neo4j_conn)
    await bootstrap_mongodb_indexes(mongo_conn)
    logger.info("All database schemas bootstrapped")
