
from __future__ import annotations

from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

from src.core.exceptions import Neo4jError
from src.core.logging import get_logger
from src.core.models import CodeEntity, CodeRelationship, EntityType

logger = get_logger(__name__)


class Neo4jConnection:

    def __init__(self, uri: str, user: str, password: str) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        try:
            if self.user and not self.password:
                raise Neo4jError(
                    "NEO4J_PASSWORD is empty. Set NEO4J_PASSWORD in .env/.env.develop."
                )
            self._driver = AsyncGraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            await self._driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", self.uri)
        except Neo4jError:
            raise
        except Exception as exc:
            raise Neo4jError(f"Failed to connect to Neo4j at {self.uri}: {exc}") from exc

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            logger.info("Neo4j connection closed")

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            raise Neo4jError("Neo4j driver not initialized; call connect() first")
        return self._driver

    def session(self) -> AsyncSession:
        return self.driver.session()

    async def health_check(self) -> bool:
        try:
            await self.driver.verify_connectivity()
            return True
        except Exception:
            return False


class Neo4jRepository:

    def __init__(self, connection: Neo4jConnection) -> None:
        self.conn = connection

    async def create_entity(self, entity: CodeEntity) -> str:
        label = entity.entity_type.value
        query = f"""
        MERGE (n:{label} {{entity_id: $entity_id}})
        SET n.name = $name,
            n.file_path = $file_path,
            n.signature = $signature,
            n.start_line = $start_line,
            n.end_line = $end_line,
            n.doc = $doc
        RETURN n.entity_id AS entity_id
        """
        params = {
            "entity_id": entity.entity_id,
            "name": entity.name,
            "file_path": entity.file_path,
            "signature": entity.signature,
            "start_line": entity.start_line,
            "end_line": entity.end_line,
            "doc": entity.doc,
        }
        try:
            async with self.conn.session() as session:
                result = await session.run(query, params)
                record = await result.single()
                return record["entity_id"] if record else entity.entity_id
        except Exception as exc:
            raise Neo4jError(f"Failed to create entity {entity.name}: {exc}") from exc

    async def create_file_node(
        self, repo: str, path: str, sha: str, language: str = "python"
    ) -> None:
        query = """
        MERGE (f:File {repo: $repo, path: $path})
        SET f.sha = $sha,
            f.language = $language,
            f.last_indexed_at = datetime()
        """
        async with self.conn.session() as session:
            await session.run(query, {"repo": repo, "path": path, "sha": sha, "language": language})

    async def create_relationship(self, rel: CodeRelationship) -> None:
        query = f"""
        MATCH (a {{entity_id: $source_id}})
        MATCH (b {{entity_id: $target_id}})
        MERGE (a)-[r:{rel.relationship_type}]->(b)
        """
        try:
            async with self.conn.session() as session:
                await session.run(
                    query,
                    {"source_id": rel.source_id, "target_id": rel.target_id},
                )
        except Exception as exc:
            raise Neo4jError(
                f"Failed to create relationship {rel.relationship_type}: {exc}"
            ) from exc

    async def create_chunk_node(
        self,
        chunk_id: str,
        entity_id: str,
        file_path: str,
        content: str,
        token_count: int,
        embedding_id: str = "",
        embedding_model: str = "",
    ) -> None:
        query = """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.entity_id = $entity_id,
            c.file_path = $file_path,
            c.content = $content,
            c.token_count = $token_count,
            c.embedding_id = $embedding_id,
            c.embedding_model = $embedding_model
        WITH c
        MATCH (e {entity_id: $entity_id})
        MERGE (e)-[:HAS_CHUNK]->(c)
        """
        async with self.conn.session() as session:
            await session.run(query, {
                "chunk_id": chunk_id,
                "entity_id": entity_id,
                "file_path": file_path,
                "content": content,
                "token_count": token_count,
                "embedding_id": embedding_id,
                "embedding_model": embedding_model,
            })

    async def find_entity_by_name(
        self, name: str, entity_type: EntityType | None = None
    ) -> list[dict[str, Any]]:
        if entity_type:
            query = f"""
            MATCH (n:{entity_type.value})
            WHERE toLower(n.name) CONTAINS toLower($name)
            RETURN n, labels(n) AS labels
            LIMIT 20
            """
        else:
            query = """
            MATCH (n)
            WHERE n.name IS NOT NULL AND toLower(n.name) CONTAINS toLower($name)
            RETURN n, labels(n) AS labels
            LIMIT 20
            """
        async with self.conn.session() as session:
            result = await session.run(query, {"name": name})
            records = await result.data()
            return [{"properties": dict(r["n"]), "labels": r["labels"]} for r in records]

    async def get_dependencies(self, entity_id: str, max_depth: int = 3) -> list[dict[str, Any]]:
        query = """
        MATCH (source {entity_id: $entity_id})-[:DEPENDS_ON|IMPORTS|CALLS*1..$max_depth]->(target)
        RETURN DISTINCT target, labels(target) AS labels
        LIMIT 50
        """
        async with self.conn.session() as session:
            result = await session.run(
                query, {"entity_id": entity_id, "max_depth": max_depth}
            )
            records = await result.data()
            return [{"properties": dict(r["target"]), "labels": r["labels"]} for r in records]

    async def get_dependents(self, entity_id: str, max_depth: int = 3) -> list[dict[str, Any]]:
        query = """
        MATCH (source)-[:DEPENDS_ON|IMPORTS|CALLS*1..$max_depth]->(target {entity_id: $entity_id})
        RETURN DISTINCT source, labels(source) AS labels
        LIMIT 50
        """
        async with self.conn.session() as session:
            result = await session.run(
                query, {"entity_id": entity_id, "max_depth": max_depth}
            )
            records = await result.data()
            return [{"properties": dict(r["source"]), "labels": r["labels"]} for r in records]

    async def trace_imports(self, module_name: str) -> list[dict[str, Any]]:
        query = """
        MATCH path = (imp:Import)-[:IMPORTS*1..5]->(target)
        WHERE toLower(imp.module) CONTAINS toLower($module_name)
           OR toLower(imp.name) CONTAINS toLower($module_name)
        RETURN imp, target, labels(target) AS labels,
               length(path) AS depth
        ORDER BY depth
        LIMIT 50
        """
        async with self.conn.session() as session:
            result = await session.run(query, {"module_name": module_name})
            return await result.data()

    async def find_related(
        self, entity_id: str, relationship_type: str
    ) -> list[dict[str, Any]]:
        query = f"""
        MATCH (source {{entity_id: $entity_id}})-[r:{relationship_type}]->(target)
        RETURN target, labels(target) AS labels, type(r) AS rel_type
        UNION
        MATCH (source)-[r:{relationship_type}]->(target {{entity_id: $entity_id}})
        RETURN source AS target, labels(source) AS labels, type(r) AS rel_type
        """
        async with self.conn.session() as session:
            result = await session.run(query, {"entity_id": entity_id})
            return await result.data()

    async def execute_safe_query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        # Safety: deny write operations
        upper = cypher.upper().strip()
        forbidden = ["CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP", "CALL"]
        for keyword in forbidden:
            if keyword in upper.split():
                raise Neo4jError(f"Write operation '{keyword}' not allowed in safe query mode")

        async with self.conn.session() as session:
            result = await session.run(cypher, params or {})
            return await result.data()

    async def get_statistics(self) -> dict[str, Any]:
        stats: dict[str, Any] = {}

        async with self.conn.session() as session:
            # Node counts by label
            result = await session.run(
                "CALL db.labels() YIELD label "
                "CALL { WITH label MATCH (n) WHERE label IN labels(n) RETURN count(n) AS cnt } "
                "RETURN label, cnt"
            )
            label_counts = await result.data()
            stats["nodes_by_type"] = {r["label"]: r["cnt"] for r in label_counts}
            stats["total_nodes"] = sum(stats["nodes_by_type"].values())

            # Relationship counts by type
            result = await session.run(
                "CALL db.relationshipTypes() YIELD relationshipType "
                "CALL { WITH relationshipType "
                "MATCH ()-[r]->() WHERE type(r) = relationshipType RETURN count(r) AS cnt } "
                "RETURN relationshipType, cnt"
            )
            rel_counts = await result.data()
            stats["relationships_by_type"] = {
                r["relationshipType"]: r["cnt"] for r in rel_counts
            }
            stats["total_relationships"] = sum(stats["relationships_by_type"].values())

            # Indexed files count
            result = await session.run("MATCH (f:File) RETURN count(f) AS cnt")
            record = await result.single()
            stats["indexed_files"] = record["cnt"] if record else 0

            # Indexed repos
            result = await session.run(
                "MATCH (f:File) RETURN DISTINCT f.repo AS repo"
            )
            repos = await result.data()
            stats["indexed_repos"] = [r["repo"] for r in repos if r["repo"]]

        return stats

    async def clear_repo_data(self, repo: str) -> int:
        query = """
        MATCH (n) WHERE n.file_path STARTS WITH $repo OR n.repo = $repo
        DETACH DELETE n
        RETURN count(n) AS deleted
        """
        async with self.conn.session() as session:
            result = await session.run(query, {"repo": repo})
            record = await result.single()
            return record["deleted"] if record else 0
