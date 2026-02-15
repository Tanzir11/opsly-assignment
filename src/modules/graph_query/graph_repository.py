
from __future__ import annotations

from typing import Any

from src.core.exceptions import Neo4jError, QuerySafetyError
from src.core.logging import get_logger
from src.db.neo4j import Neo4jRepository

logger = get_logger(__name__)

# ─── Allowed Cypher keywords for safe query execution ───────────────────────

SAFE_KEYWORDS = {"MATCH", "RETURN", "WITH", "WHERE", "ORDER", "BY", "LIMIT", "SKIP",
                 "OPTIONAL", "UNWIND", "DISTINCT", "AS", "AND", "OR", "NOT", "IN",
                 "IS", "NULL", "TRUE", "FALSE", "ASC", "DESC", "CASE", "WHEN",
                 "THEN", "ELSE", "END", "COUNT", "SUM", "AVG", "COLLECT", "SIZE",
                 "EXISTS", "NONE", "ANY", "ALL", "SINGLE", "STARTS", "ENDS",
                 "CONTAINS", "CALL", "YIELD", "UNION"}

FORBIDDEN_KEYWORDS = {"CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE",
                      "DROP", "FOREACH", "LOAD", "CSV"}


class GraphQueryRepository:

    def __init__(self, neo4j_repo: Neo4jRepository) -> None:
        self.repo = neo4j_repo

    async def find_entity(
        self,
        name: str,
        entity_type: str | None = None,
    ) -> list[dict[str, Any]]:
        from src.core.models import EntityType

        et = None
        if entity_type:
            try:
                et = EntityType(entity_type)
            except ValueError:
                pass

        return await self.repo.find_entity_by_name(name, et)

    async def get_dependencies(
        self,
        entity_id: str,
        max_depth: int = 3,
    ) -> list[dict[str, Any]]:
        max_depth = min(max_depth, 5)
        return await self.repo.get_dependencies(entity_id, max_depth)

    async def get_dependents(
        self,
        entity_id: str,
        max_depth: int = 3,
    ) -> list[dict[str, Any]]:
        max_depth = min(max_depth, 5)
        return await self.repo.get_dependents(entity_id, max_depth)

    async def trace_imports(self, module_name: str) -> list[dict[str, Any]]:
        return await self.repo.trace_imports(module_name)

    async def find_related(
        self,
        entity_id: str,
        relationship_type: str,
    ) -> list[dict[str, Any]]:
        # Validate relationship type
        allowed_rels = {
            "CONTAINS", "IMPORTS", "INHERITS_FROM", "CALLS",
            "DECORATED_BY", "HAS_PARAMETER", "DOCUMENTED_BY", "DEPENDS_ON",
            "HAS_CHUNK",
        }
        if relationship_type.upper() not in allowed_rels:
            raise QuerySafetyError(
                f"Relationship type '{relationship_type}' not in allowed set"
            )

        return await self.repo.find_related(entity_id, relationship_type.upper())

    async def execute_query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self._validate_query_safety(cypher)
        return await self.repo.execute_safe_query(cypher, params)

    @staticmethod
    def _validate_query_safety(cypher: str) -> None:
        upper_words = set(cypher.upper().split())
        violations = upper_words & FORBIDDEN_KEYWORDS
        if violations:
            raise QuerySafetyError(
                f"Query contains forbidden operations: {violations}"
            )

    async def get_class_hierarchy(self, class_name: str) -> list[dict[str, Any]]:
        query = """
        MATCH path = (child:Class)-[:INHERITS_FROM*1..5]->(ancestor:Class)
        WHERE toLower(child.name) CONTAINS toLower($name)
           OR toLower(ancestor.name) CONTAINS toLower($name)
        RETURN child, ancestor, length(path) AS depth
        ORDER BY depth
        LIMIT 50
        """
        return await self.repo.execute_safe_query(query, {"name": class_name})

    async def get_module_structure(self, file_path: str) -> list[dict[str, Any]]:
        query = """
        MATCH (f:File {path: $path})-[:CONTAINS*1..2]->(entity)
        RETURN entity, labels(entity) AS labels
        ORDER BY entity.start_line
        """
        return await self.repo.execute_safe_query(query, {"path": file_path})

    async def find_callers(self, function_name: str) -> list[dict[str, Any]]:
        query = """
        MATCH (caller)-[:CALLS]->(callee)
        WHERE toLower(callee.name) CONTAINS toLower($name)
        RETURN caller, labels(caller) AS caller_labels,
               callee, labels(callee) AS callee_labels
        LIMIT 30
        """
        return await self.repo.execute_safe_query(query, {"name": function_name})
