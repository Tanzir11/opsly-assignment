
from __future__ import annotations

from typing import Any

from src.core.logging import get_logger
from src.db.neo4j import Neo4jRepository
from src.modules.graph_query.graph_repository import GraphQueryRepository

logger = get_logger(__name__)


class GraphQueryTools:

    def __init__(self, neo4j_repo: Neo4jRepository) -> None:
        self.graph = GraphQueryRepository(neo4j_repo)
        self.neo4j = neo4j_repo

    async def find_entity(
        self,
        name: str,
        entity_type: str | None = None,
    ) -> dict[str, Any]:
        results = await self.graph.find_entity(name, entity_type)
        return {
            "query": name,
            "entity_type_filter": entity_type,
            "count": len(results),
            "entities": [
                {
                    "name": r["properties"].get("name", ""),
                    "entity_id": r["properties"].get("entity_id", ""),
                    "type": r["labels"][0] if r["labels"] else "Unknown",
                    "file_path": r["properties"].get("file_path", ""),
                    "signature": r["properties"].get("signature", ""),
                    "doc": (r["properties"].get("doc", "") or "")[:300],
                    "start_line": r["properties"].get("start_line"),
                    "end_line": r["properties"].get("end_line"),
                }
                for r in results
            ],
        }

    async def get_dependencies(
        self,
        entity_id: str,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        results = await self.graph.get_dependencies(entity_id, max_depth)
        return {
            "entity_id": entity_id,
            "max_depth": max_depth,
            "count": len(results),
            "dependencies": [
                {
                    "name": r["properties"].get("name", ""),
                    "entity_id": r["properties"].get("entity_id", ""),
                    "type": r["labels"][0] if r["labels"] else "Unknown",
                    "file_path": r["properties"].get("file_path", ""),
                }
                for r in results
            ],
        }

    async def get_dependents(
        self,
        entity_id: str,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        results = await self.graph.get_dependents(entity_id, max_depth)
        return {
            "entity_id": entity_id,
            "max_depth": max_depth,
            "count": len(results),
            "dependents": [
                {
                    "name": r["properties"].get("name", ""),
                    "entity_id": r["properties"].get("entity_id", ""),
                    "type": r["labels"][0] if r["labels"] else "Unknown",
                    "file_path": r["properties"].get("file_path", ""),
                }
                for r in results
            ],
        }

    async def trace_imports(self, module_name: str) -> dict[str, Any]:
        results = await self.graph.trace_imports(module_name)
        return {
            "module": module_name,
            "count": len(results),
            "import_chain": results,
        }

    async def find_related(
        self,
        entity_id: str,
        relationship_type: str,
    ) -> dict[str, Any]:
        results = await self.graph.find_related(entity_id, relationship_type)
        return {
            "entity_id": entity_id,
            "relationship_type": relationship_type,
            "count": len(results),
            "related": results,
        }

    async def execute_query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        results = await self.graph.execute_query(cypher, params)
        return {
            "query": cypher,
            "count": len(results),
            "results": results,
        }

    async def get_statistics(self) -> dict[str, Any]:
        return await self.neo4j.get_statistics()
