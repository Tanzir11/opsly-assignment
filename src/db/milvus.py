
from __future__ import annotations

from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from src.core.exceptions import MilvusError
from src.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_COLLECTION = "code_chunks_embeddings"


class MilvusConnection:

    def __init__(
        self, host: str = "localhost", port: int = 19530, alias: str = "default"
    ) -> None:
        self.host = host
        self.port = port
        self.alias = alias

    async def connect(self) -> None:
        try:
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=str(self.port),
            )
            logger.info("Connected to Milvus at %s:%s", self.host, self.port)
        except Exception as exc:
            raise MilvusError(f"Failed to connect to Milvus: {exc}") from exc

    async def close(self) -> None:
        try:
            connections.disconnect(alias=self.alias)
            logger.info("Milvus connection closed")
        except Exception:
            pass

    async def health_check(self) -> bool:
        try:
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=str(self.port),
            )
            return True
        except Exception:
            return False


class MilvusRepository:

    def __init__(
        self,
        connection: MilvusConnection,
        collection_name: str = DEFAULT_COLLECTION,
        dimension: int = 384,
    ) -> None:
        self.conn = connection
        self.collection_name = collection_name
        self.dimension = dimension
        self._collection: Collection | None = None

    def _get_or_create_collection(self) -> Collection:
        if self._collection is not None:
            return self._collection

        if utility.has_collection(self.collection_name):
            self._collection = Collection(self.collection_name)
            self._collection.load()
            return self._collection

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="repo", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="entity_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dimension,
            ),
        ]
        schema = CollectionSchema(
            fields=fields,
            description="Code chunk embeddings for semantic search",
        )

        self._collection = Collection(
            name=self.collection_name,
            schema=schema,
        )

        # Create IVF_FLAT index on embedding field
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self._collection.create_index("embedding", index_params)
        self._collection.load()

        logger.info(
            "Created Milvus collection '%s' with dimension=%d",
            self.collection_name,
            self.dimension,
        )
        return self._collection

    async def insert_embeddings(
        self,
        chunk_ids: list[str],
        repos: list[str],
        entity_types: list[str],
        file_paths: list[str],
        contents: list[str],
        embeddings: list[list[float]],
    ) -> int:
        try:
            collection = self._get_or_create_collection()

            # Truncate content to max length
            truncated_contents = [c[:8000] for c in contents]

            data = [
                chunk_ids,
                repos,
                entity_types,
                file_paths,
                truncated_contents,
                embeddings,
            ]
            result = collection.insert(data)
            collection.flush()
            logger.info("Inserted %d embeddings into Milvus", len(chunk_ids))
            return result.insert_count
        except Exception as exc:
            raise MilvusError(f"Failed to insert embeddings: {exc}") from exc

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        try:
            collection = self._get_or_create_collection()

            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16},
            }

            # Build filter expression
            expr = ""
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(f'{key} == "{value}"')
                expr = " && ".join(conditions)

            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr or None,
                output_fields=["chunk_id", "repo", "entity_type", "file_path", "content"],
            )

            hits: list[dict[str, Any]] = []
            for hit in results[0]:
                hits.append({
                    "chunk_id": hit.entity.get("chunk_id"),
                    "repo": hit.entity.get("repo"),
                    "entity_type": hit.entity.get("entity_type"),
                    "file_path": hit.entity.get("file_path"),
                    "content": hit.entity.get("content"),
                    "score": hit.distance,
                })

            return hits
        except Exception as exc:
            raise MilvusError(f"Milvus search failed: {exc}") from exc

    async def delete_by_repo(self, repo: str) -> None:
        try:
            collection = self._get_or_create_collection()
            collection.delete(f'repo == "{repo}"')
            collection.flush()
            logger.info("Deleted embeddings for repo: %s", repo)
        except Exception as exc:
            raise MilvusError(f"Failed to delete embeddings: {exc}") from exc

    async def get_collection_stats(self) -> dict[str, Any]:
        try:
            collection = self._get_or_create_collection()
            return {
                "collection_name": self.collection_name,
                "num_entities": collection.num_entities,
                "dimension": self.dimension,
            }
        except Exception as exc:
            return {"error": str(exc)}
