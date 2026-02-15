
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.embedding_router import EmbeddingRouter
from src.core.logging import get_logger
from src.core.models import IndexJobStatus
from src.db.milvus import MilvusRepository
from src.db.mongodb import MongoDBRepository
from src.db.neo4j import Neo4jRepository
from src.modules.indexer.ast_parser import PythonASTParser
from src.modules.indexer.entity_extractor import EntityExtractor
from src.modules.indexer.repo_loader import RepoLoader

logger = get_logger(__name__)


class IndexerTools:

    def __init__(
        self,
        neo4j_repo: Neo4jRepository,
        mongo_repo: MongoDBRepository,
        milvus_repo: MilvusRepository | None,
        repo_workspace: str = "/tmp/opsly/repos",
        default_repo_url: str = "https://github.com/tiangolo/fastapi.git",
        embedding_api_keys: dict[str, str] | None = None,
        embedding_default_provider: str = "local",
    ) -> None:
        self.neo4j = neo4j_repo
        self.mongo = mongo_repo
        self.milvus = milvus_repo
        self.parser = PythonASTParser()
        self.loader = RepoLoader(repo_workspace)
        self.embedding_router = EmbeddingRouter(
            api_keys=embedding_api_keys or {},
            default_provider=embedding_default_provider,
        )
        self.default_repo_url = default_repo_url

    async def index_repository(
        self,
        repo_url: str | None = None,
        incremental: bool = False,
    ) -> dict[str, Any]:
        repo_url = repo_url or self.default_repo_url
        job_id = uuid.uuid4().hex[:12]

        # Create job record
        await self.mongo.create_index_job(job_id, repo_url)

        # Run indexing in background
        asyncio.create_task(
            self._run_indexing(job_id, repo_url, incremental)
        )

        return {"job_id": job_id, "status": "pending", "repo": repo_url}

    async def _run_indexing(
        self, job_id: str, repo_url: str, incremental: bool
    ) -> None:
        try:
            await self.mongo.update_index_job(job_id, {
                "status": "running",
                "started_at": datetime.now(timezone.utc),
            })

            # Clone or pull repository
            repo_path = await self.loader.clone_or_pull(repo_url)
            repo_name = repo_path.name

            # Discover files
            if incremental:
                python_files = self.loader.get_changed_files(repo_path)
            else:
                python_files = self.loader.discover_python_files(repo_path)

            total_files = len(python_files)
            await self.mongo.update_index_job(job_id, {"files_total": total_files})

            extractor = EntityExtractor(repo_name=repo_name)
            total_entities = 0
            errors: list[str] = []

            for i, py_file in enumerate(python_files):
                try:
                    rel_path = self.loader.get_relative_path(py_file, repo_path)
                    file_sha = self.loader.get_file_sha(py_file)
                    source = py_file.read_text(encoding="utf-8", errors="replace")

                    # Parse AST
                    parsed = self.parser.parse_file(str(py_file), source)
                    if parsed.errors:
                        errors.extend(parsed.errors)
                        continue

                    # Extract entities
                    extraction = extractor.extract(parsed)

                    # Store file node
                    await self.neo4j.create_file_node(
                        repo=repo_name, path=rel_path, sha=file_sha
                    )

                    # Store entities in Neo4j
                    for entity in extraction.entities:
                        await self.neo4j.create_entity(entity)
                        total_entities += 1

                    # Store relationships
                    for rel in extraction.relationships:
                        try:
                            await self.neo4j.create_relationship(rel)
                        except Exception as rel_exc:
                            logger.debug("Relationship creation skipped: %s", rel_exc)

                    # Generate and store embeddings for chunks
                    if extraction.chunks:
                        await self._embed_and_store_chunks(extraction.chunks)

                    # Store chunk nodes in Neo4j
                    for chunk in extraction.chunks:
                        await self.neo4j.create_chunk_node(
                            chunk_id=chunk["chunk_id"],
                            entity_id=chunk["entity_id"],
                            file_path=chunk["file_path"],
                            content=chunk["content"][:2000],
                            token_count=chunk["token_count"],
                        )

                except Exception as file_exc:
                    errors.append(f"{py_file}: {file_exc}")
                    logger.warning("Failed to index %s: %s", py_file, file_exc)

                # Update progress
                progress = (i + 1) / total_files if total_files > 0 else 1.0
                await self.mongo.update_index_job(job_id, {
                    "files_processed": i + 1,
                    "entities_extracted": total_entities,
                    "progress": progress,
                })

            # Mark complete
            await self.mongo.update_index_job(job_id, {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc),
                "entities_extracted": total_entities,
                "errors": errors[:50],  # Limit stored errors
            })
            logger.info(
                "Indexing job %s completed: %d files, %d entities, %d errors",
                job_id, total_files, total_entities, len(errors),
            )

        except Exception as exc:
            logger.error("Indexing job %s failed: %s", job_id, exc, exc_info=True)
            await self.mongo.update_index_job(job_id, {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc),
                "errors": [str(exc)],
            })

    async def _embed_and_store_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if self.milvus is None:
            logger.debug("Skipping embedding storage because Milvus is disabled")
            return

        contents = [c["content"] for c in chunks]

        try:
            embed_result = await self.embedding_router.embed(
                contents, task="code_indexing"
            )
            embeddings = embed_result["embeddings"]

            await self.milvus.insert_embeddings(
                chunk_ids=[c["chunk_id"] for c in chunks],
                repos=[c["repo"] for c in chunks],
                entity_types=[c["entity_type"] for c in chunks],
                file_paths=[c["file_path"] for c in chunks],
                contents=contents,
                embeddings=embeddings,
            )
        except Exception as exc:
            logger.warning("Embedding/storage failed for batch: %s", exc)

    async def index_file(
        self, file_path: str, repo_name: str = "fastapi"
    ) -> dict[str, Any]:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")
        parsed = self.parser.parse_file(file_path, source)

        if parsed.errors:
            return {"success": False, "errors": parsed.errors}

        extractor = EntityExtractor(repo_name=repo_name)
        extraction = extractor.extract(parsed)

        for entity in extraction.entities:
            await self.neo4j.create_entity(entity)
        for rel in extraction.relationships:
            try:
                await self.neo4j.create_relationship(rel)
            except Exception:
                pass

        if extraction.chunks:
            await self._embed_and_store_chunks(extraction.chunks)

        return {
            "success": True,
            "file_path": file_path,
            "entities": len(extraction.entities),
            "relationships": len(extraction.relationships),
            "chunks": len(extraction.chunks),
        }

    async def parse_python_ast(
        self, file_path: str, source: str | None = None
    ) -> dict[str, Any]:
        parsed = self.parser.parse_file(file_path, source)

        return {
            "file_path": file_path,
            "entities": [
                {
                    "type": e.entity_type.value,
                    "name": e.name,
                    "signature": e.signature,
                    "start_line": e.start_line,
                    "end_line": e.end_line,
                    "doc": e.doc[:200] if e.doc else "",
                }
                for e in parsed.entities
            ],
            "imports": parsed.imports,
            "errors": parsed.errors,
        }

    async def extract_entities(
        self, file_path: str, source: str | None = None, repo_name: str = "fastapi"
    ) -> dict[str, Any]:
        parsed = self.parser.parse_file(file_path, source)
        extractor = EntityExtractor(repo_name=repo_name)
        extraction = extractor.extract(parsed)

        return {
            "file_path": file_path,
            "entities": [
                {
                    "entity_id": e.entity_id,
                    "type": e.entity_type.value,
                    "name": e.name,
                    "signature": e.signature,
                    "file_path": e.file_path,
                }
                for e in extraction.entities
            ],
            "relationships": [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "type": r.relationship_type,
                }
                for r in extraction.relationships
            ],
            "chunks": len(extraction.chunks),
        }

    async def get_index_status(self, job_id: str) -> dict[str, Any]:
        job = await self.mongo.get_index_job(job_id)
        if not job:
            return {"error": f"Job {job_id} not found"}
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "repo": job["repo"],
            "progress": job.get("progress", 0),
            "files_processed": job.get("files_processed", 0),
            "files_total": job.get("files_total", 0),
            "entities_extracted": job.get("entities_extracted", 0),
            "errors": job.get("errors", []),
            "started_at": str(job.get("started_at", "")),
            "finished_at": str(job.get("finished_at", "")),
        }
