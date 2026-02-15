
from __future__ import annotations

import hashlib
import uuid
from typing import Any

from src.core.logging import get_logger
from src.core.models import CodeEntity, CodeRelationship, EntityType
from src.modules.indexer.ast_parser import ParsedEntity, ParsedFile

logger = get_logger(__name__)


class EntityExtractor:

    def __init__(self, repo_name: str = "fastapi", chunk_max_tokens: int = 512) -> None:
        self.repo_name = repo_name
        self.chunk_max_tokens = chunk_max_tokens

    def extract(self, parsed_file: ParsedFile) -> ExtractionResult:
        result = ExtractionResult(file_path=parsed_file.file_path)

        # Map from ParsedEntity to graph entity
        entity_id_map: dict[str, str] = {}  # (type:name:line) -> entity_id

        for parsed in parsed_file.entities:
            entity = self._to_code_entity(parsed)
            result.entities.append(entity)
            entity_id_map[self._entity_key(parsed)] = entity.entity_id

            # Create chunks from entities with body source
            if parsed.body_source and parsed.entity_type in (
                EntityType.CLASS,
                EntityType.FUNCTION,
                EntityType.METHOD,
            ):
                chunks = self._create_chunks(entity, parsed.body_source)
                result.chunks.extend(chunks)

        # Build relationships
        relationships = self._build_relationships(parsed_file, entity_id_map)
        result.relationships.extend(relationships)

        return result

    def _to_code_entity(self, parsed: ParsedEntity) -> CodeEntity:
        return CodeEntity(
            entity_id=parsed.entity_id,
            entity_type=parsed.entity_type,
            name=parsed.name,
            file_path=parsed.file_path,
            signature=parsed.signature,
            start_line=parsed.start_line,
            end_line=parsed.end_line,
            doc=parsed.doc,
            metadata={
                "decorators": parsed.decorators,
                "parameters": parsed.parameters,
                "bases": parsed.bases,
                "repo": self.repo_name,
            },
        )

    def _create_chunks(
        self, entity: CodeEntity, source: str
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []

        # Estimate tokens (~4 chars per token)
        approx_tokens = len(source) // 4

        if approx_tokens <= self.chunk_max_tokens:
            # Single chunk
            chunk_content = f"# {entity.signature}\n{source}" if entity.signature else source
            chunk_id = hashlib.sha256(
                f"{entity.entity_id}:0".encode()
            ).hexdigest()[:16]
            chunks.append({
                "chunk_id": chunk_id,
                "entity_id": entity.entity_id,
                "file_path": entity.file_path,
                "content": chunk_content,
                "token_count": len(chunk_content) // 4,
                "entity_type": entity.entity_type.value,
                "repo": self.repo_name,
            })
        else:
            # Split into multiple chunks by lines
            lines = source.splitlines()
            current_chunk: list[str] = []
            current_tokens = 0
            chunk_idx = 0

            for line in lines:
                line_tokens = len(line) // 4 + 1
                if current_tokens + line_tokens > self.chunk_max_tokens and current_chunk:
                    chunk_content = "\n".join(current_chunk)
                    if entity.signature and chunk_idx == 0:
                        chunk_content = f"# {entity.signature}\n{chunk_content}"
                    chunk_id = hashlib.sha256(
                        f"{entity.entity_id}:{chunk_idx}".encode()
                    ).hexdigest()[:16]
                    chunks.append({
                        "chunk_id": chunk_id,
                        "entity_id": entity.entity_id,
                        "file_path": entity.file_path,
                        "content": chunk_content,
                        "token_count": len(chunk_content) // 4,
                        "entity_type": entity.entity_type.value,
                        "repo": self.repo_name,
                    })
                    current_chunk = []
                    current_tokens = 0
                    chunk_idx += 1

                current_chunk.append(line)
                current_tokens += line_tokens

            # Final chunk
            if current_chunk:
                chunk_content = "\n".join(current_chunk)
                chunk_id = hashlib.sha256(
                    f"{entity.entity_id}:{chunk_idx}".encode()
                ).hexdigest()[:16]
                chunks.append({
                    "chunk_id": chunk_id,
                    "entity_id": entity.entity_id,
                    "file_path": entity.file_path,
                    "content": chunk_content,
                    "token_count": len(chunk_content) // 4,
                    "entity_type": entity.entity_type.value,
                    "repo": self.repo_name,
                })

        return chunks

    def _build_relationships(
        self,
        parsed_file: ParsedFile,
        entity_id_map: dict[str, str],
    ) -> list[CodeRelationship]:
        relationships: list[CodeRelationship] = []

        for parsed in parsed_file.entities:
            source_id = parsed.entity_id

            # CONTAINS: parent -> child
            if parsed.parent_id:
                relationships.append(
                    CodeRelationship(
                        source_id=parsed.parent_id,
                        target_id=source_id,
                        relationship_type="CONTAINS",
                    )
                )

            # INHERITS_FROM: class -> base classes
            for base in parsed.bases:
                # Try to find the base class in our entity map
                for other in parsed_file.entities:
                    if other.name == base and other.entity_type == EntityType.CLASS:
                        relationships.append(
                            CodeRelationship(
                                source_id=source_id,
                                target_id=other.entity_id,
                                relationship_type="INHERITS_FROM",
                            )
                        )
                        break

            # DECORATED_BY: entity -> decorator
            for dec in parsed.decorators:
                for other in parsed_file.entities:
                    if other.name == dec and other.entity_type == EntityType.DECORATOR:
                        relationships.append(
                            CodeRelationship(
                                source_id=source_id,
                                target_id=other.entity_id,
                                relationship_type="DECORATED_BY",
                            )
                        )
                        break

            # DOCUMENTED_BY: entity -> docstring (if entity has doc)
            if parsed.doc:
                doc_entity_id = hashlib.sha256(
                    f"{parsed.file_path}:Docstring:{parsed.name}:{parsed.start_line}".encode()
                ).hexdigest()[:16]
                relationships.append(
                    CodeRelationship(
                        source_id=source_id,
                        target_id=doc_entity_id,
                        relationship_type="DOCUMENTED_BY",
                        metadata={"doc_content": parsed.doc[:500]},
                    )
                )

        # IMPORTS relationships from file-level imports
        for imp in parsed_file.imports:
            imp_entity_id = hashlib.sha256(
                f"{parsed_file.file_path}:Import:{imp.get('name', '')}".encode()
            ).hexdigest()[:16]
            # We'll link imports to the module entity if present
            for entity in parsed_file.entities:
                if entity.entity_type == EntityType.MODULE:
                    relationships.append(
                        CodeRelationship(
                            source_id=entity.entity_id,
                            target_id=imp_entity_id,
                            relationship_type="IMPORTS",
                            metadata={"module": imp.get("module", "")},
                        )
                    )
                    break

        return relationships

    @staticmethod
    def _entity_key(parsed: ParsedEntity) -> str:
        return f"{parsed.entity_type.value}:{parsed.name}:{parsed.start_line}"


class ExtractionResult:

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.entities: list[CodeEntity] = []
        self.relationships: list[CodeRelationship] = []
        self.chunks: list[dict[str, Any]] = []
