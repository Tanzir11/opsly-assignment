
from __future__ import annotations

import pytest

from src.core.models import EntityType
from src.modules.indexer.ast_parser import PythonASTParser
from src.modules.indexer.entity_extractor import EntityExtractor


class TestEntityExtractor:

    def setup_method(self) -> None:
        self.parser = PythonASTParser()
        self.extractor = EntityExtractor(repo_name="test-repo")

    def test_extract_entities(self, sample_python_code: str) -> None:
        parsed = self.parser.parse_file("/test.py", sample_python_code)
        result = self.extractor.extract(parsed)

        assert len(result.entities) > 0
        entity_types = {e.entity_type for e in result.entities}
        assert EntityType.CLASS in entity_types
        assert EntityType.FUNCTION in entity_types

    def test_extract_relationships(self, sample_python_code: str) -> None:
        parsed = self.parser.parse_file("/test.py", sample_python_code)
        result = self.extractor.extract(parsed)

        assert len(result.relationships) > 0
        rel_types = {r.relationship_type for r in result.relationships}
        assert "CONTAINS" in rel_types

    def test_extract_chunks(self, sample_python_code: str) -> None:
        parsed = self.parser.parse_file("/test.py", sample_python_code)
        result = self.extractor.extract(parsed)

        assert len(result.chunks) > 0
        # Each chunk should have required fields
        for chunk in result.chunks:
            assert "chunk_id" in chunk
            assert "entity_id" in chunk
            assert "content" in chunk
            assert "token_count" in chunk
            assert chunk["repo"] == "test-repo"

    def test_chunk_ids_are_unique(self, sample_python_code: str) -> None:
        parsed = self.parser.parse_file("/test.py", sample_python_code)
        result = self.extractor.extract(parsed)

        chunk_ids = [c["chunk_id"] for c in result.chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_inheritance_relationship(self, sample_python_code: str) -> None:
        parsed = self.parser.parse_file("/test.py", sample_python_code)
        result = self.extractor.extract(parsed)

        inherit_rels = [
            r for r in result.relationships
            if r.relationship_type == "INHERITS_FROM"
        ]
        assert len(inherit_rels) >= 1

    def test_entity_metadata_includes_repo(self, sample_python_code: str) -> None:
        parsed = self.parser.parse_file("/test.py", sample_python_code)
        result = self.extractor.extract(parsed)

        for entity in result.entities:
            if entity.metadata.get("repo"):
                assert entity.metadata["repo"] == "test-repo"

    def test_documented_by_relationship(self, sample_python_code: str) -> None:
        parsed = self.parser.parse_file("/test.py", sample_python_code)
        result = self.extractor.extract(parsed)

        doc_rels = [
            r for r in result.relationships
            if r.relationship_type == "DOCUMENTED_BY"
        ]
        assert len(doc_rels) > 0

    def test_extract_empty_file(self) -> None:
        parsed = self.parser.parse_file("/empty.py", "")
        result = self.extractor.extract(parsed)
        assert len(result.entities) == 0
        assert len(result.relationships) == 0
        assert len(result.chunks) == 0
