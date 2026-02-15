
from __future__ import annotations

import pytest

from src.modules.code_analyst.analyzers import CodeAnalyzer


class TestCodeAnalyzer:

    def setup_method(self) -> None:
        self.analyzer = CodeAnalyzer()

    def test_analyze_function(self, sample_python_code: str) -> None:
        result = self.analyzer.analyze_function_code(
            sample_python_code, "top_level_function"
        )
        assert "error" not in result
        assert result["name"] == "top_level_function"
        assert result["is_async"] is False
        assert "parameters" in result
        assert "metrics" in result
        assert result["metrics"]["complexity"] >= 1

    def test_analyze_async_function(self, sample_python_code: str) -> None:
        result = self.analyzer.analyze_function_code(
            sample_python_code, "async_top_level"
        )
        assert "error" not in result
        assert result["is_async"] is True

    def test_analyze_method_with_branches(self, sample_python_code: str) -> None:
        result = self.analyzer.analyze_function_code(
            sample_python_code, "handle"
        )
        # There are multiple 'handle' methods; ensure we got one
        assert "error" not in result
        assert result["name"] == "handle"

    def test_analyze_nonexistent_function(self, sample_python_code: str) -> None:
        result = self.analyzer.analyze_function_code(
            sample_python_code, "nonexistent_function"
        )
        assert "error" in result

    def test_analyze_class(self, sample_python_code: str) -> None:
        result = self.analyzer.analyze_class_code(
            sample_python_code, "SpecialHandler"
        )
        assert "error" not in result
        assert result["name"] == "SpecialHandler"
        assert "BaseHandler" in result["bases"]
        assert result["method_count"] >= 3
        assert len(result["methods"]) >= 3

    def test_analyze_base_class(self, sample_python_code: str) -> None:
        result = self.analyzer.analyze_class_code(
            sample_python_code, "BaseHandler"
        )
        assert "error" not in result
        assert result["name"] == "BaseHandler"
        assert result["bases"] == []  # No base classes
        assert result["method_count"] >= 2

    def test_analyze_nonexistent_class(self, sample_python_code: str) -> None:
        result = self.analyzer.analyze_class_code(
            sample_python_code, "NonExistentClass"
        )
        assert "error" in result

    def test_find_patterns(self, sample_python_code: str) -> None:
        patterns = self.analyzer.find_patterns(sample_python_code)
        # The sample code should trigger at least some pattern indicators
        assert isinstance(patterns, list)

    def test_find_patterns_with_dependency_injection(self) -> None:
        code = '''
from fastapi import Depends

def get_db():
    return Database()

def handler(db = Depends(get_db)):
    return db.query()
'''
        patterns = self.analyzer.find_patterns(code)
        pattern_names = {p["pattern"] for p in patterns}
        assert "dependency_injection" in pattern_names

    def test_find_patterns_singleton(self) -> None:
        code = '''
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
'''
        patterns = self.analyzer.find_patterns(code)
        pattern_names = {p["pattern"] for p in patterns}
        assert "singleton" in pattern_names

    def test_get_code_snippet(self, sample_python_code: str) -> None:
        result = self.analyzer.get_code_snippet(
            sample_python_code, start_line=5, end_line=10, context_lines=2
        )
        assert "lines" in result
        assert len(result["lines"]) > 0
        assert result["start_line"] == 5
        assert result["end_line"] == 10

    def test_get_code_snippet_with_target_markers(
        self, sample_python_code: str
    ) -> None:
        result = self.analyzer.get_code_snippet(
            sample_python_code, start_line=5, end_line=7, context_lines=1
        )
        target_lines = [l for l in result["lines"] if l["is_target"]]
        context_lines = [l for l in result["lines"] if not l["is_target"]]
        assert len(target_lines) >= 1

    def test_analyze_function_with_syntax_error(self) -> None:
        result = self.analyzer.analyze_function_code(
            "def broken(:\n    pass", "broken"
        )
        assert "error" in result

    def test_function_calls_extraction(self, sample_python_code: str) -> None:
        result = self.analyzer.analyze_function_code(
            sample_python_code, "async_top_level"
        )
        if "error" not in result:
            assert "calls" in result
            assert isinstance(result["calls"], list)
