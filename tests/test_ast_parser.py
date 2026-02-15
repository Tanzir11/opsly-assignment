
from __future__ import annotations

import pytest

from src.core.models import EntityType
from src.modules.indexer.ast_parser import PythonASTParser


class TestPythonASTParser:

    def setup_method(self) -> None:
        self.parser = PythonASTParser()

    def test_parse_module_docstring(self, sample_python_code: str) -> None:
        result = self.parser.parse_file("/test.py", sample_python_code)
        module_entities = [
            e for e in result.entities if e.entity_type == EntityType.MODULE
        ]
        assert len(module_entities) == 1
        assert "Sample module" in module_entities[0].doc

    def test_parse_classes(self, sample_python_code: str) -> None:
        result = self.parser.parse_file("/test.py", sample_python_code)
        classes = [e for e in result.entities if e.entity_type == EntityType.CLASS]
        assert len(classes) == 2
        class_names = {c.name for c in classes}
        assert "BaseHandler" in class_names
        assert "SpecialHandler" in class_names

    def test_parse_class_with_bases(self, sample_python_code: str) -> None:
        result = self.parser.parse_file("/test.py", sample_python_code)
        special = next(
            e for e in result.entities
            if e.entity_type == EntityType.CLASS and e.name == "SpecialHandler"
        )
        assert "BaseHandler" in special.bases

    def test_parse_methods(self, sample_python_code: str) -> None:
        result = self.parser.parse_file("/test.py", sample_python_code)
        methods = [e for e in result.entities if e.entity_type == EntityType.METHOD]
        method_names = {m.name for m in methods}
        assert "handle" in method_names
        assert "__init__" in method_names
        assert "async_handle" in method_names

    def test_parse_functions(self, sample_python_code: str) -> None:
        result = self.parser.parse_file("/test.py", sample_python_code)
        functions = [e for e in result.entities if e.entity_type == EntityType.FUNCTION]
        func_names = {f.name for f in functions}
        assert "top_level_function" in func_names
        assert "async_top_level" in func_names

    def test_parse_function_parameters(self, sample_python_code: str) -> None:
        result = self.parser.parse_file("/test.py", sample_python_code)
        func = next(
            e for e in result.entities
            if e.entity_type == EntityType.FUNCTION and e.name == "top_level_function"
        )
        assert len(func.parameters) >= 2
        param_names = {p["name"] for p in func.parameters}
        assert "x" in param_names
        assert "y" in param_names

    def test_parse_imports(self, sample_python_code: str) -> None:
        result = self.parser.parse_file("/test.py", sample_python_code)
        assert len(result.imports) > 0
        import_modules = {imp.get("module", "") for imp in result.imports}
        assert "os" in import_modules or any("os" in m for m in import_modules)

    def test_parse_docstrings(self, sample_python_code: str) -> None:
        result = self.parser.parse_file("/test.py", sample_python_code)
        documented = [e for e in result.entities if e.doc]
        assert len(documented) >= 3  # Module, classes, functions

    def test_parse_empty_file(self) -> None:
        result = self.parser.parse_file("/empty.py", "")
        assert len(result.errors) == 0
        assert len(result.entities) == 0

    def test_parse_syntax_error(self) -> None:
        result = self.parser.parse_file("/bad.py", "def broken(:\n    pass")
        assert len(result.errors) > 0

    def test_entity_ids_are_deterministic(self, sample_python_code: str) -> None:
        result1 = self.parser.parse_file("/test.py", sample_python_code)
        result2 = self.parser.parse_file("/test.py", sample_python_code)

        ids1 = {e.entity_id for e in result1.entities}
        ids2 = {e.entity_id for e in result2.entities}
        assert ids1 == ids2

    def test_parse_decorators(self) -> None:
        code = '''
class MyClass:
    @staticmethod
    def static_method():
        pass

    @property
    def prop(self):
        return self._value
'''
        result = self.parser.parse_file("/deco.py", code)
        methods = [e for e in result.entities if e.entity_type == EntityType.METHOD]
        decorated = [m for m in methods if m.decorators]
        assert len(decorated) >= 2

    def test_parse_async_functions(self, sample_python_code: str) -> None:
        result = self.parser.parse_file("/test.py", sample_python_code)
        async_funcs = [
            e for e in result.entities
            if e.entity_type == EntityType.FUNCTION
            and "async" in e.signature
        ]
        assert len(async_funcs) >= 1
