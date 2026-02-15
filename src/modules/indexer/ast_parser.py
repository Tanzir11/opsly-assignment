
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.exceptions import ASTParsingError
from src.core.logging import get_logger
from src.core.models import EntityType

logger = get_logger(__name__)


@dataclass
class ParsedEntity:

    entity_type: EntityType
    name: str
    file_path: str
    start_line: int
    end_line: int
    signature: str = ""
    doc: str = ""
    parent_id: str | None = None
    decorators: list[str] = field(default_factory=list)
    parameters: list[dict[str, Any]] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)  # For classes
    imports: list[dict[str, str]] = field(default_factory=list)
    body_source: str = ""

    @property
    def entity_id(self) -> str:
        raw = f"{self.file_path}:{self.entity_type.value}:{self.name}:{self.start_line}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class ParsedFile:

    file_path: str
    entities: list[ParsedEntity] = field(default_factory=list)
    imports: list[dict[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class PythonASTParser:

    def parse_file(self, file_path: str, source: str | None = None) -> ParsedFile:
        result = ParsedFile(file_path=file_path)

        if source is None:
            try:
                source = Path(file_path).read_text(encoding="utf-8")
            except Exception as exc:
                result.errors.append(f"Cannot read file: {exc}")
                return result

        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError as exc:
            result.errors.append(f"Syntax error: {exc}")
            return result
        except Exception as exc:
            raise ASTParsingError(f"Failed to parse {file_path}: {exc}") from exc

        lines = source.splitlines()

        # Extract module-level docstring
        module_doc = ast.get_docstring(tree)
        if module_doc:
            result.entities.append(
                ParsedEntity(
                    entity_type=EntityType.MODULE,
                    name=Path(file_path).stem,
                    file_path=file_path,
                    start_line=1,
                    end_line=1,
                    doc=module_doc,
                )
            )

        # Extract imports at module level
        result.imports = self._extract_imports(tree)

        # Walk the AST to extract entities
        self._walk_module(tree, file_path, lines, result)

        return result

    def _walk_module(
        self,
        tree: ast.Module,
        file_path: str,
        lines: list[str],
        result: ParsedFile,
        parent_id: str | None = None,
    ) -> None:
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                self._extract_class(node, file_path, lines, result, parent_id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._extract_function(
                    node, file_path, lines, result, parent_id, is_method=False
                )
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for imp in self._parse_import_node(node):
                    entity = ParsedEntity(
                        entity_type=EntityType.IMPORT,
                        name=imp.get("name", ""),
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=node.lineno,
                    )
                    entity.imports = [imp]
                    result.entities.append(entity)

    def _extract_class(
        self,
        node: ast.ClassDef,
        file_path: str,
        lines: list[str],
        result: ParsedFile,
        parent_id: str | None,
    ) -> None:
        decorators = [self._decorator_name(d) for d in node.decorator_list]
        bases = [self._node_name(b) for b in node.bases]
        doc = ast.get_docstring(node) or ""
        signature = self._build_class_signature(node)
        end_line = node.end_lineno or node.lineno

        entity = ParsedEntity(
            entity_type=EntityType.CLASS,
            name=node.name,
            file_path=file_path,
            start_line=node.lineno,
            end_line=end_line,
            signature=signature,
            doc=doc,
            parent_id=parent_id,
            decorators=decorators,
            bases=bases,
            body_source=self._get_source_range(lines, node.lineno, end_line),
        )
        result.entities.append(entity)

        # Extract methods within the class
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._extract_function(
                    child, file_path, lines, result,
                    parent_id=entity.entity_id, is_method=True
                )

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: str,
        lines: list[str],
        result: ParsedFile,
        parent_id: str | None,
        is_method: bool,
    ) -> None:
        decorators = [self._decorator_name(d) for d in node.decorator_list]
        doc = ast.get_docstring(node) or ""
        signature = self._build_function_signature(node)
        parameters = self._extract_parameters(node)
        end_line = node.end_lineno or node.lineno

        entity_type = EntityType.METHOD if is_method else EntityType.FUNCTION

        entity = ParsedEntity(
            entity_type=entity_type,
            name=node.name,
            file_path=file_path,
            start_line=node.lineno,
            end_line=end_line,
            signature=signature,
            doc=doc,
            parent_id=parent_id,
            decorators=decorators,
            parameters=parameters,
            body_source=self._get_source_range(lines, node.lineno, end_line),
        )
        result.entities.append(entity)

    def _extract_imports(self, tree: ast.Module) -> list[dict[str, str]]:
        imports: list[dict[str, str]] = []
        for node in ast.walk(tree):
            for imp in self._parse_import_node(node):
                imports.append(imp)
        return imports

    def _parse_import_node(self, node: ast.AST) -> list[dict[str, str]]:
        imports: list[dict[str, str]] = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "module": alias.name,
                    "name": alias.name.split(".")[-1],
                    "alias": alias.asname or "",
                    "type": "import",
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append({
                    "module": module,
                    "name": alias.name,
                    "alias": alias.asname or "",
                    "type": "from_import",
                })
        return imports

    def _extract_parameters(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[dict[str, Any]]:
        params: list[dict[str, Any]] = []
        args = node.args

        # Calculate default offset
        num_defaults = len(args.defaults)
        num_args = len(args.args)
        default_offset = num_args - num_defaults

        for i, arg in enumerate(args.args):
            param: dict[str, Any] = {
                "name": arg.arg,
                "annotation": self._node_name(arg.annotation) if arg.annotation else "",
            }
            default_idx = i - default_offset
            if default_idx >= 0 and default_idx < len(args.defaults):
                param["default"] = ast.dump(args.defaults[default_idx])
            params.append(param)

        # *args
        if args.vararg:
            params.append({
                "name": f"*{args.vararg.arg}",
                "annotation": self._node_name(args.vararg.annotation) if args.vararg.annotation else "",
            })

        # **kwargs
        if args.kwarg:
            params.append({
                "name": f"**{args.kwarg.arg}",
                "annotation": self._node_name(args.kwarg.annotation) if args.kwarg.annotation else "",
            })

        return params

    @staticmethod
    def _decorator_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return ast.dump(node)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id
            elif isinstance(node.func, ast.Attribute):
                return ast.dump(node.func)
        return ast.dump(node)

    @staticmethod
    def _node_name(node: ast.AST | None) -> str:
        if node is None:
            return ""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{PythonASTParser._node_name(node.value)}.{node.attr}"
        if isinstance(node, ast.Constant):
            return str(node.value)
        if isinstance(node, ast.Subscript):
            return f"{PythonASTParser._node_name(node.value)}[{PythonASTParser._node_name(node.slice)}]"
        if isinstance(node, ast.Tuple):
            return ", ".join(PythonASTParser._node_name(e) for e in node.elts)
        return ast.dump(node)

    @staticmethod
    def _build_class_signature(node: ast.ClassDef) -> str:
        bases = ", ".join(PythonASTParser._node_name(b) for b in node.bases)
        if bases:
            return f"class {node.name}({bases})"
        return f"class {node.name}"

    @staticmethod
    def _build_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        args_str = ast.dump(node.args) if node.args else ""

        # Build a simplified args representation
        parts: list[str] = []
        for arg in node.args.args:
            ann = f": {PythonASTParser._node_name(arg.annotation)}" if arg.annotation else ""
            parts.append(f"{arg.arg}{ann}")
        if node.args.vararg:
            parts.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            parts.append(f"**{node.args.kwarg.arg}")

        ret = ""
        if node.returns:
            ret = f" -> {PythonASTParser._node_name(node.returns)}"

        return f"{prefix} {node.name}({', '.join(parts)}){ret}"

    @staticmethod
    def _get_source_range(lines: list[str], start: int, end: int) -> str:
        start_idx = max(0, start - 1)
        end_idx = min(len(lines), end)
        return "\n".join(lines[start_idx:end_idx])
