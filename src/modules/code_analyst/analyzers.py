
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from src.core.logging import get_logger

logger = get_logger(__name__)


# ─── Design pattern signatures ──────────────────────────────────────────────

DESIGN_PATTERNS = {
    "singleton": {
        "description": "Singleton pattern: ensures only one instance exists",
        "indicators": ["_instance", "__new__", "cls._instance"],
    },
    "factory": {
        "description": "Factory pattern: creates objects without specifying exact class",
        "indicators": ["create_", "factory", "build_", "make_"],
    },
    "decorator": {
        "description": "Decorator pattern: adds behavior to objects dynamically",
        "indicators": ["@", "wrapper", "functools.wraps", "decorator"],
    },
    "observer": {
        "description": "Observer pattern: defines a subscription mechanism",
        "indicators": ["subscribe", "notify", "observer", "listener", "emit", "on_"],
    },
    "strategy": {
        "description": "Strategy pattern: defines a family of interchangeable algorithms",
        "indicators": ["strategy", "algorithm", "handler", "processor"],
    },
    "dependency_injection": {
        "description": "Dependency Injection: dependencies are provided externally",
        "indicators": ["Depends", "inject", "provider", "dependency"],
    },
    "middleware": {
        "description": "Middleware pattern: chain of processing handlers",
        "indicators": ["middleware", "Middleware", "dispatch", "process_request"],
    },
    "context_manager": {
        "description": "Context Manager pattern: resource management with __enter__/__exit__",
        "indicators": ["__enter__", "__exit__", "contextmanager", "asynccontextmanager"],
    },
}


class CodeAnalyzer:

    def analyze_function_code(self, source: str, function_name: str) -> dict[str, Any]:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return {"error": f"Syntax error: {exc}"}

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name:
                    return self._analyze_function_node(node, source)

        return {"error": f"Function '{function_name}' not found in source"}

    def _analyze_function_node(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source: str,
    ) -> dict[str, Any]:
        is_async = isinstance(node, ast.AsyncFunctionDef)
        doc = ast.get_docstring(node) or ""
        lines = source.splitlines()

        # Extract function source
        start = node.lineno - 1
        end = node.end_lineno or node.lineno
        func_source = "\n".join(lines[start:end])

        # Count various metrics
        return_count = sum(1 for n in ast.walk(node) if isinstance(n, ast.Return))
        yield_count = sum(1 for n in ast.walk(node) if isinstance(n, (ast.Yield, ast.YieldFrom)))
        branch_count = sum(
            1 for n in ast.walk(node)
            if isinstance(n, (ast.If, ast.IfExp, ast.Match))
        )
        loop_count = sum(
            1 for n in ast.walk(node)
            if isinstance(n, (ast.For, ast.While, ast.AsyncFor))
        )
        try_count = sum(1 for n in ast.walk(node) if isinstance(n, ast.Try))
        call_count = sum(1 for n in ast.walk(node) if isinstance(n, ast.Call))

        # Extract called functions
        calls = self._extract_calls(node)

        # Detect decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        # Parameters
        params = []
        for arg in node.args.args:
            annotation = ""
            if arg.annotation:
                annotation = ast.dump(arg.annotation)
            params.append({"name": arg.arg, "annotation": annotation})

        # Return annotation
        return_annotation = ""
        if node.returns:
            return_annotation = ast.dump(node.returns)

        # Complexity estimate (simplified cyclomatic)
        complexity = 1 + branch_count + loop_count + try_count

        return {
            "name": node.name,
            "is_async": is_async,
            "docstring": doc,
            "line_count": end - start,
            "parameters": params,
            "return_annotation": return_annotation,
            "decorators": decorators,
            "metrics": {
                "complexity": complexity,
                "return_count": return_count,
                "yield_count": yield_count,
                "branch_count": branch_count,
                "loop_count": loop_count,
                "try_except_count": try_count,
                "call_count": call_count,
            },
            "calls": calls,
            "is_generator": yield_count > 0,
            "source": func_source,
        }

    def analyze_class_code(self, source: str, class_name: str) -> dict[str, Any]:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return {"error": f"Syntax error: {exc}"}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return self._analyze_class_node(node, source)

        return {"error": f"Class '{class_name}' not found in source"}

    def _analyze_class_node(
        self, node: ast.ClassDef, source: str
    ) -> dict[str, Any]:
        doc = ast.get_docstring(node) or ""

        # Base classes
        bases = [self._node_to_str(b) for b in node.bases]

        # Methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = {
                    "name": item.name,
                    "is_async": isinstance(item, ast.AsyncFunctionDef),
                    "decorators": [self._get_decorator_name(d) for d in item.decorator_list],
                    "is_private": item.name.startswith("_") and not item.name.startswith("__"),
                    "is_dunder": item.name.startswith("__") and item.name.endswith("__"),
                    "doc": (ast.get_docstring(item) or "")[:200],
                    "line_count": (item.end_lineno or item.lineno) - item.lineno + 1,
                }
                methods.append(method_info)

        # Class variables
        class_vars = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                class_vars.append({
                    "name": item.target.id,
                    "annotation": self._node_to_str(item.annotation) if item.annotation else "",
                })
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_vars.append({"name": target.id, "annotation": ""})

        # Decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        return {
            "name": node.name,
            "docstring": doc,
            "bases": bases,
            "decorators": decorators,
            "methods": methods,
            "class_variables": class_vars,
            "method_count": len(methods),
            "public_methods": len([m for m in methods if not m["is_private"]]),
            "private_methods": len([m for m in methods if m["is_private"]]),
            "dunder_methods": len([m for m in methods if m["is_dunder"]]),
        }

    def find_patterns(self, source: str) -> list[dict[str, Any]]:
        detected: list[dict[str, Any]] = []
        source_lower = source.lower()

        for pattern_name, pattern_info in DESIGN_PATTERNS.items():
            matches = []
            for indicator in pattern_info["indicators"]:
                if indicator.lower() in source_lower:
                    # Find the line containing the indicator
                    for i, line in enumerate(source.splitlines(), 1):
                        if indicator.lower() in line.lower():
                            matches.append({
                                "indicator": indicator,
                                "line": i,
                                "context": line.strip()[:100],
                            })
                            break

            if matches:
                detected.append({
                    "pattern": pattern_name,
                    "description": pattern_info["description"],
                    "confidence": min(len(matches) / len(pattern_info["indicators"]), 1.0),
                    "evidence": matches,
                })

        return detected

    @staticmethod
    def get_code_snippet(
        source: str,
        start_line: int,
        end_line: int,
        context_lines: int = 3,
    ) -> dict[str, Any]:
        lines = source.splitlines()
        total = len(lines)

        ctx_start = max(0, start_line - 1 - context_lines)
        ctx_end = min(total, end_line + context_lines)

        snippet_lines = []
        for i in range(ctx_start, ctx_end):
            is_target = start_line - 1 <= i < end_line
            snippet_lines.append({
                "line_number": i + 1,
                "content": lines[i],
                "is_target": is_target,
            })

        return {
            "start_line": start_line,
            "end_line": end_line,
            "context_start": ctx_start + 1,
            "context_end": ctx_end,
            "lines": snippet_lines,
            "snippet": "\n".join(lines[max(0, start_line - 1):end_line]),
        }

    @staticmethod
    def _extract_calls(node: ast.AST) -> list[str]:
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        return list(set(calls))

    @staticmethod
    def _get_decorator_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id
            elif isinstance(node.func, ast.Attribute):
                return node.func.attr
        elif isinstance(node, ast.Attribute):
            return node.attr
        return ast.dump(node)

    @staticmethod
    def _node_to_str(node: ast.AST | None) -> str:
        if node is None:
            return ""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{CodeAnalyzer._node_to_str(node.value)}.{node.attr}"
        if isinstance(node, ast.Constant):
            return str(node.value)
        if isinstance(node, ast.Subscript):
            return f"{CodeAnalyzer._node_to_str(node.value)}[{CodeAnalyzer._node_to_str(node.slice)}]"
        return ast.dump(node)
