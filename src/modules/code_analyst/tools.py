
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.llm_router import LLMRouter
from src.core.logging import get_logger
from src.db.neo4j import Neo4jRepository
from src.modules.code_analyst.analyzers import CodeAnalyzer

logger = get_logger(__name__)


class CodeAnalystTools:

    def __init__(
        self,
        neo4j_repo: Neo4jRepository,
        repo_workspace: str = "/tmp/opsly/repos",
        llm_api_keys: dict[str, list[str]] | None = None,
    ) -> None:
        self.neo4j = neo4j_repo
        self.analyzer = CodeAnalyzer()
        self.repo_workspace = Path(repo_workspace)
        self.llm = LLMRouter(api_keys=llm_api_keys or {})

    def _read_file_source(self, file_path: str) -> str | None:
        # Try direct path
        path = Path(file_path)
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")

        # Try relative to repo workspace
        for repo_dir in self.repo_workspace.iterdir():
            candidate = repo_dir / file_path
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8", errors="replace")

        return None

    @staticmethod
    def _labels_match_entity_type(labels: list[str], entity_type: str) -> bool:
        if entity_type == "class":
            return "Class" in labels
        if entity_type == "function":
            return "Function" in labels or "Method" in labels
        return False

    def _rank_entity_candidates(
        self,
        entities: list[dict[str, Any]],
        name: str,
        entity_type: str,
    ) -> list[dict[str, Any]]:
        exact_name = name.strip().lower()

        compatible = [
            e for e in entities
            if self._labels_match_entity_type(e.get("labels", []), entity_type)
        ]
        fallback = [e for e in entities if e not in compatible]
        candidates = compatible or fallback

        # Exact name matches first, then keep existing order
        exact = [
            e for e in candidates
            if str(e.get("properties", {}).get("name", "")).strip().lower() == exact_name
        ]
        non_exact = [e for e in candidates if e not in exact]
        return exact + non_exact

    async def analyze_function(
        self,
        name: str,
        file_path: str | None = None,
        entity_id: str | None = None,
    ) -> dict[str, Any]:
        source = None

        # Try to find the function source
        if file_path:
            source = self._read_file_source(file_path)
        elif entity_id:
            # Look up in Neo4j
            entities = await self.neo4j.find_entity_by_name(name)
            for e in entities:
                fp = e["properties"].get("file_path", "")
                if fp:
                    source = self._read_file_source(fp)
                    if source:
                        file_path = fp
                        break

        if not source:
            # Try searching graph by name
            entities = await self.neo4j.find_entity_by_name(name)
            for e in entities:
                fp = e["properties"].get("file_path", "")
                if fp:
                    source = self._read_file_source(fp)
                    if source:
                        file_path = fp
                        break

        if not source:
            return {"error": f"Could not find source for function '{name}'"}

        analysis = self.analyzer.analyze_function_code(source, name)
        if "error" in analysis:
            return analysis

        analysis["file_path"] = file_path or ""
        return analysis

    async def analyze_class(
        self,
        name: str,
        file_path: str | None = None,
        entity_id: str | None = None,
    ) -> dict[str, Any]:
        source = None

        if file_path:
            source = self._read_file_source(file_path)
        elif entity_id:
            entities = await self.neo4j.find_entity_by_name(name)
            for e in entities:
                fp = e["properties"].get("file_path", "")
                if fp:
                    source = self._read_file_source(fp)
                    if source:
                        file_path = fp
                        break

        if not source:
            entities = await self.neo4j.find_entity_by_name(name)
            for e in entities:
                fp = e["properties"].get("file_path", "")
                if fp:
                    source = self._read_file_source(fp)
                    if source:
                        file_path = fp
                        break

        if not source:
            return {"error": f"Could not find source for class '{name}'"}

        analysis = self.analyzer.analyze_class_code(source, name)
        if "error" in analysis:
            return analysis

        analysis["file_path"] = file_path or ""
        return analysis

    async def find_patterns(
        self,
        file_path: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        if source is None and file_path:
            source = self._read_file_source(file_path)

        if not source:
            return {"error": "No source code provided or found"}

        patterns = self.analyzer.find_patterns(source)
        return {
            "file_path": file_path or "",
            "patterns_found": len(patterns),
            "patterns": patterns,
        }

    async def get_code_snippet(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        context_lines: int = 3,
    ) -> dict[str, Any]:
        source = self._read_file_source(file_path)
        if not source:
            return {"error": f"Cannot read file: {file_path}"}

        return self.analyzer.get_code_snippet(source, start_line, end_line, context_lines)

    async def explain_implementation(
        self,
        name: str,
        file_path: str | None = None,
        entity_type: str = "function",
    ) -> dict[str, Any]:
        # Normalize entity type to known options
        entity_type = "class" if entity_type == "class" else "function"
        analyze_code = (
            self.analyzer.analyze_class_code
            if entity_type == "class"
            else self.analyzer.analyze_function_code
        )

        # Try explicit file first, then ranked graph candidates.
        tried_files: set[str] = set()
        last_analysis_error = ""
        analysis: dict[str, Any] | None = None
        source = None

        if file_path:
            source = self._read_file_source(file_path)
            if source:
                analysis = analyze_code(source, name)
                if "error" not in analysis:
                    tried_files.add(file_path)
                else:
                    last_analysis_error = str(analysis.get("error", ""))

        if analysis is None or "error" in analysis:
            entities = await self.neo4j.find_entity_by_name(name)
            for entity in self._rank_entity_candidates(entities, name, entity_type):
                fp = str(entity.get("properties", {}).get("file_path", "")).strip()
                if not fp or fp in tried_files:
                    continue
                candidate_source = self._read_file_source(fp)
                if not candidate_source:
                    continue
                candidate_analysis = analyze_code(candidate_source, name)
                if "error" in candidate_analysis:
                    tried_files.add(fp)
                    last_analysis_error = str(candidate_analysis.get("error", ""))
                    continue
                file_path = fp
                source = candidate_source
                analysis = candidate_analysis
                break

        if not source or analysis is None or "error" in analysis:
            if last_analysis_error:
                return {"error": last_analysis_error}
            return {"error": f"Could not find source for '{name}'"}

        # Build LLM prompt
        code_snippet = analysis.get("source", "")
        if not code_snippet:
            code_snippet = source[:3000]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Python code analyst. Provide clear, concise "
                    "explanations of code implementations. Reference specific lines "
                    "and patterns. Be precise and technical."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Explain the following Python {entity_type} '{name}':\n\n"
                    f"```python\n{code_snippet[:4000]}\n```\n\n"
                    "Provide:\n"
                    "1. Purpose and responsibility\n"
                    "2. Key implementation details\n"
                    "3. Design patterns used\n"
                    "4. Important relationships/dependencies"
                ),
            },
        ]

        try:
            llm_result = await self.llm.generate(messages, task="code_explanation")
            explanation = llm_result["content"]
        except Exception as exc:
            logger.warning("LLM explanation failed: %s", exc)
            explanation = (
                f"LLM explanation unavailable. "
                f"Based on AST analysis: {entity_type} '{name}' "
                f"has {analysis.get('method_count', analysis.get('metrics', {}).get('complexity', 'unknown'))} "
                f"complexity indicators."
            )

        return {
            "name": name,
            "entity_type": entity_type,
            "file_path": file_path or "",
            "explanation": explanation,
            "analysis": analysis,
        }

    async def compare_implementations(
        self,
        name_a: str,
        name_b: str,
        file_path_a: str | None = None,
        file_path_b: str | None = None,
    ) -> dict[str, Any]:
        # Analyze both entities
        analysis_a = await self.analyze_function(name_a, file_path_a)
        analysis_b = await self.analyze_function(name_b, file_path_b)

        # If function analysis failed, try class analysis
        if "error" in analysis_a:
            analysis_a = await self.analyze_class(name_a, file_path_a)
        if "error" in analysis_b:
            analysis_b = await self.analyze_class(name_b, file_path_b)

        comparison = {
            "entity_a": {"name": name_a, "analysis": analysis_a},
            "entity_b": {"name": name_b, "analysis": analysis_b},
        }

        # Try LLM comparison if both have source
        source_a = analysis_a.get("source", "")
        source_b = analysis_b.get("source", "")

        if source_a and source_b:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert Python code analyst. Compare two code implementations precisely.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Compare these two implementations:\n\n"
                        f"## {name_a}\n```python\n{source_a[:2000]}\n```\n\n"
                        f"## {name_b}\n```python\n{source_b[:2000]}\n```\n\n"
                        "Analyze: similarities, differences, trade-offs, and "
                        "which approach is better for what use case."
                    ),
                },
            ]
            try:
                llm_result = await self.llm.generate(messages, task="code_explanation")
                comparison["llm_comparison"] = llm_result["content"]
            except Exception as exc:
                comparison["llm_comparison"] = f"LLM comparison unavailable: {exc}"

        return comparison
