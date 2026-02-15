
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import git

from src.core.exceptions import RepositoryCloneError
from src.core.logging import get_logger

logger = get_logger(__name__)


class RepoLoader:

    def __init__(self, workspace_dir: str = "/tmp/opsly/repos") -> None:
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def _repo_path(self, repo_url: str) -> Path:
        # Extract repo name from URL
        name = repo_url.rstrip("/").split("/")[-1]
        if name.endswith(".git"):
            name = name[:-4]
        return self.workspace_dir / name

    async def clone_or_pull(self, repo_url: str) -> Path:
        repo_path = self._repo_path(repo_url)

        try:
            if repo_path.exists() and (repo_path / ".git").exists():
                logger.info("Pulling latest changes for %s", repo_url)
                repo = git.Repo(repo_path)
                origin = repo.remotes.origin
                origin.pull()
                logger.info("Pull complete for %s", repo_url)
            else:
                if repo_path.exists():
                    shutil.rmtree(repo_path)
                logger.info("Cloning %s to %s", repo_url, repo_path)
                git.Repo.clone_from(
                    repo_url, repo_path, depth=1, single_branch=True
                )
                logger.info("Clone complete for %s", repo_url)
        except git.GitCommandError as exc:
            raise RepositoryCloneError(
                f"Git operation failed for {repo_url}: {exc}"
            ) from exc
        except Exception as exc:
            raise RepositoryCloneError(
                f"Failed to clone/pull {repo_url}: {exc}"
            ) from exc

        return repo_path

    def discover_python_files(
        self, repo_path: Path, exclude_patterns: list[str] | None = None
    ) -> list[Path]:
        exclude = set(exclude_patterns or [
            "__pycache__",
            ".git",
            ".tox",
            "venv",
            ".venv",
            "node_modules",
            ".eggs",
            "*.egg-info",
            "dist",
            "build",
        ])

        python_files: list[Path] = []

        for py_file in repo_path.rglob("*.py"):
            # Check if any part of the path matches exclude patterns
            parts = py_file.relative_to(repo_path).parts
            if any(exc in parts for exc in exclude):
                continue
            python_files.append(py_file)

        python_files.sort()
        logger.info("Discovered %d Python files in %s", len(python_files), repo_path)
        return python_files

    def get_file_sha(self, file_path: Path) -> str:
        import hashlib

        content = file_path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]

    def get_relative_path(self, file_path: Path, repo_path: Path) -> str:
        try:
            return str(file_path.relative_to(repo_path))
        except ValueError:
            return str(file_path)

    def get_changed_files(self, repo_path: Path) -> list[Path]:
        try:
            repo = git.Repo(repo_path)
            changed: list[Path] = []

            # Check for uncommitted changes
            for item in repo.index.diff(None):
                path = repo_path / item.a_path
                if path.suffix == ".py" and path.exists():
                    changed.append(path)

            # Also include untracked files
            for path_str in repo.untracked_files:
                path = repo_path / path_str
                if path.suffix == ".py":
                    changed.append(path)

            return sorted(set(changed))
        except Exception as exc:
            logger.warning("Could not detect changed files: %s", exc)
            return []
