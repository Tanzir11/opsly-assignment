
from __future__ import annotations

import os
import pytest

# Set test environment before importing configs
os.environ["APP_ENV"] = "development"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "test_password"
os.environ["MONGODB_URI"] = "mongodb://localhost:27017"
os.environ["MILVUS_HOST"] = "localhost"


SAMPLE_PYTHON_CODE = '''
"""Sample module for testing."""

import os
from pathlib import Path
from typing import Optional


class BaseHandler:
    """Base class for all handlers."""

    def __init__(self, name: str) -> None:
        self.name = name

    def handle(self) -> str:
        """Handle the request."""
        return f"handled by {self.name}"


class SpecialHandler(BaseHandler):
    """A special handler with extra functionality."""

    def __init__(self, name: str, level: int = 1) -> None:
        super().__init__(name)
        self.level = level

    def handle(self) -> str:
        """Override handle with special logic."""
        if self.level > 5:
            return "critical"
        return f"special-{self.level}: {super().handle()}"

    async def async_handle(self) -> str:
        """Async handler method."""
        return await self._process()

    async def _process(self) -> str:
        return "processed"


def top_level_function(x: int, y: int = 10) -> int:
    """Add two numbers."""
    return x + y


async def async_top_level(path: str) -> Optional[str]:
    """Async function example."""
    if os.path.exists(path):
        return Path(path).read_text()
    return None
'''


@pytest.fixture
def sample_python_code() -> str:
    return SAMPLE_PYTHON_CODE


@pytest.fixture
def sample_file_path() -> str:
    return "/tmp/test_sample.py"
