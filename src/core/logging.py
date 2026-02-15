
from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

import json

# Context variable for correlation ID - propagated across async calls
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")


def generate_correlation_id() -> str:
    return uuid.uuid4().hex[:16]


class StructuredFormatter(logging.Formatter):

    def __init__(self, service_name: str = "opsly") -> None:
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.default_time_format),
            "level": record.levelname,
            "service": self.service_name,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(""),
            "session_id": session_id_var.get(""),
        }

        # Add extra fields from record
        if hasattr(record, "agent"):
            log_entry["agent"] = record.agent  # type: ignore[attr-defined]
        if hasattr(record, "latency_ms"):
            log_entry["latency_ms"] = record.latency_ms  # type: ignore[attr-defined]
        if hasattr(record, "tool"):
            log_entry["tool"] = record.tool  # type: ignore[attr-defined]

        # Add exception info if present
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, default=str)


def setup_logging(
    service_name: str = "opsly",
    level: str = "INFO",
) -> logging.Logger:
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    logger.handlers.clear()

    # Add structured JSON handler to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter(service_name=service_name))
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str, service_name: str = "opsly") -> logging.Logger:
    return logging.getLogger(f"{service_name}.{name}")
