"""
Structured Logging Configuration for Adaptive Multi-Agent Orchestration System.

Provides standard library logging with JSON-friendly formatting and contextual fields.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """
    Formats log records as structured, single-line JSON objects.
    Ensures easy parsing by cloud logging agents and telemetry systems.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include location info for WARN and ERROR
        if record.levelno >= logging.WARNING:
            log_entry["module"] = record.module
            log_entry["function"] = record.funcName
            log_entry["line"] = record.lineno

        # Extract extra structured fields if attached
        if hasattr(record, "run_id") and record.run_id:
            log_entry["run_id"] = record.run_id
        if hasattr(record, "task_id") and record.task_id:
            log_entry["task_id"] = record.task_id
        if hasattr(record, "extra_context") and isinstance(record.extra_context, dict):
            log_entry["context"] = record.extra_context

        # Exception formatting
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class StandardFormatter(logging.Formatter):
    """
    Human-readable console formatter for local terminal readability.
    """

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def configure_logging(level: Optional[str] = None, json_format: Optional[bool] = None) -> None:
    """
    Configure the root logger and standard library logging handlers.
    """
    log_level_str = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    use_json = json_format if json_format is not None else (
        os.getenv("LOG_JSON", "false").lower() in ("true", "1", "yes")
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers on re-configuration
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    if use_json:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(StandardFormatter())

    root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve a named logger instance configured for the system.
    """
    return logging.getLogger(name)
