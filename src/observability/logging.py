import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Optional

import structlog

from src.core.config import settings

# ── Correlation ID context var ────────────────────────────────────────────────
# Every request sets this var at the API layer so every downstream log line
# automatically carries the same ID — enabling full per-request log filtering.
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    return correlation_id_var.get()


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Set and return a correlation ID. Generates a new UUID if none provided."""
    cid = cid or str(uuid.uuid4())
    correlation_id_var.set(cid)
    return cid


# ── structlog processors ──────────────────────────────────────────────────────

def _add_correlation_id(logger, method_name, event_dict):
    """Inject the current correlation_id into every log record."""
    cid = get_correlation_id()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


def _add_log_level(logger, method_name, event_dict):
    event_dict["level"] = method_name.upper()
    return event_dict


# ── Configure structlog once at import time ───────────────────────────────────

def _build_processors(log_format: str):
    shared = [
        structlog.stdlib.add_logger_name,
        _add_correlation_id,
        _add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_format == "json":
        shared.append(structlog.processors.JSONRenderer())
    else:
        # Human-readable colorized output for local development
        shared.append(structlog.dev.ConsoleRenderer())

    return shared


def configure_logging():
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    log_format = settings.LOG_FORMAT.lower()

    structlog.configure(
        processors=_build_processors(log_format),
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard library logging so third-party libs (e.g.
    # Tesseract, shapely) route their logs through the same structured pipeline.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a bound structlog logger for the given module name.
    Usage:
        logger = get_logger(__name__)
        logger.info("entity_extracted", page=1, count=42)
    """
    configure_logging()
    return structlog.get_logger(name)
