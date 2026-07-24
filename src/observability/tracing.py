import functools
import time
from typing import Callable, Optional

from src.observability.logging import get_logger

logger = get_logger(__name__)


def trace_stage(stage_name: str, capture_result_len: bool = False):
    """
    Decorator that times any synchronous function and emits a structured log
    entry at completion (or on failure).

    Parameters
    ----------
    stage_name : str
        Human-readable label for the stage, e.g. "pdf_vector_extraction",
        "delta_matching", "llm_call".
    capture_result_len : bool
        If True and the decorated function returns a list, log its length
        as `output_count`. Useful for adapter functions that return entity lists.

    Usage
    -----
        @trace_stage("pdf_vector_extraction", capture_result_len=True)
        def extract(path: str) -> list[CanonicalEntity]:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

                log_kwargs = {
                    "stage": stage_name,
                    "status": "success",
                    "elapsed_ms": elapsed_ms,
                }
                if capture_result_len and isinstance(result, (list, tuple)):
                    log_kwargs["output_count"] = len(result)

                logger.info("stage_complete", **log_kwargs)
                return result

            except Exception as exc:
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.error(
                    "stage_failed",
                    stage=stage_name,
                    status="error",
                    elapsed_ms=elapsed_ms,
                    error=str(exc),
                    exc_info=True,
                )
                raise  # Re-raise so the caller handles it — we never swallow errors

        return wrapper
    return decorator


class LLMTrace:
    """
    Context manager that records LLM call telemetry:
    prompt token count, completion token count, estimated cost, and latency.

    Usage
    -----
        with LLMTrace(model_name="llama-3.3-70b-versatile") as t:
            response = llm.invoke(prompt)
            t.record(
                prompt_tokens=response.usage_metadata["input_tokens"],
                completion_tokens=response.usage_metadata["output_tokens"],
            )
    """

    # Groq public pricing (USD per 1M tokens) as of 2025.
    # These are read-time constants — not config — because they are set by
    # the provider, not by us.
    _COST_PER_1M_INPUT: float = 0.05
    _COST_PER_1M_OUTPUT: float = 0.08

    def __init__(self, model_name: str, stage_label: str = "llm_call"):
        self.model_name = model_name
        self.stage_label = stage_label
        self._start: float = 0.0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def record(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = round((time.perf_counter() - self._start) * 1000, 2)
        estimated_cost_usd = (
            (self.prompt_tokens / 1_000_000) * self._COST_PER_1M_INPUT
            + (self.completion_tokens / 1_000_000) * self._COST_PER_1M_OUTPUT
        )

        if exc_type is None:
            logger.info(
                "llm_telemetry",
                stage=self.stage_label,
                model=self.model_name,
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                total_tokens=self.prompt_tokens + self.completion_tokens,
                estimated_cost_usd=round(estimated_cost_usd, 6),
                elapsed_ms=elapsed_ms,
            )
        else:
            logger.error(
                "llm_call_failed",
                stage=self.stage_label,
                model=self.model_name,
                elapsed_ms=elapsed_ms,
                error=str(exc_val),
            )
        return False  # Do not suppress exceptions
