from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.core.config import settings
from src.observability.logging import get_logger
from src.observability.tracing import LLMTrace

logger = get_logger(__name__)


def get_llm() -> ChatGroq:
    """
    Return a ChatGroq LLM client configured from environment variables.
    Temperature is set to settings.LLM_TEMPERATURE (default 0.0) to ensure
    deterministic, grounded answers. The LLM is never used in the ingestion
    or delta engine stages.
    """
    return ChatGroq(
        temperature=settings.LLM_TEMPERATURE,
        model_name=settings.LLM_MODEL,
        groq_api_key=settings.GROQ_API_KEY,
    )


def call_llm(system_prompt: str, user_message: str, model_name: str) -> str:
    """
    Make a single LLM call with full telemetry tracking.

    Wraps the call in LLMTrace to capture:
    - prompt token count
    - completion token count
    - estimated USD cost
    - wall-clock latency

    Parameters
    ----------
    system_prompt : str
        The grounded context + citation instructions injected into the system role.
    user_message : str
        The user's raw question.
    model_name : str
        The model name string for telemetry logging.

    Returns
    -------
    str
        The LLM's response text.
    """
    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    with LLMTrace(model_name=model_name, stage_label="grounded_chat") as trace:
        response = llm.invoke(messages)

        # Capture token usage if available (Groq returns this in response metadata)
        usage = getattr(response, "usage_metadata", None) or getattr(
            response, "response_metadata", {}
        ).get("token_usage", {})

        trace.record(
            prompt_tokens=usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            completion_tokens=usage.get("output_tokens", usage.get("completion_tokens", 0)),
        )

    return response.content
