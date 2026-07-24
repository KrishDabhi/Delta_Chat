from typing import List, Tuple

from langchain_core.documents import Document

from src.chat.index import retrieve_relevant_chunks
from src.chat.llm import call_llm
from src.core.config import settings
from src.core.models import ChatResponse
from src.observability.logging import get_logger
from src.observability.tracing import trace_stage

logger = get_logger(__name__)

# System prompt template. The {context} placeholder is filled with retrieved chunks.
# The instructions explicitly require citations and prohibit fabrication.
_SYSTEM_PROMPT_TEMPLATE = """You are a precise document analysis assistant.
You are given retrieved excerpts from two document revisions (Rev A and Rev B)
and a structured delta report showing exactly what changed between them.

RULES YOU MUST FOLLOW:
1. Answer ONLY using the information in the context below.
2. Every factual claim in your answer must end with a citation in this exact format:
   [Source: <source_label> | Page <N> | <region>]
   where source_label is one of: "Rev A", "Rev B", or "Delta Report".
3. If the context does not contain enough information to answer the question,
   respond with: "I cannot find that information in the provided documents."
   Do NOT guess or hallucinate.
4. Be concise. Do not repeat the question back.

CONTEXT:
{context}
"""


@trace_stage("grounded_answer")
def get_grounded_answer(query: str, job_id: str) -> ChatResponse:
    """
    Full RAG pipeline for one user query:
    1. Retrieve top-k relevant chunks from Pinecone (scoped to job_id namespace).
    2. Build a system prompt that injects the chunks as grounded context.
    3. Call the LLM with that prompt and the user's question.
    4. Parse citations from the response.
    5. Return a ChatResponse with answer and citation list.

    Parameters
    ----------
    query : str
        The user's natural language question.
    job_id : str
        Scopes the Pinecone retrieval to only this job's indexed documents.

    Returns
    -------
    ChatResponse
        answer: The LLM's grounded answer.
        citations: Extracted source references from the answer.
        retrieval_chunk_count: How many chunks were retrieved.
    """
    # Step 1: Retrieve
    results: List[Tuple[Document, float]] = retrieve_relevant_chunks(
        query=query, job_id=job_id, k=8
    )

    if not results:
        return ChatResponse(
            answer="No relevant content found for this job. "
                   "Please ensure the documents have been ingested first.",
            citations=[],
            retrieval_chunk_count=0,
        )

    # Step 2: Build context string from retrieved chunks
    context_parts = []
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        chunk_text = doc.page_content
        context_parts.append(f"[{source.upper()} | relevance: {score:.2f}]\n{chunk_text}")

    context = "\n\n---\n\n".join(context_parts)

    # Step 3: Build system prompt
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(context=context)

    logger.debug(
        "rag_context_built",
        job_id=job_id,
        chunk_count=len(results),
        context_length=len(context),
    )

    # Step 4: Call LLM
    answer_text = call_llm(
        system_prompt=system_prompt,
        user_message=query,
        model_name=settings.LLM_MODEL,
    )

    # Step 5: Extract citations from the answer text
    citations = _extract_citations(answer_text)

    return ChatResponse(
        answer=answer_text,
        citations=citations,
        retrieval_chunk_count=len(results),
    )


def _extract_citations(answer_text: str) -> List[str]:
    """
    Extract citation strings from the LLM answer.
    Citations follow the format: [Source: <label> | Page <N> | <region>]

    We parse these out and return them as a deduplicated list so the API
    consumer can display them separately from the main answer text.
    """
    import re
    pattern = re.compile(r"\[Source:[^\]]+\]")
    found = pattern.findall(answer_text)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in found:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique
