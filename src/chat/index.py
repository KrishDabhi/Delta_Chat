import uuid
from typing import List, Optional

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

from src.core.config import settings
from src.observability.logging import get_logger
from src.observability.tracing import trace_stage

logger = get_logger(__name__)

# Embedding model — runs locally, no API key required
# all-MiniLM-L6-v2 is 80MB, 384-dim, fast on CPU, good retrieval quality
_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBED_DIM = 384


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=_EMBED_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_or_create_index(index_name: str) -> None:
    """
    Create the Pinecone index if it doesn't already exist.
    Uses a serverless spec on AWS us-east-1 (free tier available).
    """
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=_EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info("pinecone_index_created", index_name=index_name)
    else:
        logger.debug("pinecone_index_exists", index_name=index_name)


def get_vector_store(namespace: str) -> PineconeVectorStore:
    """
    Return a LangChain PineconeVectorStore for the given namespace.
    Each job_id gets its own namespace so documents from different jobs
    never mix during retrieval.
    """
    _get_or_create_index(settings.PINECONE_INDEX_NAME)
    return PineconeVectorStore(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=_get_embeddings(),
        namespace=namespace,
    )


@trace_stage("index_document_chunks", capture_result_len=True)
def index_chunks(chunks: List[str], job_id: str, source_label: str) -> int:
    """
    Embed and upsert a list of text chunks into Pinecone.

    Parameters
    ----------
    chunks : List[str]
        Plain text strings to embed. Each becomes one Pinecone vector.
    job_id : str
        Used as the Pinecone namespace — isolates this job's data.
    source_label : str
        One of: "rev_a", "rev_b", "delta_report".
        Stored in the chunk metadata for citation purposes.

    Returns
    -------
    int
        Number of chunks successfully indexed.
    """
    if not chunks:
        return 0

    store = get_vector_store(namespace=job_id)
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content=chunk,
            metadata={
                "job_id": job_id,
                "source": source_label,
                "chunk_index": i,
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    # Deterministic IDs prevent duplicate upserts if the job is re-run
    import hashlib
    ids = [
        hashlib.md5(f"{job_id}:{source_label}:{i}".encode()).hexdigest()
        for i in range(len(docs))
    ]

    store.add_documents(docs, ids=ids)
    logger.info(
        "chunks_indexed",
        job_id=job_id,
        source=source_label,
        count=len(docs),
    )
    return len(docs)


@trace_stage("retrieve_chunks")
def retrieve_relevant_chunks(
    query: str, job_id: str, k: int = 8
) -> List[tuple]:
    """
    Retrieve the top-k most semantically similar chunks for a query,
    scoped to the given job_id namespace.

    Returns
    -------
    List[tuple]
        List of (Document, score) pairs sorted by relevance (highest first).
    """
    store = get_vector_store(namespace=job_id)
    results = store.similarity_search_with_score(query, k=k)
    logger.debug("retrieval_complete", query=query[:60], k=k, hits=len(results))
    return results
