from app.schemas.retrieval import RetrievedChunk


def retrieval_is_too_weak(chunks: list[RetrievedChunk], similarity_threshold: float) -> bool:
    """True if nothing was retrieved, or even the best match falls below the
    similarity threshold — a signal the question is likely outside what the
    knowledge base actually covers. Checked before calling the LLM at all, so
    a weak-retrieval case costs nothing beyond the retrieval call itself, and
    the LLM never gets a chance to improvise an answer from irrelevant context.
    """
    if not chunks:
        return True
    return chunks[0].similarity < similarity_threshold
