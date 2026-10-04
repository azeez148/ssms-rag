from app.schemas.retrieval import RetrievedChunk


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Formats retrieved chunks into a numbered context block the LLM can cite by index."""
    if not chunks:
        return ""

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.document_title or chunk.document_id
        parts.append(f"[{i}] {title}\n{chunk.content}")

    return "\n\n".join(parts)
