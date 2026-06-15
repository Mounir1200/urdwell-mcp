"""MCP server exposing ContextMemory to language models.

MCP servers are passive: the client-side LLM decides when to call each tool.
Tool docstrings therefore define the behavior visible to the model.

Debug: ``uv run mcp dev context_memory.py``
Production over stdio: ``uv run context_memory.py``
"""

import sys
from typing import Literal

from mcp.server.fastmcp import FastMCP

import embeddings
from models import Memory, VALID_MEMORY_TYPES
import pipeline
from storage import JsonStore


mcp = FastMCP("ContextMemory")
store = JsonStore()

# Preload the model on the main thread. FastMCP runs synchronous tools in
# worker threads, and importing torch there can deadlock on Windows.
# Human-readable process messages must use stderr because stdout belongs to
# the JSON-RPC protocol when using stdio transport.
print(
    f"ContextMemory: initializing {embeddings.backend_name()} embeddings...",
    file=sys.stderr,
)
embeddings.embed("warmup")
print("ContextMemory: embedding backend ready.", file=sys.stderr)


@mcp.tool()
def save_memory(
    content: str,
    memory_type: str,
    source: str | None = None,
    confidence: float = 0.8,
    decision: Literal["ADD", "IGNORE", "EXPIRE"] | None = None,
    target_id: str | None = None,
) -> dict:
    """Store an important piece of information in long-term memory.

    Call this tool as soon as a user preference, decision, project fact, or
    temporary state is worth retaining.

    Args:
        content: One precise, self-contained sentence.
        memory_type: "fact", "preference", "decision", or "temporary_state".
        source: Exact source quote or session reference when available.
        confidence: Confidence score from 0 to 1.
        decision: Use only after ARBITRATION_REQUIRED. Choose ADD when facts
            are compatible, IGNORE for a duplicate, or EXPIRE when the new
            fact supersedes an old one.
        target_id: Candidate ID required for IGNORE and EXPIRE.

    Returns:
        A pipeline report. ARBITRATION_REQUIRED means nothing was written;
        compare the candidates and immediately call save_memory again with
        the same content and an explicit decision.
    """
    if memory_type not in VALID_MEMORY_TYPES:
        return {
            "error": (
                f"invalid memory type; expected one of "
                f"{sorted(VALID_MEMORY_TYPES)}"
            )
        }
    memory = Memory(
        content=content,
        type=memory_type,
        source=source,
        confidence=confidence,
    )
    return pipeline.process_memory(
        store,
        memory,
        decision=decision,
        target_id=target_id,
    )


@mcp.tool()
def archive_exchange(role: str, content: str, session: str | None = None) -> str:
    """Append a verbatim exchange to the immutable raw archive.

    Use this source-of-truth layer when exact wording must survive context
    compaction. No analysis or consolidation is performed.
    """
    store.append_archive(role, content, session)
    return "archived"


@mcp.tool()
def search_memory(
    query: str,
    k: int = 5,
    include_expired: bool = False,
) -> list[dict]:
    """Search memories semantically.

    Args:
        query: A natural-language question or topic.
        k: Maximum number of results.
        include_expired: Include invalidated historical memories.
    """
    query_embedding = embeddings.embed(query)
    candidates = store.all(active_only=not include_expired)
    stored_embeddings = store.all_embeddings()
    scores = []
    for memory in candidates:
        stored_embedding = stored_embeddings.get(memory.id)
        if stored_embedding is None:
            continue
        score = embeddings.cosine_similarity(query_embedding, stored_embedding)
        scores.append((memory, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return [
        {**memory.to_dict(), "score": round(score, 3)}
        for memory, score in scores[:k]
        if score >= pipeline.SIMILARITY_THRESHOLD
    ]


@mcp.tool()
def list_memories(
    memory_type: str | None = None,
    include_expired: bool = False,
) -> list[dict]:
    """List memories, optionally filtered by type."""
    memories = store.all(active_only=not include_expired)
    if memory_type is not None:
        memories = [memory for memory in memories if memory.type == memory_type]
    return [memory.to_dict() for memory in memories]


@mcp.tool()
def check_conflicts(content: str) -> list[dict]:
    """Return active memories that may conflict with the supplied fact."""
    content_embedding = embeddings.embed(content)
    similar_memories = pipeline.find_similar_memories(store, content_embedding)
    return [
        {
            "id": memory.id,
            "content": memory.content,
            "type": memory.type,
            "valid_from": memory.valid_from,
            "score": round(score, 3),
        }
        for memory, score in similar_memories
    ]


@mcp.tool()
def read_archive(last_n: int = 50) -> list[dict]:
    """Read the last entries from the verbatim raw archive."""
    return store.read_archive(last_n)


if __name__ == "__main__":
    mcp.run(transport="stdio")
