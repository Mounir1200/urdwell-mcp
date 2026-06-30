"""MCP server exposing UrdWell to language models.

MCP servers are passive: the client-side LLM decides when to call each tool.
Tool docstrings therefore define the behavior visible to the model.

Debug: ``uv run mcp dev urdwell/server.py``
Production over stdio: ``uv run urdwell``
"""

import sys
from typing import Literal

from mcp.server.fastmcp import FastMCP

from urdwell import embeddings
from urdwell import pipeline
from urdwell import ranking
from urdwell.models import Memory, VALID_MEMORY_TYPES
from urdwell.storage import ParquetStore

# Defensive ceilings on tool inputs. Memories are short sentences, so these are
# generous guardrails against accidental or hostile oversized payloads, not
# functional limits.
MAX_CONTENT_CHARS = 10_000
MAX_SEARCH_RESULTS = 50
MAX_ARCHIVE_READ = 1_000

MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0


mcp = FastMCP("UrdWell")
store = ParquetStore()

# Identity of the agent this server instance is wired into, set by `serve` from
# the `--agent` flag that `urdwell install` bakes into each agent's config. It is
# stamped on every memory so its origin is known. None for direct/manual runs.
_agent_id: str | None = None


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
    if len(content) > MAX_CONTENT_CHARS:
        return {"error": f"content exceeds {MAX_CONTENT_CHARS} characters"}
    memory = Memory(
        content=content,
        type=memory_type,
        source=source,
        agent=_agent_id,
        confidence=min(max(confidence, MIN_CONFIDENCE), MAX_CONFIDENCE),
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
    if len(content) > MAX_CONTENT_CHARS:
        return f"error: content exceeds {MAX_CONTENT_CHARS} characters"
    store.append_archive(role, content, session)
    return "archived"


@mcp.tool()
def search_memory(
    query: str,
    k: int = 5,
    include_expired: bool = False,
) -> list[dict]:
    """Search memories with hybrid semantic + keyword ranking.

    Results are ordered by fusing dense (semantic) and lexical (exact-term)
    relevance, so memories that match a name, identifier, or rare word surface
    even when the embedding alone ranks them low. Returns nothing when no
    memory is semantically relevant.

    Args:
        query: A natural-language question or topic.
        k: Maximum number of results.
        include_expired: Include invalidated historical memories.
    """
    k = min(max(k, 1), MAX_SEARCH_RESULTS)
    query_embedding = embeddings.embed(query)
    stored_embeddings = store.all_embeddings()
    candidates = [
        (memory, stored_embeddings[memory.id])
        for memory in store.all(active_only=not include_expired)
        if memory.id in stored_embeddings
    ]
    ranked = ranking.hybrid_rank(
        query,
        query_embedding,
        candidates,
        k,
        cosine_floor=pipeline.SIMILARITY_THRESHOLD,
    )
    return [
        {**memory.to_dict(), "score": round(score, 3)}
        for memory, score in ranked
    ]


@mcp.tool()
def list_memories(
    memory_type: str | None = None,
    agent: str | None = None,
    include_expired: bool = False,
) -> list[dict]:
    """List memories, optionally filtered by type and writing agent."""
    memories = store.all(active_only=not include_expired)
    if memory_type is not None:
        memories = [memory for memory in memories if memory.type == memory_type]
    if agent is not None:
        memories = [memory for memory in memories if memory.agent == agent]
    return [memory.to_dict() for memory in memories]


@mcp.tool()
def check_conflicts(content: str) -> list[dict]:
    """Return active memories that may conflict with the supplied fact."""
    if len(content) > MAX_CONTENT_CHARS:
        return [{"error": f"content exceeds {MAX_CONTENT_CHARS} characters"}]
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
    return store.read_archive(min(max(last_n, 0), MAX_ARCHIVE_READ))


def serve(agent: str | None = None) -> None:
    """Serve UrdWell over stdio (the ``urdwell serve`` command).

    ``agent`` identifies the coding agent this server is wired into. It is
    recorded on every memory written during the session, so each memory carries
    the provenance of the agent that produced it.

    The embedding model is preloaded here, on the main thread, before requests
    arrive: FastMCP runs synchronous tools in worker threads and importing the
    ML backend there can deadlock on Windows. Human-readable messages go to
    stderr because stdout carries the JSON-RPC protocol over stdio.
    """
    global _agent_id
    _agent_id = agent
    print(
        f"UrdWell: initializing {embeddings.backend_name()} embeddings...",
        file=sys.stderr,
    )
    embeddings.embed("warmup")
    print("UrdWell: embedding backend ready.", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    serve()
