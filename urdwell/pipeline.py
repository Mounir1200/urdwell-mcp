"""Memory consolidation pipeline.

Each incoming memory is embedded, compared with active memories, classified as
a duplicate, contradiction, or new fact, and then persisted. Contradictions
expire old memories instead of deleting them.
"""

import unicodedata

from urdwell import embeddings
from urdwell.models import Memory
from urdwell.storage import ParquetStore

# Tune this threshold empirically for the selected embedding model.
SIMILARITY_THRESHOLD = 0.55

ARBITRATION_ACTIONS = {"ADD", "IGNORE", "EXPIRE"}


def find_similar_memories(
    store: ParquetStore,
    embedding: list[float],
    k: int = 5,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[tuple[Memory, float]]:
    """Return the highest-scoring active memories above the threshold.

    A linear scan is sufficient for a few thousand memories. Larger stores
    should use a vector index.
    """
    scores = []
    stored_embeddings = store.all_embeddings()
    for memory in store.all(active_only=True):
        stored_embedding = stored_embeddings.get(memory.id)
        if stored_embedding is None:
            continue
        score = embeddings.cosine_similarity(embedding, stored_embedding)
        if score >= threshold:
            scores.append((memory, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:k]


def normalize_text(text: str) -> str:
    """Normalize text conservatively to detect literal duplicates."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def decide_action(
    new_memory: Memory,
    similar_memories: list[tuple[Memory, float]],
    decision: str | None = None,
    target_id: str | None = None,
) -> tuple[str, str | None]:
    """Resolve safe cases automatically, otherwise apply the LLM decision."""
    normalized_content = normalize_text(new_memory.content)
    for memory, _ in similar_memories:
        if (
            memory.type == new_memory.type
            and normalize_text(memory.content) == normalized_content
        ):
            return ("IGNORE", memory.id)

    if decision == "ADD" and target_id is None:
        return ("ADD", None)

    candidate_ids = {memory.id for memory, _ in similar_memories}
    if decision in {"IGNORE", "EXPIRE"} and target_id in candidate_ids:
        return (decision, target_id)

    if decision is None and not similar_memories:
        return ("ADD", None)

    return ("ARBITRATION_REQUIRED", None)


def _serialize_candidates(
    similar_memories: list[tuple[Memory, float]],
) -> list[dict]:
    return [
        {
            "id": memory.id,
            "content": memory.content,
            "type": memory.type,
            "score": round(score, 3),
        }
        for memory, score in similar_memories
    ]


def _arbitration_report(
    new_memory: Memory,
    similar_memories: list[tuple[Memory, float]],
    error: str | None = None,
) -> dict:
    report = {
        "action": "ARBITRATION_REQUIRED",
        "written": False,
        "new_memory": {
            "content": new_memory.content,
            "type": new_memory.type,
            "source": new_memory.source,
            "confidence": new_memory.confidence,
        },
        "candidates": _serialize_candidates(similar_memories),
        "instruction": (
            "Compare the new memory with the candidates, then call save_memory "
            "again with the same arguments and decision='ADD' if they are "
            "compatible, decision='IGNORE' with target_id for a duplicate, or "
            "decision='EXPIRE' with target_id for a contradiction."
        ),
    }
    if error is not None:
        report["error"] = error
    return report


def process_memory(
    store: ParquetStore,
    new_memory: Memory,
    decision: str | None = None,
    target_id: str | None = None,
) -> dict:
    """Process an incoming memory and return a report for the calling LLM."""
    embedding = embeddings.embed(new_memory.content)
    similar_memories = find_similar_memories(store, embedding)
    action, validated_target = decide_action(
        new_memory,
        similar_memories,
        decision=decision,
        target_id=target_id,
    )

    if action == "ARBITRATION_REQUIRED":
        error = None
        if decision is not None:
            if decision not in ARBITRATION_ACTIONS:
                error = f"invalid decision: {decision}"
            elif decision == "ADD" and target_id is not None:
                error = "target_id must be omitted for ADD"
            elif decision in {"IGNORE", "EXPIRE"} and target_id is None:
                error = f"target_id is required for {decision}"
            else:
                error = "target_id is no longer an active similar candidate"
        return _arbitration_report(new_memory, similar_memories, error)

    if action == "IGNORE":
        return {
            "action": "IGNORE",
            "reason": f"duplicate of memory {validated_target}",
            "id": validated_target,
        }

    if action == "EXPIRE":
        old_memory = store.get(validated_target) if validated_target else None
        if old_memory is None or not old_memory.is_active:
            return _arbitration_report(
                new_memory,
                similar_memories,
                "the target is no longer active; evaluate the candidates again",
            )
        # End the old fact when the replacement becomes valid. With the default
        # timestamps this is equivalent to "now", while replayed histories keep
        # their original event time.
        old_memory.valid_until = new_memory.valid_from
        store.replace(old_memory)
        new_memory.supersedes = old_memory.id
        store.add(new_memory, embedding)
        return {
            "action": "EXPIRE",
            "expired": {"id": old_memory.id, "content": old_memory.content},
            "new_memory_id": new_memory.id,
        }

    store.add(new_memory, embedding)
    return {
        "action": "ADD",
        "new_memory_id": new_memory.id,
        "similar_memories": _serialize_candidates(similar_memories),
    }
