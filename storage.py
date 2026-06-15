"""Two-layer JSON persistence for ContextMemory.

Layer 1: ``data/archive.jsonl`` is an append-only verbatim event log.
Layer 2: ``data/memories.json`` stores structured, expirable memories.

Embedding vectors live in a separate file so the memory data remains readable.
"""

import json
import os
from pathlib import Path

from models import Memory, now_utc

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR_ENV_VAR = "CONTEXT_MEMORY_DATA_DIR"


class JsonStore:
    def __init__(self, data_dir: Path | None = None):
        configured_dir = data_dir or Path(os.getenv(DATA_DIR_ENV_VAR, DATA_DIR))
        self.dir = Path(configured_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.archive_path = self.dir / "archive.jsonl"
        self.memories_path = self.dir / "memories.json"
        self.legacy_memories_path = self.dir / "souvenirs.json"
        self.embeddings_path = self.dir / "embeddings.json"

    # ---------- Layer 1: raw append-only archive ----------

    def append_archive(self, role: str, content: str, session: str | None = None) -> None:
        """Append one entry without modifying any previous archive content."""
        entry = {"ts": now_utc(), "role": role, "content": content, "session": session}
        with open(self.archive_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_archive(self, last_n: int = 50) -> list[dict]:
        if not self.archive_path.exists():
            return []
        with open(self.archive_path, encoding="utf-8") as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines[-last_n:]]

    # ---------- Layer 2: structured memories ----------

    def _load(self) -> list[Memory]:
        path = (
            self.memories_path
            if self.memories_path.exists()
            else self.legacy_memories_path
        )
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return [Memory.from_dict(data) for data in json.load(f)]

    def _save(self, memories: list[Memory]) -> None:
        with open(self.memories_path, "w", encoding="utf-8") as f:
            json.dump(
                [memory.to_dict() for memory in memories],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def all(self, active_only: bool = True) -> list[Memory]:
        memories = self._load()
        if active_only:
            memories = [memory for memory in memories if memory.is_active]
        return memories

    def get(self, memory_id: str) -> Memory | None:
        for memory in self._load():
            if memory.id == memory_id:
                return memory
        return None

    def add(self, memory: Memory, embedding: list[float]) -> None:
        memories = self._load()
        memories.append(memory)
        self._save(memories)
        self._set_embedding(memory.id, embedding)

    def add_many(
        self,
        memories: list[Memory],
        embedding_vectors: list[list[float]],
    ) -> None:
        """Persist several memories and vectors in one filesystem transaction."""
        if len(memories) != len(embedding_vectors):
            raise ValueError("memories and embedding_vectors must have equal length")
        stored_memories = self._load()
        stored_memories.extend(memories)
        self._save(stored_memories)
        stored_embeddings = self._load_embeddings()
        stored_embeddings.update(
            {
                memory.id: vector
                for memory, vector in zip(memories, embedding_vectors)
            }
        )
        with open(self.embeddings_path, "w", encoding="utf-8") as f:
            json.dump(stored_embeddings, f)

    def replace(self, memory: Memory) -> None:
        """Rewrite an existing memory with the same ID."""
        memories = self._load()
        for index, current in enumerate(memories):
            if current.id == memory.id:
                memories[index] = memory
                self._save(memories)
                return
        raise KeyError(f"memory not found: {memory.id}")

    # ---------- Embeddings ----------

    def _load_embeddings(self) -> dict[str, list[float]]:
        if not self.embeddings_path.exists():
            return {}
        with open(self.embeddings_path, encoding="utf-8") as f:
            return json.load(f)

    def _set_embedding(self, memory_id: str, embedding: list[float]) -> None:
        embeddings = self._load_embeddings()
        embeddings[memory_id] = embedding
        with open(self.embeddings_path, "w", encoding="utf-8") as f:
            json.dump(embeddings, f)

    def get_embedding(self, memory_id: str) -> list[float] | None:
        return self._load_embeddings().get(memory_id)

    def all_embeddings(self) -> dict[str, list[float]]:
        """Return all stored vectors with a single filesystem read."""
        return self._load_embeddings()
