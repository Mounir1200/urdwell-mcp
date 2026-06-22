"""Two-layer Parquet persistence for UrdWell.

Layer 1: ``archive.parquet`` stores the verbatim exchange archive.
Layer 2: ``memories.parquet`` stores structured memories and their embeddings.

Each write creates a complete sibling Parquet file and atomically replaces the
previous one. A process-wide lock serializes read-modify-write cycles because
FastMCP runs synchronous tools in worker threads. Existing JSON/JSONL stores are
migrated automatically on first use and left untouched as recovery copies.
"""

import json
import os
import platform
import tempfile
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from urdwell.models import Memory, now_utc

DATA_DIR_ENV_VAR = "URDWELL_DATA_DIR"
LEGACY_DATA_DIR_ENV_VAR = "CONTEXT_MEMORY_DATA_DIR"

_ARCHIVE_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.string(), nullable=False),
        pa.field("role", pa.string(), nullable=False),
        pa.field("content", pa.string(), nullable=False),
        pa.field("session", pa.string()),
    ]
)

_MEMORY_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("content", pa.string(), nullable=False),
        pa.field("type", pa.string(), nullable=False),
        pa.field("source", pa.string()),
        pa.field("user", pa.string(), nullable=False),
        pa.field("written_at", pa.string(), nullable=False),
        pa.field("valid_from", pa.string(), nullable=False),
        pa.field("valid_until", pa.string()),
        pa.field("confidence", pa.float64(), nullable=False),
        pa.field("supersedes", pa.string()),
        pa.field("embedding", pa.list_(pa.float32())),
    ]
)


def _platform_data_dirs() -> tuple[Path, Path]:
    """Return the new and pre-0.3 default data directories for this platform."""
    system = platform.system()
    if system == "Windows":
        base = os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        return (Path(base) / "UrdWell", Path(base) / "ContextMemory")
    if system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
        return (base / "UrdWell", base / "ContextMemory")
    base = os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return (Path(base) / "urdwell", Path(base) / "contextmemory")


def default_data_dir() -> Path:
    """Return the stable data directory, reusing a pre-0.3 store when present."""
    current, legacy = _platform_data_dirs()
    if not current.exists() and legacy.exists():
        return legacy
    return current


def _atomic_write_parquet(
    path: Path,
    rows: list[dict],
    schema: pa.Schema,
) -> None:
    """Write rows to Parquet without ever exposing a partial destination file."""
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    os.close(handle)
    temporary = Path(temp_name)
    try:
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class ParquetStore:
    """Persist the archive, memories, and vectors in typed Parquet files."""

    def __init__(self, data_dir: Path | None = None):
        configured_dir = (
            data_dir
            or os.getenv(DATA_DIR_ENV_VAR)
            or os.getenv(LEGACY_DATA_DIR_ENV_VAR)
            or default_data_dir()
        )
        self.dir = Path(configured_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        self.archive_path = self.dir / "archive.parquet"
        self.memories_path = self.dir / "memories.parquet"

        self.legacy_archive_path = self.dir / "archive.jsonl"
        self.legacy_memories_paths = (
            self.dir / "memories.json",
            self.dir / "souvenirs.json",
        )
        self.legacy_embeddings_path = self.dir / "embeddings.json"

        # Reentrant so a locked public method can call another locked helper.
        self._lock = threading.RLock()
        self._migrate_legacy_files()

    # ---------- Legacy migration ----------

    def _migrate_legacy_files(self) -> None:
        """Materialize old JSON data as Parquet, preserving the source files."""
        with self._lock:
            if not self.archive_path.exists() and self.legacy_archive_path.exists():
                _atomic_write_parquet(
                    self.archive_path,
                    self._load_legacy_archive(),
                    _ARCHIVE_SCHEMA,
                )
            if not self.memories_path.exists():
                legacy_path = next(
                    (path for path in self.legacy_memories_paths if path.exists()),
                    None,
                )
                if legacy_path is not None:
                    _atomic_write_parquet(
                        self.memories_path,
                        self._load_legacy_memories(legacy_path),
                        _MEMORY_SCHEMA,
                    )

    def _load_legacy_archive(self) -> list[dict]:
        with open(self.legacy_archive_path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _load_legacy_memories(self, path: Path) -> list[dict]:
        with open(path, encoding="utf-8") as handle:
            memories = [Memory.from_dict(data) for data in json.load(handle)]

        embeddings: dict[str, list[float]] = {}
        if self.legacy_embeddings_path.exists():
            with open(self.legacy_embeddings_path, encoding="utf-8") as handle:
                embeddings = json.load(handle)

        return [
            self._memory_record(memory, embeddings.get(memory.id))
            for memory in memories
        ]

    # ---------- Layer 1: raw archive ----------

    def _load_archive(self) -> list[dict]:
        if not self.archive_path.exists():
            return []
        return pq.read_table(self.archive_path).to_pylist()

    def append_archive(self, role: str, content: str, session: str | None = None) -> None:
        """Append an archive entry and atomically publish the new Parquet file."""
        entry = {"ts": now_utc(), "role": role, "content": content, "session": session}
        with self._lock:
            entries = self._load_archive()
            entries.append(entry)
            _atomic_write_parquet(self.archive_path, entries, _ARCHIVE_SCHEMA)

    def read_archive(self, last_n: int = 50) -> list[dict]:
        """Return the last ``last_n`` archive entries."""
        if last_n <= 0 or not self.archive_path.exists():
            return []
        table = pq.read_table(self.archive_path)
        offset = max(table.num_rows - last_n, 0)
        return table.slice(offset).to_pylist()

    # ---------- Layer 2: structured memories + embeddings ----------

    @staticmethod
    def _memory_record(
        memory: Memory,
        embedding: list[float] | None,
    ) -> dict:
        return {**memory.to_dict(), "embedding": embedding}

    @staticmethod
    def _record_memory(record: dict) -> Memory:
        fields = {key: value for key, value in record.items() if key != "embedding"}
        return Memory.from_dict(fields)

    def _load_records(self) -> list[dict]:
        if not self.memories_path.exists():
            return []
        return pq.read_table(self.memories_path).to_pylist()

    def _save_records(self, records: list[dict]) -> None:
        _atomic_write_parquet(self.memories_path, records, _MEMORY_SCHEMA)

    def all(self, active_only: bool = True) -> list[Memory]:
        memories = [self._record_memory(record) for record in self._load_records()]
        if active_only:
            memories = [memory for memory in memories if memory.is_active]
        return memories

    def get(self, memory_id: str) -> Memory | None:
        for record in self._load_records():
            if record["id"] == memory_id:
                return self._record_memory(record)
        return None

    def add(self, memory: Memory, embedding: list[float]) -> None:
        with self._lock:
            records = self._load_records()
            records.append(self._memory_record(memory, embedding))
            self._save_records(records)

    def add_many(
        self,
        memories: list[Memory],
        embedding_vectors: list[list[float]],
    ) -> None:
        """Persist several memories and vectors in one atomic file replacement."""
        if len(memories) != len(embedding_vectors):
            raise ValueError("memories and embedding_vectors must have equal length")
        with self._lock:
            records = self._load_records()
            records.extend(
                self._memory_record(memory, vector)
                for memory, vector in zip(memories, embedding_vectors)
            )
            self._save_records(records)

    def replace(self, memory: Memory) -> None:
        """Rewrite an existing memory while preserving its embedding."""
        with self._lock:
            records = self._load_records()
            for index, current in enumerate(records):
                if current["id"] == memory.id:
                    records[index] = self._memory_record(
                        memory,
                        current.get("embedding"),
                    )
                    self._save_records(records)
                    return
            raise KeyError(f"memory not found: {memory.id}")

    def get_embedding(self, memory_id: str) -> list[float] | None:
        for record in self._load_records():
            if record["id"] == memory_id:
                return record.get("embedding")
        return None

    def all_embeddings(self) -> dict[str, list[float]]:
        """Return every available vector with one Parquet read."""
        return {
            record["id"]: record["embedding"]
            for record in self._load_records()
            if record.get("embedding") is not None
        }


# Import compatibility for clients that used the pre-0.3 class name. The alias
# writes Parquet exclusively; it does not retain the old JSON implementation.
JsonStore = ParquetStore
