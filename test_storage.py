import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from urdwell import storage
from urdwell.models import Memory
from urdwell.storage import ParquetStore


class DefaultDataDirTests(unittest.TestCase):
    def test_default_is_outside_the_installed_package(self):
        package_dir = Path(storage.__file__).resolve().parent
        default = storage.default_data_dir().resolve()
        self.assertFalse(
            str(default).startswith(str(package_dir)),
            "memories must not live inside the package; an upgrade would erase them",
        )

    def test_explicit_data_dir_argument_wins(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ParquetStore(Path(temp_dir) / "explicit")
            self.assertEqual(store.dir, Path(temp_dir) / "explicit")

    def test_environment_variable_overrides_default(self):
        previous = os.environ.get(storage.DATA_DIR_ENV_VAR)
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ[storage.DATA_DIR_ENV_VAR] = temp_dir
            try:
                store = ParquetStore()
                self.assertEqual(store.dir, Path(temp_dir))
            finally:
                if previous is None:
                    os.environ.pop(storage.DATA_DIR_ENV_VAR, None)
                else:
                    os.environ[storage.DATA_DIR_ENV_VAR] = previous

    def test_legacy_environment_variable_remains_supported(self):
        previous = os.environ.pop(storage.DATA_DIR_ENV_VAR, None)
        previous_legacy = os.environ.get(storage.LEGACY_DATA_DIR_ENV_VAR)
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ[storage.LEGACY_DATA_DIR_ENV_VAR] = temp_dir
            try:
                store = ParquetStore()
                self.assertEqual(store.dir, Path(temp_dir))
            finally:
                if previous is not None:
                    os.environ[storage.DATA_DIR_ENV_VAR] = previous
                if previous_legacy is None:
                    os.environ.pop(storage.LEGACY_DATA_DIR_ENV_VAR, None)
                else:
                    os.environ[storage.LEGACY_DATA_DIR_ENV_VAR] = previous_legacy

    def test_existing_pre_rename_default_store_is_reused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            current = Path(temp_dir) / "UrdWell"
            legacy = Path(temp_dir) / "ContextMemory"
            legacy.mkdir()
            with patch(
                "urdwell.storage._platform_data_dirs",
                return_value=(current, legacy),
            ):
                self.assertEqual(storage.default_data_dir(), legacy)

                current.mkdir()
                self.assertEqual(storage.default_data_dir(), current)


class ParquetStoreTests(unittest.TestCase):
    def test_memory_and_embedding_share_one_typed_parquet_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ParquetStore(Path(temp_dir))
            memory = Memory(content="La base utilise PostgreSQL.", type="fact")

            store.add(memory, [0.25, 0.5])

            self.assertEqual(store.get(memory.id), memory)
            self.assertEqual(store.get_embedding(memory.id), [0.25, 0.5])
            self.assertEqual(pq.read_table(store.memories_path).num_rows, 1)
            self.assertFalse((Path(temp_dir) / "memories.json").exists())
            self.assertFalse((Path(temp_dir) / "embeddings.json").exists())

    def test_agent_provenance_round_trips(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ParquetStore(Path(temp_dir))
            memory = Memory(content="A fact.", type="fact", agent="claude-code")

            store.add(memory, [0.1, 0.2])

            self.assertEqual(store.get(memory.id).agent, "claude-code")

    def test_archive_round_trip_uses_parquet(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ParquetStore(Path(temp_dir))
            store.append_archive("user", "Texte exact", "session-1")

            self.assertEqual(store.read_archive(1)[0]["content"], "Texte exact")
            self.assertEqual(pq.read_table(store.archive_path).num_rows, 1)

    def test_legacy_json_is_migrated_without_deleting_recovery_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            memory = Memory(content="Legacy memory", type="fact")
            memories_json = data_dir / "memories.json"
            embeddings_json = data_dir / "embeddings.json"
            archive_jsonl = data_dir / "archive.jsonl"
            memories_json.write_text(
                json.dumps([memory.to_dict()]),
                encoding="utf-8",
            )
            embeddings_json.write_text(
                json.dumps({memory.id: [0.1, 0.2]}),
                encoding="utf-8",
            )
            archive_jsonl.write_text(
                json.dumps(
                    {
                        "ts": "2026-01-01T00:00:00+00:00",
                        "role": "user",
                        "content": "Legacy archive",
                        "session": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            store = ParquetStore(data_dir)

            self.assertEqual(store.get(memory.id), memory)
            self.assertAlmostEqual(store.get_embedding(memory.id)[0], 0.1)
            self.assertEqual(store.read_archive(1)[0]["content"], "Legacy archive")
            self.assertTrue(store.memories_path.exists())
            self.assertTrue(store.archive_path.exists())
            self.assertTrue(memories_json.exists())
            self.assertTrue(archive_jsonl.exists())


if __name__ == "__main__":
    unittest.main()
