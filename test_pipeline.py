import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from urdwell import pipeline
from urdwell.models import Memory
from urdwell.storage import ParquetStore


class PipelineArbitrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = ParquetStore(Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def _add_existing(self, content: str, memory_type: str = "fact") -> Memory:
        memory = Memory(content=content, type=memory_type)
        self.store.add(memory, [1.0, 0.0])
        return memory

    def test_no_candidates_adds_memory_immediately(self):
        new_memory = Memory(content="The project uses SQLite", type="decision")
        with (
            patch("urdwell.pipeline.embeddings.embed", return_value=[1.0, 0.0]),
            patch("urdwell.pipeline.find_similar_memories", return_value=[]),
        ):
            report = pipeline.process_memory(self.store, new_memory)

        self.assertEqual(report["action"], "ADD")
        self.assertEqual(len(self.store.all()), 1)

    def test_add_many_persists_memories_and_embeddings_together(self):
        memories = [
            Memory(content="First", type="fact"),
            Memory(content="Second", type="decision"),
        ]

        self.store.add_many(memories, [[1.0, 0.0], [0.0, 1.0]])

        self.assertEqual(len(self.store.all()), 2)
        self.assertEqual(self.store.all_embeddings()[memories[1].id], [0.0, 1.0])

    def test_literal_duplicate_is_ignored(self):
        old_memory = self._add_existing("Mounir likes coffee", "preference")
        new_memory = Memory(content="  mounir LIKES coffee  ", type="preference")
        with (
            patch("urdwell.pipeline.embeddings.embed", return_value=[1.0, 0.0]),
            patch(
                "urdwell.pipeline.find_similar_memories",
                return_value=[(old_memory, 1.0)],
            ),
        ):
            report = pipeline.process_memory(self.store, new_memory)

        self.assertEqual(report["action"], "IGNORE")
        self.assertEqual(len(self.store.all()), 1)

    def test_compatible_preferences_wait_then_coexist(self):
        coffee = self._add_existing("Mounir likes coffee", "preference")
        tea_for_arbitration = Memory(
            content="Mounir likes tea",
            type="preference",
        )
        tea_to_add = Memory(content="Mounir likes tea", type="preference")
        similar_memories = [(coffee, 0.782)]

        with (
            patch("urdwell.pipeline.embeddings.embed", return_value=[1.0, 0.0]),
            patch(
                "urdwell.pipeline.find_similar_memories",
                return_value=similar_memories,
            ),
        ):
            pending = pipeline.process_memory(self.store, tea_for_arbitration)
            added = pipeline.process_memory(
                self.store,
                tea_to_add,
                decision="ADD",
            )

        self.assertEqual(pending["action"], "ARBITRATION_REQUIRED")
        self.assertFalse(pending["written"])
        self.assertEqual(added["action"], "ADD")
        self.assertEqual(len(self.store.all()), 2)

    def test_new_fact_expires_contradicted_fact_after_arbitration(self):
        old_memory = self._add_existing("x^2 + 1 = 0 has no solutions")
        fact_for_arbitration = Memory(
            content="x^2 + 1 = 0 has complex solutions",
            type="fact",
        )
        fact_to_add = Memory(
            content="x^2 + 1 = 0 has complex solutions",
            type="fact",
        )
        similar_memories = [(old_memory, 0.878)]

        with (
            patch("urdwell.pipeline.embeddings.embed", return_value=[1.0, 0.0]),
            patch(
                "urdwell.pipeline.find_similar_memories",
                return_value=similar_memories,
            ),
        ):
            pending = pipeline.process_memory(
                self.store,
                fact_for_arbitration,
            )
            replacement = pipeline.process_memory(
                self.store,
                fact_to_add,
                decision="EXPIRE",
                target_id=old_memory.id,
            )

        self.assertEqual(pending["action"], "ARBITRATION_REQUIRED")
        self.assertEqual(replacement["action"], "EXPIRE")
        self.assertFalse(self.store.get(old_memory.id).is_active)
        self.assertEqual(
            self.store.get(old_memory.id).valid_until,
            fact_to_add.valid_from,
        )
        self.assertEqual(
            self.store.get(fact_to_add.id).supersedes,
            old_memory.id,
        )
        self.assertEqual(len(self.store.all()), 1)

    def test_missing_target_does_not_add_memory(self):
        new_memory = Memory(
            content="x^2 + 1 = 0 has complex solutions",
            type="fact",
        )

        with (
            patch("urdwell.pipeline.embeddings.embed", return_value=[1.0, 0.0]),
            patch("urdwell.pipeline.find_similar_memories", return_value=[]),
        ):
            report = pipeline.process_memory(
                self.store,
                new_memory,
                decision="EXPIRE",
                target_id="missing-target",
            )

        self.assertEqual(report["action"], "ARBITRATION_REQUIRED")
        self.assertIn("error", report)
        self.assertEqual(self.store.all(), [])

    def test_non_candidate_target_is_rejected_without_writing(self):
        candidate = self._add_existing("Mounir likes coffee", "preference")
        unrelated = self._add_existing("The project uses JSON", "decision")
        new_memory = Memory(content="Mounir likes tea", type="preference")

        with (
            patch("urdwell.pipeline.embeddings.embed", return_value=[1.0, 0.0]),
            patch(
                "urdwell.pipeline.find_similar_memories",
                return_value=[(candidate, 0.782)],
            ),
        ):
            report = pipeline.process_memory(
                self.store,
                new_memory,
                decision="EXPIRE",
                target_id=unrelated.id,
            )

        self.assertEqual(report["action"], "ARBITRATION_REQUIRED")
        self.assertIn("error", report)
        self.assertEqual(len(self.store.all()), 2)

    def test_legacy_french_schema_is_loaded(self):
        legacy = Memory.from_dict(
            {
                "content": "Legacy memory",
                "type": "fait",
                "ecrit_le": "2026-01-01T00:00:00+00:00",
                "valide_depuis": "2026-01-01T00:00:00+00:00",
                "valide_jusqua": None,
                "confiance": 0.9,
                "remplace": None,
            }
        )

        self.assertEqual(legacy.type, "fact")
        self.assertEqual(legacy.confidence, 0.9)
        self.assertTrue(legacy.is_active)


if __name__ == "__main__":
    unittest.main()
