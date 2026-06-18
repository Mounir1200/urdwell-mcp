import json
import os
from pathlib import Path
import tempfile
import unittest

from benchmarks.longmemeval.llm_client import ChatResponse
from benchmarks.longmemeval.run_end_to_end import (
    answer_case,
    answer_without_context_memory,
    estimate,
    ingest_case,
    ingest_verbatim_case,
    official_judge_prompt,
)
from contextmemory.storage import ParquetStore


class FakeClient:
    model = "fake-model"

    def complete(
        self,
        prompt,
        *,
        max_tokens,
        temperature=0.0,
        json_mode=False,
    ):
        if "Extract durable memories" in prompt:
            city = "Paris" if "Paris" in prompt else "London"
            return ChatResponse(
                json.dumps(
                    {
                        "memories": [
                            {
                                "content": f"The user lives in {city}.",
                                "type": "fact",
                                "confidence": 0.95,
                            }
                        ]
                    }
                ),
                prompt_tokens=10,
                completion_tokens=5,
            )
        if "Resolve a long-term memory consolidation decision" in prompt:
            candidates = json.loads(prompt.split("Candidates:\n", 1)[1])
            return ChatResponse(
                json.dumps(
                    {
                        "decision": "EXPIRE",
                        "target_id": candidates[0]["id"],
                    }
                ),
                prompt_tokens=8,
                completion_tokens=3,
            )
        if "Answer the question" in prompt:
            return ChatResponse(
                "The user lives in London.",
                prompt_tokens=12,
                completion_tokens=5,
            )
        raise AssertionError(f"unexpected prompt: {prompt[:100]}")


class LongMemEvalEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.previous_backend = os.environ.get(
            "CONTEXT_MEMORY_EMBEDDING_BACKEND"
        )
        os.environ["CONTEXT_MEMORY_EMBEDDING_BACKEND"] = "hashing"
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()
        if self.previous_backend is None:
            os.environ.pop("CONTEXT_MEMORY_EMBEDDING_BACKEND", None)
        else:
            os.environ["CONTEXT_MEMORY_EMBEDDING_BACKEND"] = self.previous_backend

    def _entry(self):
        return {
            "question_id": "q1",
            "question_type": "knowledge-update",
            "question": "Where does the user live now?",
            "answer": "London",
            "question_date": "2026/02/01 (Sun) 12:00",
            "haystack_session_ids": ["new", "old"],
            "haystack_dates": [
                "2026/01/20 (Tue) 09:00",
                "2026/01/01 (Thu) 09:00",
            ],
            "haystack_sessions": [
                [{"role": "user", "content": "I moved to London."}],
                [{"role": "user", "content": "I live in Paris."}],
            ],
        }

    def test_ingestion_preserves_event_time_and_answers_from_memory(self):
        store = ParquetStore(Path(self.temp_dir.name) / "store")
        client = FakeClient()

        logs, usage = ingest_case(
            self._entry(),
            store,
            client,
            Path(self.temp_dir.name) / "cache",
        )

        memories = store.all(active_only=False)
        old_memory = next(memory for memory in memories if "Paris" in memory.content)
        new_memory = next(memory for memory in memories if "London" in memory.content)
        self.assertEqual(old_memory.valid_from, "2026/01/01 (Thu) 09:00")
        self.assertEqual(old_memory.valid_until, "2026/01/20 (Tue) 09:00")
        self.assertTrue(new_memory.is_active)
        self.assertEqual(len(logs), 2)
        self.assertGreaterEqual(usage["requests"], 3)

        hypothesis, retrieved, _, response = answer_case(
            self._entry(),
            store,
            client,
            top_k=5,
            threshold=0.0,
            include_expired=True,
        )

        self.assertEqual(hypothesis, "The user lives in London.")
        self.assertEqual(len(retrieved), 2)
        self.assertIsNotNone(response)

    def test_estimate_deduplicates_identical_sessions(self):
        entry = self._entry()
        duplicate = dict(entry)
        duplicate["question_id"] = "q2"

        workload = estimate([entry, duplicate], "context-memory", "llm")

        self.assertEqual(workload["instances"], 2)
        self.assertEqual(workload["sessions"], 4)
        self.assertEqual(workload["unique_sessions"], 2)
        self.assertEqual(workload["minimum_llm_requests"], 6)
        self.assertGreater(workload["approx_unique_source_tokens"], 0)

        raw_workload = estimate([entry, duplicate], "raw-history", "llm")
        self.assertEqual(raw_workload["minimum_llm_requests"], 4)

        verbatim_workload = estimate(
            [entry, duplicate],
            "context-memory",
            "verbatim",
        )
        self.assertEqual(verbatim_workload["minimum_llm_requests"], 4)

    def test_verbatim_ingestion_stores_dated_rounds_in_bulk(self):
        store = ParquetStore(Path(self.temp_dir.name) / "verbatim-store")

        logs, usage = ingest_verbatim_case(self._entry(), store)

        memories = store.all()
        self.assertEqual(len(memories), 2)
        self.assertTrue(all("Session date:" in memory.content for memory in memories))
        self.assertEqual(len(store.all_embeddings()), 2)
        self.assertEqual(sum(item["stored_rounds"] for item in logs), 2)
        self.assertEqual(usage["requests"], 0)

    def test_raw_history_baseline_answers_without_memory_store(self):
        hypothesis, prompt, response = answer_without_context_memory(
            self._entry(),
            FakeClient(),
            "raw-history",
        )

        self.assertEqual(hypothesis, "The user lives in London.")
        self.assertIn("I moved to London", prompt)
        self.assertIsNotNone(response)

    def test_official_judge_prompt_handles_abstention(self):
        entry = self._entry()
        entry["question_id"] = "q1_abs"

        prompt = official_judge_prompt(
            entry,
            "I cannot determine that from the available memories.",
        )

        self.assertIn("unanswerable", prompt)
        self.assertIn("yes or no only", prompt)


if __name__ == "__main__":
    unittest.main()
