import unittest

from benchmarks.longmemeval.run_retrieval import (
    build_case,
    calibrate_thresholds,
    retrieval_metrics,
    select_entries,
)


class LongMemEvalHarnessTests(unittest.TestCase):
    def test_turn_case_indexes_user_turns_and_marks_targets(self):
        entry = {
            "question_id": "q1",
            "question_type": "single-session-user",
            "question": "What database is used?",
            "haystack_session_ids": ["answer_session"],
            "haystack_dates": ["2026/01/01"],
            "haystack_sessions": [
                [
                    {
                        "role": "user",
                        "content": "The project uses PostgreSQL.",
                        "has_answer": True,
                    },
                    {
                        "role": "assistant",
                        "content": "Understood.",
                        "has_answer": False,
                    },
                ]
            ],
        }

        case = build_case(entry, "turn")

        self.assertEqual(len(case.corpus), 1)
        self.assertEqual(case.correct_ids, {"answer_session_1"})
        self.assertEqual(case.corpus[0], "The project uses PostgreSQL.")

        dated_case = build_case(entry, "turn", include_date=True)
        self.assertEqual(
            dated_case.corpus[0],
            "Session date: 2026/01/01\nThe project uses PostgreSQL.",
        )

    def test_retrieval_metrics_match_expected_ranking(self):
        metrics = retrieval_metrics(
            ["wrong", "answer_1", "answer_2"],
            {"answer_1", "answer_2"},
            k=2,
        )

        self.assertEqual(metrics["recall_any@2"], 1.0)
        self.assertEqual(metrics["recall_all@2"], 0.0)
        self.assertGreater(metrics["ndcg_any@2"], 0.0)

    def test_selection_is_reproducible(self):
        entries = [
            {
                "question_id": f"q{i}",
                "question_type": "type-a" if i % 2 else "type-b",
            }
            for i in range(20)
        ]

        first = select_entries(entries, limit=8, seed=7)
        second = select_entries(entries, limit=8, seed=7)

        self.assertEqual(
            [entry["question_id"] for entry in first],
            [entry["question_id"] for entry in second],
        )

    def test_threshold_calibration_balances_recall_and_abstention(self):
        rows = calibrate_thresholds(
            answerable_rankings=[
                ([("answer", 0.45), ("wrong", 0.3)], {"answer"}),
            ],
            abstention_top_scores=[0.35],
            thresholds=(0.3, 0.4, 0.5),
        )

        self.assertEqual(rows[0]["recall_any@5"], 1.0)
        self.assertEqual(rows[0]["abstention_specificity"], 0.0)
        self.assertEqual(rows[1]["balanced_score"], 1.0)
        self.assertEqual(rows[2]["recall_any@5"], 0.0)


if __name__ == "__main__":
    unittest.main()
