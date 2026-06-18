import unittest

from contextmemory import ranking
from contextmemory.models import Memory


class Bm25Tests(unittest.TestCase):
    def test_no_documents_returns_empty(self):
        self.assertEqual(ranking.bm25_scores("rooibos", []), [])

    def test_document_with_query_term_outscores_one_without(self):
        scores = ranking.bm25_scores("rooibos", ["le rooibos de Mounir", "le café"])
        self.assertGreater(scores[0], 0.0)
        self.assertEqual(scores[1], 0.0)


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_consensus_outranks_a_single_strong_list(self):
        fused = ranking.reciprocal_rank_fusion(["a", "b", "c"], ["c", "a", "b"])
        # "a" is 1st then 2nd; "c" is 3rd then 1st; "b" is always low.
        self.assertGreater(fused["a"], fused["c"])
        self.assertGreater(fused["c"], fused["b"])

    def test_item_present_in_one_ranking_is_still_scored(self):
        fused = ranking.reciprocal_rank_fusion(["a"], ["b"])
        self.assertIn("a", fused)
        self.assertIn("b", fused)


class HybridRankTests(unittest.TestCase):
    def _candidate(self, content, vector):
        return (Memory(content=content, type="preference"), vector)

    def test_abstains_when_best_cosine_is_below_floor(self):
        # The lexical leg is strong (exact term) but cosine says nothing is
        # relevant, so abstention must win.
        rooibos = self._candidate("Rooibos", [0.0, 1.0])
        other = self._candidate("autre chose", [0.4, 0.917])
        ranked = ranking.hybrid_rank(
            "Rooibos", [1.0, 0.0], [rooibos, other], k=5, cosine_floor=0.55
        )
        self.assertEqual(ranked, [])

    def test_lexical_match_is_rescued_above_a_higher_cosine_document(self):
        exact = self._candidate("Rooibos", [0.30, 0.95])        # cosine ~0.30
        semantic_top = self._candidate("boissons chaudes", [0.99, 0.14])  # cosine ~0.99
        distractor = self._candidate("café noir", [0.90, 0.44])  # cosine ~0.90

        ranked = ranking.hybrid_rank(
            "Rooibos",
            [1.0, 0.0],
            [exact, semantic_top, distractor],
            k=5,
            cosine_floor=0.55,
        )
        ranked_ids = [memory.id for memory, _ in ranked]

        # The top cosine clears the floor, so we answer rather than abstain.
        self.assertNotEqual(ranked, [])
        # The exact-term doc surfaces despite a sub-floor cosine (decoupling)...
        self.assertIn(exact[0].id, ranked_ids)
        # ...and is rescued above the higher-cosine distractor by the fusion.
        self.assertLess(ranked_ids.index(exact[0].id), ranked_ids.index(distractor[0].id))

    def test_returned_score_is_the_cosine_not_the_fusion_score(self):
        exact = self._candidate("Rooibos", [0.30, 0.95])
        semantic_top = self._candidate("boissons chaudes", [0.99, 0.14])
        ranked = ranking.hybrid_rank(
            "Rooibos", [1.0, 0.0], [exact, semantic_top], k=5, cosine_floor=0.55
        )
        scores = {memory.id: score for memory, score in ranked}
        self.assertAlmostEqual(scores[exact[0].id], 0.30, places=2)


if __name__ == "__main__":
    unittest.main()
