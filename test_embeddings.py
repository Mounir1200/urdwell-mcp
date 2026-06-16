import math
import os
import unittest

from contextmemory import embeddings


class BackendSelectionTests(unittest.TestCase):
    def setUp(self):
        self.previous = os.environ.get("CONTEXT_MEMORY_EMBEDDING_BACKEND")

    def tearDown(self):
        if self.previous is None:
            os.environ.pop("CONTEXT_MEMORY_EMBEDDING_BACKEND", None)
        else:
            os.environ["CONTEXT_MEMORY_EMBEDDING_BACKEND"] = self.previous

    def test_default_backend_is_fastembed(self):
        os.environ.pop("CONTEXT_MEMORY_EMBEDDING_BACKEND", None)
        self.assertEqual(embeddings.backend_name(), "fastembed")

    def test_invalid_backend_raises(self):
        os.environ["CONTEXT_MEMORY_EMBEDDING_BACKEND"] = "bogus"
        with self.assertRaises(ValueError):
            embeddings.backend_name()

    def test_hashing_embed_is_deterministic_and_normalized(self):
        os.environ["CONTEXT_MEMORY_EMBEDDING_BACKEND"] = "hashing"
        first = embeddings.embed("Mounir likes coffee")
        second = embeddings.embed("Mounir likes coffee")

        self.assertEqual(first, second)
        self.assertAlmostEqual(
            math.sqrt(sum(value * value for value in first)), 1.0, places=6
        )

    def test_cosine_similarity_guards_zero_vectors(self):
        self.assertEqual(embeddings.cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)
        self.assertAlmostEqual(
            embeddings.cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0
        )


if __name__ == "__main__":
    unittest.main()
