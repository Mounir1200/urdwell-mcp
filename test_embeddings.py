import math
import os
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

from urdwell import embeddings


class BackendSelectionTests(unittest.TestCase):
    def setUp(self):
        self.previous = os.environ.get("URDWELL_EMBEDDING_BACKEND")
        self.previous_legacy = os.environ.get("CONTEXT_MEMORY_EMBEDDING_BACKEND")

    def tearDown(self):
        if self.previous is None:
            os.environ.pop("URDWELL_EMBEDDING_BACKEND", None)
        else:
            os.environ["URDWELL_EMBEDDING_BACKEND"] = self.previous
        if self.previous_legacy is None:
            os.environ.pop("CONTEXT_MEMORY_EMBEDDING_BACKEND", None)
        else:
            os.environ["CONTEXT_MEMORY_EMBEDDING_BACKEND"] = self.previous_legacy

    def test_default_backend_is_fastembed(self):
        os.environ.pop("URDWELL_EMBEDDING_BACKEND", None)
        os.environ.pop("CONTEXT_MEMORY_EMBEDDING_BACKEND", None)
        self.assertEqual(embeddings.backend_name(), "fastembed")

    def test_invalid_backend_raises(self):
        os.environ["URDWELL_EMBEDDING_BACKEND"] = "bogus"
        with self.assertRaises(ValueError):
            embeddings.backend_name()

    def test_pytorch_backend_is_not_available(self):
        os.environ["URDWELL_EMBEDDING_BACKEND"] = "transformer"
        with self.assertRaises(ValueError):
            embeddings.backend_name()

    def test_hashing_embed_is_deterministic_and_normalized(self):
        os.environ["URDWELL_EMBEDDING_BACKEND"] = "hashing"
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

    def test_legacy_backend_environment_variable_is_supported(self):
        os.environ.pop("URDWELL_EMBEDDING_BACKEND", None)
        os.environ["CONTEXT_MEMORY_EMBEDDING_BACKEND"] = "hashing"
        self.assertEqual(embeddings.backend_name(), "hashing")

    def test_default_model_is_multilingual_paraphrase_minilm(self):
        self.assertEqual(
            embeddings._DEFAULT_MODEL_NAME,
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )

    def test_fastembed_mean_pooling_migration_warning_is_suppressed(self):
        class FakeTextEmbedding:
            def __init__(self, model_name, cache_dir=None):
                self.model_name = model_name
                self.cache_dir = cache_dir
                warnings.warn(
                    f"The model {model_name} now uses mean pooling instead of "
                    "CLS embedding. In order to preserve the previous behaviour, "
                    "consider either pinning fastembed version to 0.5.1 or using "
                    "`add_custom_model` functionality.",
                    UserWarning,
                    stacklevel=2,
                )

        previous_model = embeddings._fastembed_model
        embeddings._fastembed_model = None
        fake_fastembed = types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)
        try:
            with mock.patch.dict(sys.modules, {"fastembed": fake_fastembed}):
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    loaded = embeddings._get_fastembed_model()

            self.assertEqual(loaded.model_name, embeddings.model_name())
            self.assertEqual(captured, [])
        finally:
            embeddings._fastembed_model = previous_model

    def test_model_cache_dir_prefers_explicit_override(self):
        previous = os.environ.get("FASTEMBED_CACHE_PATH")
        os.environ["FASTEMBED_CACHE_PATH"] = os.path.join("custom", "models")
        try:
            self.assertEqual(
                embeddings._model_cache_dir(),
                Path("custom") / "models",
            )
        finally:
            if previous is None:
                os.environ.pop("FASTEMBED_CACHE_PATH", None)
            else:
                os.environ["FASTEMBED_CACHE_PATH"] = previous

    def test_model_cache_dir_defaults_to_durable_location(self):
        previous = os.environ.pop("FASTEMBED_CACHE_PATH", None)
        try:
            cache_dir = str(embeddings._model_cache_dir())
            self.assertFalse(
                cache_dir.startswith(tempfile.gettempdir()),
                "the model cache must survive reboots, not live under $TMPDIR",
            )
        finally:
            if previous is not None:
                os.environ["FASTEMBED_CACHE_PATH"] = previous


if __name__ == "__main__":
    unittest.main()
