"""Convert text to vectors for semantic comparison.

Two backends are available through ``CONTEXT_MEMORY_EMBEDDING_BACKEND``:

- ``fastembed`` (default): ONNX runtime, ~0.2 GB, no PyTorch. Recommended.
- ``hashing``: deterministic, dependency-free, for tests and offline diagnostics.
"""

import hashlib
import math
import os
import re
import unicodedata
import warnings

_DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_MODEL_ENV_VAR = "CONTEXT_MEMORY_EMBEDDING_MODEL"
_BACKEND_ENV_VAR = "CONTEXT_MEMORY_EMBEDDING_BACKEND"
_DEFAULT_BACKEND = "fastembed"
_VALID_BACKENDS = ("fastembed", "hashing")
_HASHING_DIMENSIONS = 256

_fastembed_model = None


def model_name() -> str:
    """Return the multilingual FastEmbed model id."""
    return os.getenv(_MODEL_ENV_VAR, _DEFAULT_MODEL_NAME)


def backend_name() -> str:
    """Return the configured embedding backend."""
    backend = os.getenv(_BACKEND_ENV_VAR, _DEFAULT_BACKEND).casefold()
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"invalid {_BACKEND_ENV_VAR}: {backend!r}; "
            f"expected one of {_VALID_BACKENDS}"
        )
    return backend


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _get_fastembed_model():
    """Load the ONNX model lazily on first use; cached by fastembed afterwards."""
    global _fastembed_model
    if _fastembed_model is None:
        from fastembed import TextEmbedding

        selected_model = model_name()
        # FastEmbed >= 0.6 intentionally uses the model's native mean pooling.
        # The warning only matters when reusing CLS vectors produced by 0.5.1;
        # ContextMemory standardizes on mean pooling and documents that migration.
        mean_pooling_warning = (
            rf"The model {re.escape(selected_model)} now uses mean pooling "
            r"instead of CLS embedding\..*"
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=mean_pooling_warning,
                category=UserWarning,
            )
            _fastembed_model = TextEmbedding(model_name=selected_model)
    return _fastembed_model


def _hashing_embed(text: str) -> list[float]:
    """Create a deterministic local embedding for tests and offline diagnostics."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = re.findall(r"\w+", normalized)
    vector = [0.0] * _HASHING_DIMENSIONS
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % _HASHING_DIMENSIONS
        vector[index] += 1.0
    return _l2_normalize(vector)


def embed(text: str) -> list[float]:
    """Convert text to a normalized vector with the configured backend."""
    backend = backend_name()
    if backend == "hashing":
        return _hashing_embed(text)
    vector = next(iter(_get_fastembed_model().embed([text])))
    return _l2_normalize(vector.tolist())


def embed_many(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Convert multiple texts to normalized vectors efficiently."""
    backend = backend_name()
    if backend == "hashing":
        return [_hashing_embed(text) for text in texts]
    vectors = _get_fastembed_model().embed(texts, batch_size=batch_size)
    return [_l2_normalize(vector.tolist()) for vector in vectors]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity, guarding against zero-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
