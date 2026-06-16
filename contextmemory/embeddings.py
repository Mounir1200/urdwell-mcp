"""Convert text to vectors for semantic comparison.

Three interchangeable backends produce 384-dimensional vectors for the same
multilingual MiniLM model, selected with ``CONTEXT_MEMORY_EMBEDDING_BACKEND``:

- ``fastembed`` (default): ONNX runtime, ~0.2 GB, no PyTorch. Recommended.
- ``transformer``: sentence-transformers + PyTorch (~2 GB). Optional extra.
- ``hashing``: deterministic, dependency-free, for tests and offline diagnostics.
"""

import hashlib
import math
import os
import re
import unicodedata

_DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_MODEL_ENV_VAR = "CONTEXT_MEMORY_EMBEDDING_MODEL"
_BACKEND_ENV_VAR = "CONTEXT_MEMORY_EMBEDDING_BACKEND"
_REVISION_ENV_VAR = "CONTEXT_MEMORY_EMBEDDING_REVISION"

_DEFAULT_BACKEND = "fastembed"
_VALID_BACKENDS = ("fastembed", "transformer", "hashing")
_HASHING_DIMENSIONS = 256

_fastembed_model = None
_transformer_model = None


def model_name() -> str:
    """Return the embedding model id, shared by the fastembed and torch backends."""
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

        _fastembed_model = TextEmbedding(model_name=model_name())
    return _fastembed_model


def _get_transformer_model():
    """Load sentence-transformers lazily on first use.

    Loading model weights from a remote repository is a supply-chain trust
    boundary. Set ``CONTEXT_MEMORY_EMBEDDING_REVISION`` to a specific commit
    hash to pin the weights and protect against a compromised upstream model.
    """
    global _transformer_model
    if _transformer_model is None:
        from sentence_transformers import SentenceTransformer

        revision = os.getenv(_REVISION_ENV_VAR) or None
        _transformer_model = SentenceTransformer(model_name(), revision=revision)
    return _transformer_model


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
    if backend == "fastembed":
        vector = next(iter(_get_fastembed_model().embed([text])))
        return _l2_normalize(vector.tolist())
    return _get_transformer_model().encode(text, normalize_embeddings=True).tolist()


def embed_many(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Convert multiple texts to normalized vectors efficiently."""
    backend = backend_name()
    if backend == "hashing":
        return [_hashing_embed(text) for text in texts]
    if backend == "fastembed":
        vectors = _get_fastembed_model().embed(texts, batch_size=batch_size)
        return [_l2_normalize(vector.tolist()) for vector in vectors]
    return _get_transformer_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity, guarding against zero-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
