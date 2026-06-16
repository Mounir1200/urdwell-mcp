"""Convert text to vectors for semantic comparison.

The multilingual MiniLM model maps text to 384-dimensional normalized vectors.
It is downloaded on first use and then cached by Hugging Face.
"""

import hashlib
import math
import os
import re
import unicodedata

_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_BACKEND_ENV_VAR = "CONTEXT_MEMORY_EMBEDDING_BACKEND"
_REVISION_ENV_VAR = "CONTEXT_MEMORY_EMBEDDING_REVISION"
_HASHING_DIMENSIONS = 256
_model = None


def backend_name() -> str:
    """Return the configured embedding backend."""
    backend = os.getenv(_BACKEND_ENV_VAR, "transformer").casefold()
    if backend not in {"transformer", "hashing"}:
        raise ValueError(
            f"invalid {_BACKEND_ENV_VAR}: {backend!r}; "
            "expected 'transformer' or 'hashing'"
        )
    return backend


def _get_model():
    """Load the dependency and model lazily on first use.

    Loading model weights from a remote repository is a supply-chain trust
    boundary. Set ``CONTEXT_MEMORY_EMBEDDING_REVISION`` to a specific commit
    hash to pin the weights and protect against a compromised upstream model.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        revision = os.getenv(_REVISION_ENV_VAR) or None
        _model = SentenceTransformer(_MODEL_NAME, revision=revision)
    return _model


def _hashing_embed(text: str) -> list[float]:
    """Create a deterministic local embedding for tests and offline diagnostics."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = re.findall(r"\w+", normalized)
    vector = [0.0] * _HASHING_DIMENSIONS
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % _HASHING_DIMENSIONS
        vector[index] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def embed(text: str) -> list[float]:
    """Convert text to a normalized vector with the configured backend."""
    if backend_name() == "hashing":
        return _hashing_embed(text)
    return _get_model().encode(text, normalize_embeddings=True).tolist()


def embed_many(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Convert multiple texts to normalized vectors efficiently."""
    if backend_name() == "hashing":
        return [_hashing_embed(text) for text in texts]
    return _get_model().encode(
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
