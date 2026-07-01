"""Convert text to vectors for semantic comparison.

Two backends are available through ``URDWELL_EMBEDDING_BACKEND``:

- ``fastembed`` (default): ONNX runtime, ~0.2 GB, no PyTorch. Recommended.
- ``hashing``: deterministic, dependency-free, for tests and offline diagnostics.
"""

import hashlib
import math
import os
import platform
import re
import shutil
import unicodedata
import warnings
from pathlib import Path

_DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_MODEL_ENV_VAR = "URDWELL_EMBEDDING_MODEL"
_BACKEND_ENV_VAR = "URDWELL_EMBEDDING_BACKEND"
_LEGACY_MODEL_ENV_VAR = "CONTEXT_MEMORY_EMBEDDING_MODEL"
_LEGACY_BACKEND_ENV_VAR = "CONTEXT_MEMORY_EMBEDDING_BACKEND"
# fastembed reads this to locate its ONNX model cache. UrdWell honors it as a
# power-user override but otherwise supplies a durable default (see
# _model_cache_dir) instead of fastembed's volatile $TMPDIR fallback.
_FASTEMBED_CACHE_ENV_VAR = "FASTEMBED_CACHE_PATH"
_DEFAULT_BACKEND = "fastembed"
_VALID_BACKENDS = ("fastembed", "hashing")
_HASHING_DIMENSIONS = 256

_fastembed_model = None


def _configured_value(primary: str, legacy: str, default: str) -> str:
    """Read a renamed setting while preserving pre-0.3 environments."""
    return os.getenv(primary) or os.getenv(legacy) or default


def model_name() -> str:
    """Return the multilingual FastEmbed model id."""
    return _configured_value(
        _MODEL_ENV_VAR,
        _LEGACY_MODEL_ENV_VAR,
        _DEFAULT_MODEL_NAME,
    )


def backend_name() -> str:
    """Return the configured embedding backend."""
    backend = _configured_value(
        _BACKEND_ENV_VAR,
        _LEGACY_BACKEND_ENV_VAR,
        _DEFAULT_BACKEND,
    ).casefold()
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


def _platform_cache_dir() -> Path:
    """Return the per-user cache directory for downloaded models.

    Mirrors ``storage.default_data_dir`` but follows the OS *cache* convention:
    the model is re-downloadable, so it belongs in the cache tree rather than
    the data tree.
    """
    system = platform.system()
    if system == "Windows":
        base = os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        return Path(base) / "UrdWell" / "cache"
    if system == "Darwin":
        return Path.home() / "Library" / "Caches" / "UrdWell"
    base = os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "urdwell"


def _model_cache_dir() -> Path:
    """Return where fastembed should cache the ONNX model.

    An explicit ``FASTEMBED_CACHE_PATH`` always wins, so power users keep full
    control. Otherwise UrdWell anchors the cache in a durable per-user location
    instead of fastembed's default under ``$TMPDIR``: ``/tmp`` is commonly
    cleared on reboot, which deletes the model files while leaving the snapshot
    folder behind, and the model then fails to load with ``NO_SUCHFILE``.
    """
    explicit = os.getenv(_FASTEMBED_CACHE_ENV_VAR)
    if explicit:
        return Path(explicit)
    return _platform_cache_dir() / "fastembed"


def _instantiate_text_embedding(selected_model: str):
    """Build the FastEmbed model, silencing the mean-pooling migration warning.

    FastEmbed >= 0.6 intentionally uses the model's native mean pooling. The
    warning only matters when reusing CLS vectors produced by 0.5.1; UrdWell
    standardizes on mean pooling and documents that migration.
    """
    from fastembed import TextEmbedding

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
        return TextEmbedding(
            model_name=selected_model,
            cache_dir=str(_model_cache_dir()),
        )


def _load_fastembed_model():
    """Load the ONNX model, healing a corrupted cache once before giving up.

    A half-written cache — an interrupted download, or an older install whose
    ``/tmp`` copy was swept on reboot — loads with ``NO_SUCHFILE``. When UrdWell
    owns the cache location it purges and re-downloads once, so the user never
    has to clear it by hand. A cache the user pointed us at via
    ``FASTEMBED_CACHE_PATH`` is left untouched.
    """
    selected_model = model_name()
    try:
        return _instantiate_text_embedding(selected_model)
    except Exception:
        if os.getenv(_FASTEMBED_CACHE_ENV_VAR):
            raise
        shutil.rmtree(_model_cache_dir(), ignore_errors=True)
        return _instantiate_text_embedding(selected_model)


def _get_fastembed_model():
    """Load the ONNX model lazily on first use; cached by fastembed afterwards."""
    global _fastembed_model
    if _fastembed_model is None:
        _fastembed_model = _load_fastembed_model()
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
