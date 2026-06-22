"""Domain model for memories stored by UrdWell.

The model is bi-temporal:
  - ``written_at`` records when the memory was stored and never changes.
  - ``valid_from`` and ``valid_until`` describe when the fact is true.

When a fact changes, the old memory is expired instead of deleted. A new
memory points to the old one through ``supersedes``, preserving history.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import uuid

# Classify content when it is written.
VALID_MEMORY_TYPES = {
    "fact",
    "preference",
    "decision",
    "temporary_state",
}

LEGACY_MEMORY_TYPES = {
    "fait": "fact",
    "preference": "preference",
    "decision": "decision",
    "etat_temporaire": "temporary_state",
}


def now_utc() -> str:
    """Return an ISO 8601 UTC timestamp with second precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Memory:
    content: str
    type: str
    source: str | None = None

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user: str = "default"

    written_at: str = field(default_factory=now_utc)

    valid_from: str = field(default_factory=now_utc)
    valid_until: str | None = None

    confidence: float = 0.8
    supersedes: str | None = None

    def __post_init__(self):
        if self.type not in VALID_MEMORY_TYPES:
            raise ValueError(
                f"invalid memory type: {self.type!r} "
                f"(expected one of {sorted(VALID_MEMORY_TYPES)})"
            )

    @property
    def is_active(self) -> bool:
        """A memory remains active until it receives an expiration time."""
        return self.valid_until is None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """Load current data and transparently upgrade the legacy French schema."""
        converted = dict(data)
        legacy_fields = {
            "ecrit_le": "written_at",
            "valide_depuis": "valid_from",
            "valide_jusqua": "valid_until",
            "confiance": "confidence",
            "remplace": "supersedes",
        }
        for old_name, new_name in legacy_fields.items():
            legacy_value = converted.pop(old_name, None)
            if new_name not in converted and legacy_value is not None:
                converted[new_name] = legacy_value

        memory_type = converted.get("type")
        if not isinstance(memory_type, str):
            raise ValueError("memory type must be a string")
        converted["type"] = LEGACY_MEMORY_TYPES.get(memory_type, memory_type)
        return cls(**converted)
