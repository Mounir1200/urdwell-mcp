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

# A memory is either shared across agents ("global") or private to the agent
# that wrote it ("agent"), which stops tool-specific memories from colliding.
VALID_SCOPES = {"global", "agent"}

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
    # Which agent wrote this memory, stamped at write time from the server's
    # configured identity. None when the writer is unknown (manual or pre-0.3).
    agent: str | None = None
    # "global" memories are shared across agents; "agent" memories stay private
    # to their author, matched and arbitrated only within that same agent.
    scope: str = "global"

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
        if self.scope not in VALID_SCOPES:
            raise ValueError(
                f"invalid scope: {self.scope!r} "
                f"(expected one of {sorted(VALID_SCOPES)})"
            )

    def visible_to(self, agent: str | None) -> bool:
        """Whether ``agent`` may retrieve or arbitrate against this memory.

        Global memories are shared; agent-scoped memories stay with their author.
        """
        return self.scope == "global" or self.agent == agent

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
        # A null scope (pre-scope rows, or a column null-filled on migration)
        # means the default shared scope.
        if converted.get("scope") is None:
            converted.pop("scope", None)
        return cls(**converted)
