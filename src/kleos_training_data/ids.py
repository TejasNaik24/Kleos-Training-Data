"""Stable, content-derived example identity.

Position-based ids (``example-000001``) are forbidden: they change when anything
before them changes, so two dataset versions cannot be diffed and a leakage
report cannot be read across releases.

An id here is a function of the example's *semantic content* and nothing else::

    kx-npr-3f9a1c8e2b7d0456
    │  │   └─ first 16 hex of sha256 over the canonicalized payload
    │  └───── task code, so an id is triageable without a lookup
    └──────── namespace, so an id is recognizable out of context

23 characters, first character alphanumeric, every character in
``[A-Za-z0-9._:-]`` — comfortably inside the public contract's
``^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$`` with 105 characters of headroom.

What counts as content
----------------------
:func:`canonicalize` keeps ``task``, the messages, and the variation axes. It
drops ``id``, ``version``, ``domain`` (derivable from the axes) and **all** of
``metadata``.

Dropping metadata is the important decision. Provenance must not change
identity: re-reviewing an example, correcting its ``notes``, or promoting it in a
later run leaves the id alone. Conversely, changing one character of an assistant
turn *does* change it — which is precisely what makes a signed review
non-transferable, and what lets gate G08 detect an example edited after approval.

Relationship to the public content hash
---------------------------------------
``TrainingExample.content_hash()`` digests the flat conversation text only. It is
the public notion of identity, used for duplicate and leakage parity, and this
module does not use it. The two are deliberately named differently; confusing
them would silently break either deduplication or identity.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from kleos_training_data.contract.constants import TASK_CODES
from kleos_training_data.contract.schemas import ID_PATTERN
from kleos_training_data.errors import ContractViolationError
from kleos_training_data.hashing import canonical_bytes, sha256_hex

#: Namespace prefix. Short on purpose — it appears in every leakage report row.
ID_PREFIX: Final[str] = "kx"

#: Task code used when a payload names a task with no registered short code.
#: Only reachable for an unregistered task, which the contract rejects anyway.
UNKNOWN_TASK_CODE: Final[str] = "unk"

#: Digest length in hex characters. 16 hex = 64 bits.
DEFAULT_DIGEST_CHARS: Final[int] = 16

#: Lengths tried in order when a shorter id is already claimed by different
#: content. Escalation is recorded in the ledger so it is reproducible.
DIGEST_ESCALATION: Final[tuple[int, ...]] = (16, 24, 32, 48, 64)

_TRAILING_WS = re.compile(r"[ \t]+$", flags=re.MULTILINE)


def _normalize_content(text: str) -> str:
    """Normalize message text for identity purposes.

    Line endings, trailing whitespace and unicode composition are presentation,
    not content. Two files differing only in how a text editor saved them must
    produce the same id, or a round-trip through a different machine silently
    forks every example.
    """
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    trimmed = _TRAILING_WS.sub("", unified)
    return unicodedata.normalize("NFC", trimmed).strip()


def canonicalize(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project a payload down to the parts that constitute its identity.

    Args:
        payload: A training-example-shaped mapping. Need not be valid yet — an
            id is minted before promotion, and the schema gate runs separately.

    Returns:
        A dict containing exactly ``task``, ``messages`` and ``variation_axes``.
    """
    messages = []
    for message in payload.get("messages") or []:
        entry: dict[str, Any] = {
            "role": message.get("role"),
            "content": _normalize_content(str(message.get("content", ""))),
        }
        # Only carry `name` when set: `None` and absent must not differ, or an
        # id would depend on whether the producer spelled out a null.
        if message.get("name"):
            entry["name"] = message["name"]
        messages.append(entry)

    axes = {
        key: value
        for key, value in sorted((payload.get("variation_axes") or {}).items())
        if value is not None
    }

    return {"task": payload.get("task"), "messages": messages, "variation_axes": axes}


def content_hash(payload: Mapping[str, Any]) -> str:
    """Full 64-character digest over a payload's canonical content."""
    return sha256_hex(canonical_bytes(canonicalize(payload)))


def task_code(task: str | None) -> str:
    """Three-letter code for a task."""
    return TASK_CODES.get(task or "", UNKNOWN_TASK_CODE)


def format_id(digest: str, task: str | None, *, chars: int = DEFAULT_DIGEST_CHARS) -> str:
    """Render a digest and task as a contract-legal id."""
    candidate = f"{ID_PREFIX}-{task_code(task)}-{digest[:chars]}"
    if not ID_PATTERN.match(candidate):  # pragma: no cover - defensive
        raise ContractViolationError(
            f"Generated id {candidate!r} does not satisfy the public contract.",
            details={"pattern": ID_PATTERN.pattern},
        )
    return candidate


def example_id(payload: Mapping[str, Any], *, chars: int = DEFAULT_DIGEST_CHARS) -> str:
    """Mint an id for a payload, ignoring any collision ledger.

    Sufficient for tests and for previewing. Promotion uses
    :meth:`CollisionLedger.mint`, which additionally guarantees uniqueness.
    """
    return format_id(content_hash(payload), payload.get("task"), chars=chars)


@dataclass
class CollisionLedger:
    """Maps full content hashes to the ids assigned to them.

    16 hex characters is 64 bits, so a collision is negligible at any corpus size
    this project will reach. "Negligible" is not "impossible", and a silent id
    collision corrupts a leakage report in a way nobody would think to check for
    — two different examples reported as one duplicate.

    The ledger makes assignment **order-independent**: an id depends on the
    ledger's contents, never on which example happened to be promoted first, so a
    fresh checkout re-derives the same ids from the same corpus. It is committed
    for that reason, and it contains only hashes and ids — never content.
    """

    #: full content hash -> assigned id
    assignments: dict[str, str] = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, path: Path | str) -> CollisionLedger:
        """Read a ledger, or return an empty one if the file does not exist."""
        target = Path(path)
        if not target.is_file():
            return cls(assignments={}, path=target)
        payload = json.loads(target.read_text(encoding="utf-8"))
        return cls(assignments=dict(payload.get("assignments") or {}), path=target)

    def save(self, path: Path | str | None = None) -> Path:
        """Write the ledger, sorted so a diff shows only real changes."""
        target = Path(path or self.path or "")
        if not str(target):  # pragma: no cover - defensive
            raise ContractViolationError("CollisionLedger.save() needs a path.")
        target.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "_comment": (
                "Content hash to assigned example id. Contains no example content. "
                "Committed so ids are reproducible from a fresh checkout."
            ),
            "assignments": dict(sorted(self.assignments.items())),
        }
        target.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    def mint(self, payload: Mapping[str, Any]) -> str:
        """Return the id for a payload, assigning one if it is new.

        Raises:
            ContractViolationError: If every escalation length is exhausted,
                which would require a genuine 256-bit collision.
        """
        digest = content_hash(payload)
        existing = self.assignments.get(digest)
        if existing is not None:
            return existing

        claimed = set(self.assignments.values())
        for chars in DIGEST_ESCALATION:
            candidate = format_id(digest, payload.get("task"), chars=chars)
            if candidate not in claimed:
                self.assignments[digest] = candidate
                return candidate

        raise ContractViolationError(  # pragma: no cover - unreachable in practice
            "Could not mint a unique id: every digest length collided.",
            details={"content_hash": digest},
            suggestions=["This implies a sha256 collision. Verify the ledger is not corrupt."],
        )

    def verify(self, payload: Mapping[str, Any], assigned_id: str) -> bool:
        """Whether ``assigned_id`` is what this payload's content maps to.

        Promotion gate G03 calls this for every candidate. It catches an example
        whose content was edited after its id was minted — which would otherwise
        produce a release where the id no longer describes the content, breaking
        every cross-version comparison silently.
        """
        digest = content_hash(payload)
        recorded = self.assignments.get(digest)
        if recorded is not None:
            return recorded == assigned_id
        # Not yet in the ledger: accept any escalation length that matches.
        return any(
            format_id(digest, payload.get("task"), chars=chars) == assigned_id
            for chars in DIGEST_ESCALATION
        )
