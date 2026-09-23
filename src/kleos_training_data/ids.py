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

ID_PREFIX: Final[str] = "kx"

UNKNOWN_TASK_CODE: Final[str] = "unk"

DEFAULT_DIGEST_CHARS: Final[int] = 16

DIGEST_ESCALATION: Final[tuple[int, ...]] = (16, 24, 32, 48, 64)

_TRAILING_WS = re.compile(r"[ \t]+$", flags=re.MULTILINE)


def _normalize_content(text: str) -> str:
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    trimmed = _TRAILING_WS.sub("", unified)
    return unicodedata.normalize("NFC", trimmed).strip()


def canonicalize(payload: Mapping[str, Any]) -> dict[str, Any]:
    messages = []
    for message in payload.get("messages") or []:
        entry: dict[str, Any] = {
            "role": message.get("role"),
            "content": _normalize_content(str(message.get("content", ""))),
        }
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
    return sha256_hex(canonical_bytes(canonicalize(payload)))


def task_code(task: str | None) -> str:
    return TASK_CODES.get(task or "", UNKNOWN_TASK_CODE)


def format_id(digest: str, task: str | None, *, chars: int = DEFAULT_DIGEST_CHARS) -> str:
    candidate = f"{ID_PREFIX}-{task_code(task)}-{digest[:chars]}"
    if not ID_PATTERN.match(candidate):
        raise ContractViolationError(
            f"Generated id {candidate!r} does not satisfy the public contract.",
            details={"pattern": ID_PATTERN.pattern},
        )
    return candidate


def example_id(payload: Mapping[str, Any], *, chars: int = DEFAULT_DIGEST_CHARS) -> str:
    return format_id(content_hash(payload), payload.get("task"), chars=chars)


@dataclass
class CollisionLedger:
    assignments: dict[str, str] = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, path: Path | str) -> CollisionLedger:
        target = Path(path)
        if not target.is_file():
            return cls(assignments={}, path=target)
        payload = json.loads(target.read_text(encoding="utf-8"))
        return cls(assignments=dict(payload.get("assignments") or {}), path=target)

    def save(self, path: Path | str | None = None) -> Path:
        target = Path(path or self.path or "")
        if not str(target):
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

        raise ContractViolationError(
            "Could not mint a unique id: every digest length collided.",
            details={"content_hash": digest},
            suggestions=["This implies a sha256 collision. Verify the ledger is not corrupt."],
        )

    def verify(self, payload: Mapping[str, Any], assigned_id: str) -> bool:
        digest = content_hash(payload)
        recorded = self.assignments.get(digest)
        if recorded is not None:
            return recorded == assigned_id
        return any(
            format_id(digest, payload.get("task"), chars=chars) == assigned_id
            for chars in DIGEST_ESCALATION
        )
