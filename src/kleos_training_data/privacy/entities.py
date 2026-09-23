from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kleos_training_data.errors import VaultError
from kleos_training_data.privacy.detect import Detection, _digest, redacted_excerpt

VAULT_SLOTS: tuple[str, ...] = ("PERSON", "ORG", "PROJECT", "PLACE", "PRODUCT")


@dataclass
class EntityVault:
    entries: dict[str, str] = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, path: Path | str) -> EntityVault:
        target = Path(path)
        if not target.is_file():
            return cls(entries={}, path=target)
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VaultError(
                f"The entity vault at {target} is not valid JSON.",
                details={"error": str(exc)},
                suggestions=[
                    "Do not guess at its contents — a corrupted vault means "
                    "sanitization would silently stop redacting the names in it.",
                ],
            ) from exc

        entries = dict(payload.get("entries") or {})
        for literal, slot in entries.items():
            if slot not in VAULT_SLOTS:
                raise VaultError(
                    f"Vault entry has unknown slot {slot!r}.",
                    details={"valid_slots": ", ".join(VAULT_SLOTS), "literal_len": len(literal)},
                )
        return cls(entries=entries, path=target)

    def save(self, path: Path | str | None = None) -> Path:
        target = Path(path or self.path or "")
        if not str(target):
            raise VaultError("EntityVault.save() needs a path.")
        target.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "_warning": (
                "RE-IDENTIFICATION KEYS. This file maps real names to placeholder "
                "slots. It must never be committed, copied into a release, or "
                "quoted in a report. See PRIVACY.md."
            ),
            "entries": dict(sorted(self.entries.items())),
        }
        target.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        target.chmod(0o600)
        return target

    def add(self, literal: str, slot: str) -> None:
        cleaned = literal.strip()
        if not cleaned:
            raise VaultError("A vault entry cannot be empty.")
        if slot not in VAULT_SLOTS:
            raise VaultError(
                f"Unknown vault slot {slot!r}.",
                details={"valid_slots": ", ".join(VAULT_SLOTS)},
            )
        self.entries[cleaned] = slot

    def scan(self, text: str, *, field_path: str) -> list[Detection]:
        detections: list[Detection] = []
        for literal in sorted(self.entries, key=len, reverse=True):
            slot = self.entries[literal]
            start = 0
            while (index := text.find(literal, start)) != -1:
                end = index + len(literal)
                detections.append(
                    Detection(
                        rule_id=f"vault.{slot.lower()}.v1",
                        kind="vault_entity",
                        layer="vault",
                        severity="redact",
                        field_path=field_path,
                        start=index,
                        end=end,
                        matched_len=len(literal),
                        matched_sha256_8=_digest(literal),
                        excerpt=redacted_excerpt(text, index, end),
                        slot=slot,
                    )
                )
                start = end
        return detections

    def scan_payload(self, payload: dict[str, Any]) -> list[Detection]:
        from kleos_training_data.privacy.detect import iter_text_fields

        detections: list[Detection] = []
        for field_path, text in iter_text_fields(payload):
            detections.extend(self.scan(text, field_path=field_path))
        return detections

    def contains_any(self, text: str) -> bool:
        return any(literal in text for literal in self.entries)

    def __len__(self) -> int:
        return len(self.entries)
