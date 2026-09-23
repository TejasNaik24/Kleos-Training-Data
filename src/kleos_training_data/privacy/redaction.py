from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kleos_training_data.errors import SanitizationError
from kleos_training_data.privacy.detect import Detection
from kleos_training_data.scenarios.surrogates import SurrogatePool

PLACEHOLDER_TEMPLATE = "[[{slot}_{index}]]"

SLOT_POOLS: dict[str, str] = {
    "PERSON": "person_pool_a",
    "ORG": "generic_pool_a",
    "PROJECT": "generic_pool_a",
    "PLACE": "generic_pool_a",
    "PRODUCT": "generic_pool_a",
    "EMAIL": "person_pool_a",
    "PHONE": "generic_pool_a",
    "ID": "generic_pool_a",
    "PATH": "generic_pool_a",
    "URL": "generic_pool_a",
    "ADDRESS": "generic_pool_a",
    "POSTAL": "generic_pool_a",
    "HANDLE": "person_pool_a",
    "DATE": "generic_pool_a",
}

STRUCTURED_SLOTS: dict[str, str] = {
    "EMAIL": "{name}@example.invalid",
    "PHONE": "555-0100",
    "ID": "ID-000{n}",
    "PATH": "/Users/example/{name}",
    "URL": "https://example.invalid/{name}",
    "ADDRESS": "100 Example Street",
    "POSTAL": "00000",
    "HANDLE": "@{name}",
    "DATE": "next Friday",
}


@dataclass
class PlaceholderMap:
    by_digest: dict[str, str] = field(default_factory=dict)
    surrogates: dict[str, str] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    def placeholder_for(self, slot: str, digest: str) -> str:
        key = f"{slot}:{digest}"
        if key in self.by_digest:
            return self.by_digest[key]
        index = self.counters.get(slot, 0) + 1
        self.counters[slot] = index
        placeholder = PLACEHOLDER_TEMPLATE.format(slot=slot, index=index)
        self.by_digest[key] = placeholder
        return placeholder

    @property
    def placeholders(self) -> list[str]:
        return sorted(self.by_digest.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "placeholders": self.placeholders,
            "surrogates": dict(sorted(self.surrogates.items())),
            "counts": dict(sorted(self.counters.items())),
        }


def redact(text: str, detections: list[Detection], mapping: PlaceholderMap) -> str:
    redactable = sorted(
        (d for d in detections if d.redactable), key=lambda d: d.start, reverse=True
    )
    result = text
    for detection in redactable:
        assert detection.slot is not None
        placeholder = mapping.placeholder_for(detection.slot, detection.matched_sha256_8)
        result = result[: detection.start] + placeholder + result[detection.end :]
    return result


def _surrogate(
    slot: str, placeholder: str, *, scenario_family: str, pools: dict[str, SurrogatePool]
) -> str:
    pool_name = SLOT_POOLS.get(slot, "generic_pool_a")
    pool = pools.get(pool_name)
    if pool is None:
        raise SanitizationError(
            f"Surrogate pool {pool_name!r} is not available for slot {slot!r}.",
            details={"available": ", ".join(sorted(pools)) or "(none)"},
            suggestions=[f"Add data/surrogates/{pool_name}.yaml."],
        )

    name = pool.surrogate_for(scenario_family, placeholder)
    template = STRUCTURED_SLOTS.get(slot)
    if template is None:
        return name
    return template.format(name=name.lower().replace(" ", "."), n=len(placeholder) % 10)


def rehydrate(
    text: str,
    mapping: PlaceholderMap,
    *,
    scenario_family: str,
    pools: dict[str, SurrogatePool],
) -> str:
    result = text
    for placeholder in sorted(mapping.placeholders, key=len, reverse=True):
        if placeholder not in result:
            continue
        slot = placeholder.strip("[]").rsplit("_", 1)[0]
        surrogate = mapping.surrogates.get(placeholder)
        if surrogate is None:
            surrogate = _surrogate(slot, placeholder, scenario_family=scenario_family, pools=pools)
            mapping.surrogates[placeholder] = surrogate
        result = result.replace(placeholder, surrogate)
    return result


def has_placeholder_residue(text: str) -> bool:
    return "[[" in text and "]]" in text
