"""Redaction and surrogate substitution.

Two stages, and the second is the one that matters.

**Redact** replaces each detected span with a slot placeholder — ``[[PERSON_1]]``,
``[[ORG_2]]`` — recording the mapping in a :class:`PlaceholderMap`.

**Rehydrate** then replaces each placeholder with a consistent *fictional*
surrogate drawn from a committed pool.

Shipping the placeholders would be the obvious shortcut and it is wrong.
Training on ``[[PERSON_1]]`` teaches a model to emit bracket tokens and destroys
the naturalness the task depends on — a prioritization example reads as a
prioritization example or it teaches nothing. Training on the real name teaches
a private fact. A fictional surrogate keeps the text natural and keeps
coreference intact, so "Dana" in turn one is "Dana" in turn three.

Surrogates are keyed on **scenario family**. Within a family the mapping is
stable; across families the same real person maps to a *different* fictional
person. That last property is what actually defeats memorization: there is no
persistent entity spanning the corpus to memorize.

Placeholders deliberately avoid the public validator's placeholder patterns
(``TODO``, ``FIXME``, ``lorem ipsum``, a line of only ``...``), which would make
a candidate fail validation for the wrong reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kleos_training_data.errors import SanitizationError
from kleos_training_data.privacy.detect import Detection
from kleos_training_data.scenarios.surrogates import SurrogatePool

#: Placeholder syntax. Double brackets are rare in prose and are not one of the
#: public validator's placeholder markers.
PLACEHOLDER_TEMPLATE = "[[{slot}_{index}]]"

#: Which surrogate pool serves which slot.
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

#: Slots rendered as structured stand-ins rather than as a name from a pool.
#: An email replaced by "Dana Whitfield" would be nonsense; it needs to still
#: look like an email.
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
    """What each placeholder stood for, and what it became.

    The ``digest`` side is safe to keep. The real value is never stored here —
    it lives only in the vault, if anywhere.
    """

    #: placeholder -> matched digest, so repeat occurrences reuse one placeholder
    by_digest: dict[str, str] = field(default_factory=dict)
    #: placeholder -> surrogate finally substituted
    surrogates: dict[str, str] = field(default_factory=dict)
    #: slot -> next index
    counters: dict[str, int] = field(default_factory=dict)

    def placeholder_for(self, slot: str, digest: str) -> str:
        """Stable placeholder for one distinct matched value.

        Keyed on the digest so the same real value gets the same placeholder
        everywhere it appears — which is what preserves coreference across turns.
        """
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
    """Replace every redactable detection in ``text`` with a placeholder.

    Replacements are applied right-to-left so earlier offsets stay valid.
    """
    redactable = sorted(
        (d for d in detections if d.redactable), key=lambda d: d.start, reverse=True
    )
    result = text
    for detection in redactable:
        assert detection.slot is not None  # guarded by `redactable`
        placeholder = mapping.placeholder_for(detection.slot, detection.matched_sha256_8)
        result = result[: detection.start] + placeholder + result[detection.end :]
    return result


def _surrogate(
    slot: str, placeholder: str, *, scenario_family: str, pools: dict[str, SurrogatePool]
) -> str:
    """Pick the fictional stand-in for one placeholder."""
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
    # A structured slot needs to still look like what it replaced.
    return template.format(name=name.lower().replace(" ", "."), n=len(placeholder) % 10)


def rehydrate(
    text: str,
    mapping: PlaceholderMap,
    *,
    scenario_family: str,
    pools: dict[str, SurrogatePool],
) -> str:
    """Replace placeholders with consistent fictional surrogates."""
    result = text
    # Longest placeholder first, so [[PERSON_10]] is not clobbered by [[PERSON_1]].
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
    """Whether any ``[[SLOT_N]]`` marker survived rehydration.

    Promotion gate G06 checks this. A surviving placeholder means a model would
    be trained to emit bracket tokens — a visible, embarrassing failure, but
    also a sign that the placeholder-to-surrogate mapping is incomplete.
    """
    return "[[" in text and "]]" in text
