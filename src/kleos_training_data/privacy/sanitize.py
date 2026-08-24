"""Run all four detection layers and produce a sanitized candidate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kleos_training_data.ids import content_hash
from kleos_training_data.privacy.detect import Detection, ScanSummary, scan_payload, scan_text
from kleos_training_data.privacy.entities import EntityVault
from kleos_training_data.privacy.facts import FactRiskAssessment, assess
from kleos_training_data.privacy.redaction import (
    PlaceholderMap,
    has_placeholder_residue,
    redact,
    rehydrate,
)
from kleos_training_data.privacy.rules import RULESET_VERSION
from kleos_training_data.scenarios.surrogates import SurrogatePool

#: Sanitization outcomes.
STATUS_CLEAN = "clean"
STATUS_SANITIZED = "sanitized"
STATUS_NEEDS_REVIEW = "needs_review"
STATUS_BLOCKED = "blocked"


@dataclass
class SanitizationResult:
    """Everything sanitization concluded about one candidate.

    Never contains a matched value. ``input_hash`` is a pointer back into
    ``staging/`` and is deliberately kept out of the promoted example's metadata
    — gate G14's allowlist rejects it, because a release should not carry a
    reference into the staging zone.
    """

    status: str
    payload: dict[str, Any]
    detections: list[Detection] = field(default_factory=list)
    fact_risk: FactRiskAssessment = field(default_factory=FactRiskAssessment)
    replacements: int = 0
    input_hash: str = ""
    output_hash: str = ""
    ruleset_version: str = RULESET_VERSION
    surrogate_map_id: str | None = None
    placeholder_map: PlaceholderMap = field(default_factory=PlaceholderMap)

    @property
    def ok(self) -> bool:
        """Whether the candidate may proceed to review."""
        return self.status in {STATUS_CLEAN, STATUS_SANITIZED}

    @property
    def blocking(self) -> list[Detection]:
        return [d for d in self.detections if d.severity == "block"]

    @property
    def residual_rule_ids(self) -> list[str]:
        """Rules that fired and could not be resolved automatically."""
        return sorted({d.rule_id for d in self.detections if d.severity in {"block", "review"}})

    def to_dict(self) -> dict[str, Any]:
        """The ``<id>.privacy.json`` sidecar."""
        summary = ScanSummary(self.detections)
        return {
            "status": self.status,
            "ok": self.ok,
            "ruleset_version": self.ruleset_version,
            "replacements": self.replacements,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "surrogate_map_id": self.surrogate_map_id,
            "residual_rule_ids": self.residual_rule_ids,
            "counts_by_kind": summary.by_kind(),
            "counts_by_severity": summary.by_severity(),
            "detections": [d.to_dict() for d in self.detections],
            "fact_risk": self.fact_risk.to_dict(),
            "placeholders": self.placeholder_map.to_dict(),
        }


def known_surrogate_names(pools: dict[str, SurrogatePool]) -> frozenset[str]:
    """Every name the committed pools can produce, plus their component words.

    Passed to the structural layer so it does not flag the fictional names this
    repository generated as if they were somebody's personal data.
    """
    names: set[str] = set()
    for pool in pools.values():
        for value in pool.values:
            names.add(value)
            names.update(value.split())
    return frozenset(names)


def _merged_detections(
    payload: dict[str, Any],
    vault: EntityVault | None,
    known_names: frozenset[str] | None = None,
) -> list[Detection]:
    """Run the regex layers and the vault layer, then resolve overlaps *together*.

    Resolving each layer separately is not enough. A vault entry for an
    organization frequently sits inside an email address containing it, so both
    fire on overlapping spans; applying both replacements corrupts the text
    (``dana@realcompany.com`` became ``dana@example.invalid`` with the tail of
    the following word eaten). Overlap resolution has to see every layer at once.
    """
    from kleos_training_data.privacy.detect import (
        _resolve_overlaps,
        iter_text_fields,
        with_excerpts,
    )

    merged: list[Detection] = []
    for field_path, text in iter_text_fields(payload):
        candidates = scan_text(text, field_path=field_path, known_names=known_names)
        if vault is not None:
            candidates = candidates + vault.scan(text, field_path=field_path)
        merged.extend(with_excerpts(_resolve_overlaps(candidates), text))
    return sorted(merged, key=lambda d: (d.field_path, d.start))


def _apply(
    payload: dict[str, Any],
    detections: list[Detection],
    mapping: PlaceholderMap,
    *,
    scenario_family: str,
    pools: dict[str, SurrogatePool],
) -> tuple[dict[str, Any], int]:
    """Redact then rehydrate every text field. Returns the payload and a count."""
    by_field: dict[str, list[Detection]] = {}
    for detection in detections:
        by_field.setdefault(detection.field_path, []).append(detection)

    replacements = 0
    messages = []
    for index, message in enumerate(payload.get("messages") or []):
        content = str(message.get("content", ""))
        field_detections = by_field.get(f"messages[{index}].content", [])
        redactable = [d for d in field_detections if d.redactable]
        if redactable:
            content = redact(content, redactable, mapping)
            content = rehydrate(content, mapping, scenario_family=scenario_family, pools=pools)
            replacements += len(redactable)
        messages.append({**message, "content": content})

    axes = {}
    for key, value in (payload.get("variation_axes") or {}).items():
        if not isinstance(value, str):
            axes[key] = value
            continue
        field_detections = [d for d in by_field.get(f"variation_axes.{key}", []) if d.redactable]
        if field_detections:
            replaced = redact(value, field_detections, mapping)
            axes[key] = rehydrate(replaced, mapping, scenario_family=scenario_family, pools=pools)
            replacements += len(field_detections)
        else:
            axes[key] = value

    return {**payload, "messages": messages, "variation_axes": axes}, replacements


def sanitize(
    payload: dict[str, Any],
    *,
    scenario_family: str,
    vault: EntityVault | None = None,
    pools: dict[str, SurrogatePool] | None = None,
) -> SanitizationResult:
    """Run every layer over one candidate payload.

    A secret short-circuits everything: the candidate is ``blocked`` and nothing
    is redacted. Redacting a credential would produce a clean-looking candidate
    and hide that the capture path is compromised — which is a bigger problem
    than the one example.
    """
    from kleos_training_data.scenarios.surrogates import load_pools

    available = pools if pools is not None else load_pools()
    input_hash = content_hash(payload)

    known = known_surrogate_names(available)
    detections = _merged_detections(payload, vault, known)

    fact_risk = assess(payload)

    blocking = [d for d in detections if d.severity == "block"]
    if blocking:
        return SanitizationResult(
            status=STATUS_BLOCKED,
            payload=payload,
            detections=detections,
            fact_risk=fact_risk,
            replacements=0,
            input_hash=input_hash,
            output_hash=input_hash,
        )

    mapping = PlaceholderMap()
    sanitized, replacements = _apply(
        payload, detections, mapping, scenario_family=scenario_family, pools=available
    )

    # Re-scan the output. A rule that fires on the sanitized text means redaction
    # produced something still sensitive — the case where a partial replacement
    # leaves an identifying remainder behind.
    residual = scan_payload(sanitized, known_names=known)
    residual_blocking = [d for d in residual if d.severity == "block"]

    if residual_blocking:
        return SanitizationResult(
            status=STATUS_BLOCKED,
            payload=sanitized,
            detections=detections + residual,
            fact_risk=fact_risk,
            replacements=replacements,
            input_hash=input_hash,
            output_hash=content_hash(sanitized),
            placeholder_map=mapping,
        )

    needs_review = [d for d in residual if d.severity == "review"] or fact_risk.requires_human
    if replacements == 0 and not detections:
        status = STATUS_CLEAN
    elif needs_review:
        status = STATUS_NEEDS_REVIEW
    else:
        status = STATUS_SANITIZED

    return SanitizationResult(
        status=status,
        payload=sanitized,
        detections=detections + [d for d in residual if d.severity != "block"],
        fact_risk=fact_risk,
        replacements=replacements,
        input_hash=input_hash,
        output_hash=content_hash(sanitized),
        surrogate_map_id=f"sm-{scenario_family}" if replacements else None,
        placeholder_map=mapping,
    )


def verify_sanitized(payload: dict[str, Any], *, vault: EntityVault | None = None) -> list[str]:
    """Final byte-level check, used by promotion gate G06.

    Returns the problems found, empty when clean. Three things must hold:
    no secret survives, no placeholder residue survives, and no vault literal
    survives. The last is the one nothing downstream could catch — after
    sanitization, nothing else knows the string was ever sensitive.
    """
    problems: list[str] = []

    for detection in scan_payload(payload):
        if detection.severity == "block":
            problems.append(f"{detection.rule_id} at {detection.field_path}")

    for message in payload.get("messages") or []:
        content = str(message.get("content", ""))
        if has_placeholder_residue(content):
            problems.append("placeholder residue survived rehydration")
            break

    if vault is not None:
        for message in payload.get("messages") or []:
            if vault.contains_any(str(message.get("content", ""))):
                problems.append("a vault literal survived sanitization")
                break

    return problems


def scan_release_text(text: str) -> list[Detection]:
    """Scan arbitrary text, for the byte-level scan of a written release."""
    return scan_text(text, field_path="release")
