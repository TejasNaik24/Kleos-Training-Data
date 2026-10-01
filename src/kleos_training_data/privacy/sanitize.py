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

STATUS_CLEAN = "clean"
STATUS_SANITIZED = "sanitized"
STATUS_NEEDS_REVIEW = "needs_review"
STATUS_BLOCKED = "blocked"


@dataclass
class SanitizationResult:
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
        return self.status in {STATUS_CLEAN, STATUS_SANITIZED}

    @property
    def blocking(self) -> list[Detection]:
        return [d for d in self.detections if d.severity == "block"]

    @property
    def residual_rule_ids(self) -> list[str]:
        return sorted({d.rule_id for d in self.detections if d.severity in {"block", "review"}})

    def to_dict(self) -> dict[str, Any]:
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
    by_field: dict[str, list[Detection]] = {}
    for detection in detections:
        by_field.setdefault(detection.field_path, []).append(detection)

    replacements = 0
    messages = []
    for index, message in enumerate(payload.get("messages") or []):
        rebuilt = dict(message)
        for field_name in ("content", "reasoning"):
            if field_name == "reasoning" and not isinstance(message.get("reasoning"), str):
                continue
            text = str(message.get(field_name, ""))
            field_detections = by_field.get(f"messages[{index}].{field_name}", [])
            redactable = [d for d in field_detections if d.redactable]
            if redactable:
                text = redact(text, redactable, mapping)
                text = rehydrate(text, mapping, scenario_family=scenario_family, pools=pools)
                replacements += len(redactable)
            rebuilt[field_name] = text
        messages.append(rebuilt)

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
    problems: list[str] = []

    for detection in scan_payload(payload):
        if detection.severity == "block":
            problems.append(f"{detection.rule_id} at {detection.field_path}")

    texts = [
        str(message.get(field_name, ""))
        for message in payload.get("messages") or []
        for field_name in ("content", "reasoning")
        if isinstance(message.get(field_name), str)
    ]

    if any(has_placeholder_residue(text) for text in texts):
        problems.append("placeholder residue survived rehydration")

    if vault is not None and any(vault.contains_any(text) for text in texts):
        problems.append("a vault literal survived sanitization")

    return problems


def scan_release_text(text: str) -> list[Detection]:
    return scan_text(text, field_path="release")
