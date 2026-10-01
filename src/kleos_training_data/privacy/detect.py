from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from kleos_training_data.privacy.rules import (
    ALL_RULES,
    NON_REDACTING_SEVERITIES,
    SEVERITIES,
    Rule,
    is_allowlisted,
)


@dataclass(frozen=True)
class Detection:
    rule_id: str
    kind: str
    layer: str
    severity: str
    field_path: str
    start: int
    end: int
    matched_len: int
    matched_sha256_8: str
    excerpt: str
    slot: str | None = None
    placeholder: str | None = None

    @property
    def redactable(self) -> bool:
        return self.severity not in NON_REDACTING_SEVERITIES and self.slot is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "kind": self.kind,
            "layer": self.layer,
            "severity": self.severity,
            "field_path": self.field_path,
            "start": self.start,
            "end": self.end,
            "matched_len": self.matched_len,
            "matched_sha256_8": self.matched_sha256_8,
            "excerpt": self.excerpt,
            "slot": self.slot,
            "placeholder": self.placeholder,
        }


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


@lru_cache(maxsize=1)
def default_known_names() -> frozenset[str]:
    from kleos_training_data.scenarios.surrogates import load_pools

    names: set[str] = set()
    for pool in load_pools().values():
        for value in pool.values:
            names.add(value)
            names.update(value.split())
    return frozenset(names)


def redacted_excerpt(
    text: str,
    start: int,
    end: int,
    *,
    context: int = 24,
    mask_spans: tuple[tuple[int, int], ...] = (),
) -> str:
    spans = sorted({(start, end), *mask_spans})

    pieces: list[str] = []
    cursor = 0
    reported_at = 0
    for span_start, span_end in spans:
        if span_start < cursor:
            continue
        pieces.append(text[cursor:span_start])
        if (span_start, span_end) == (start, end):
            reported_at = sum(len(p) for p in pieces)
        pieces.append(f"<{span_end - span_start} chars redacted>")
        cursor = span_end
    pieces.append(text[cursor:])

    masked = "".join(pieces)
    marker_len = len(f"<{end - start} chars redacted>")

    before = masked[max(0, reported_at - context) : reported_at].replace("\n", " ")
    after = masked[reported_at + marker_len : reported_at + marker_len + context].replace("\n", " ")
    lead = "…" if reported_at - context > 0 else ""
    trail = "…" if reported_at + marker_len + context < len(masked) else ""
    return f"{lead}{before}<{end - start} chars redacted>{after}{trail}"


def scan_text(
    text: str,
    *,
    field_path: str,
    rules: tuple[Rule, ...] = ALL_RULES,
    known_names: frozenset[str] | None = None,
) -> list[Detection]:
    if known_names is None:
        known_names = default_known_names()

    found: list[Detection] = []

    for rule in rules:
        for match in rule.pattern.finditer(text):
            matched = match.group(0)
            if is_allowlisted(rule.kind, matched):
                continue
            if (
                rule.layer == "structural" or rule.kind in _SURROGATE_SHAPED_KINDS
            ) and _is_known_name(matched, known_names):
                continue
            found.append(
                Detection(
                    rule_id=rule.rule_id,
                    kind=rule.kind,
                    layer=rule.layer,
                    severity=rule.severity,
                    field_path=field_path,
                    start=match.start(),
                    end=match.end(),
                    matched_len=len(matched),
                    matched_sha256_8=_digest(matched),
                    excerpt="",
                    slot=rule.slot,
                )
            )

    return with_excerpts(_resolve_overlaps(found), text)


def with_excerpts(detections: list[Detection], text: str) -> list[Detection]:
    spans = tuple((d.start, d.end) for d in detections)
    return [
        Detection(
            **{
                **d.__dict__,
                "excerpt": redacted_excerpt(text, d.start, d.end, mask_spans=spans),
            }
        )
        for d in detections
    ]


_SURROGATE_SHAPED_KINDS: frozenset[str] = frozenset(
    {"social_handle", "email", "home_path", "person_name", "org_name"}
)

_NAME_SEPARATORS = re.compile(r"[@._\-\s/]+")

_TEMPLATE_LITERALS: frozenset[str] = frozenset(
    {"users", "home", "example", "invalid", "id", "https", "http"}
)


def _is_known_name(matched: str, known_names: frozenset[str] | None) -> bool:
    if not known_names or not matched:
        return False
    if matched in known_names:
        return True

    words = matched.split()
    if words and all(word in known_names for word in words):
        return True

    folded = {name.casefold() for name in known_names} | _TEMPLATE_LITERALS
    parts = [p for p in _NAME_SEPARATORS.split(matched) if p]
    return bool(parts) and all(part.casefold() in folded for part in parts)


def _resolve_overlaps(detections: list[Detection]) -> list[Detection]:
    ordered = sorted(
        detections,
        key=lambda d: (SEVERITIES.index(d.severity), -(d.end - d.start), d.start),
    )
    kept: list[Detection] = []
    for detection in ordered:
        if any(detection.start < k.end and k.start < detection.end for k in kept):
            continue
        kept.append(detection)
    return sorted(kept, key=lambda d: (d.field_path, d.start))


def iter_text_fields(payload: dict[str, Any]) -> Iterator[tuple[str, str]]:
    for index, message in enumerate(payload.get("messages") or []):
        content = message.get("content")
        if isinstance(content, str):
            yield f"messages[{index}].content", content
        name = message.get("name")
        if isinstance(name, str) and name:
            yield f"messages[{index}].name", name
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            yield f"messages[{index}].reasoning", reasoning

    for key, value in sorted((payload.get("variation_axes") or {}).items()):
        if isinstance(value, str):
            yield f"variation_axes.{key}", value


def scan_payload(
    payload: dict[str, Any],
    *,
    rules: tuple[Rule, ...] = ALL_RULES,
    known_names: frozenset[str] | None = None,
) -> list[Detection]:
    detections: list[Detection] = []
    for field_path, text in iter_text_fields(payload):
        detections.extend(
            scan_text(text, field_path=field_path, rules=rules, known_names=known_names)
        )
    return detections


@dataclass
class ScanSummary:
    detections: list[Detection] = field(default_factory=list)

    @property
    def blocking(self) -> list[Detection]:
        return [d for d in self.detections if d.severity == "block"]

    @property
    def redactable(self) -> list[Detection]:
        return [d for d in self.detections if d.redactable]

    @property
    def needs_review(self) -> list[Detection]:
        return [d for d in self.detections if d.severity == "review"]

    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for detection in self.detections:
            counts[detection.kind] = counts.get(detection.kind, 0) + 1
        return dict(sorted(counts.items()))

    def by_severity(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for detection in self.detections:
            counts[detection.severity] = counts.get(detection.severity, 0) + 1
        return {s: counts.get(s, 0) for s in SEVERITIES if s in counts}
