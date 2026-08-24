"""Run the detection rules over a candidate payload.

The invariant this module exists to hold: **a Detection never carries the value
it matched.** It carries a digest, a length, a location, and a redacted excerpt.

That matters because detections are written to
``staging/sanitized/<id>.privacy.json``, printed in reports, and pasted into
issues. A detection record that quoted the secret it found would relocate the
problem rather than describe it — and would do so into files that outlive the
candidate.
"""

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
    """One rule firing at one location.

    ``matched_sha256_8`` lets two detections be compared for identity — "is this
    the same token that fired in the other message?" — without either record
    holding the token.
    """

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
        """Whether this detection is replaced rather than merely reported."""
        return self.severity not in NON_REDACTING_SEVERITIES and self.slot is not None

    def to_dict(self) -> dict[str, Any]:
        """Serializable form. Contains no matched value, by construction."""
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
    """Names the committed surrogate pools can produce.

    The **default** for every scan, not an argument call sites must remember.

    Making it opt-in was a mistake worth recording: sanitization passed it and
    the reviewer packet did not, so the machine reviewer failed
    ``no_private_data`` on four candidates that sanitization had just declared
    clean. Two components disagreeing about the same bytes is worse than either
    being wrong on its own, because it makes the disagreement the thing a
    reviewer has to adjudicate.

    "The fictional names this repository generated are not personal data" is a
    global truth, so it belongs in the default rather than at each call site.
    """
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
    """Locate a hit without reprinting it, or anything else sensitive near it.

    Two things get masked, and the second is easy to miss:

    * the reported span itself, and
    * **every other detected span**, because the context window around one hit
      routinely contains the next one.

    Without the second, an excerpt reading ``"…due <10 chars redacted>. Call
    614-555-9876."`` reports the date while reprinting the phone number
    immediately after it — into a sidecar file, a report, and eventually an
    issue. Each detection looks individually correct and the set of them leaks
    everything.
    """
    spans = sorted({(start, end), *mask_spans})

    # Rebuild the text with every span masked, tracking where the reported span
    # lands so the window can be centred on it.
    pieces: list[str] = []
    cursor = 0
    reported_at = 0
    for span_start, span_end in spans:
        if span_start < cursor:  # already covered by an earlier mask
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
    """Find every rule hit in one string.

    Overlaps are resolved by severity first, then by length: a bearer token that
    also matches the generic ``assigned_secret`` rule is reported once, as the
    more specific block-severity hit. Reporting both would double-count the same
    problem and make the residual count meaningless.

    Args:
        known_names: Fictional names drawn from the committed surrogate pools.
            Flagging our own surrogates as personal data is definitionally wrong,
            and it is the fastest way to make the review lane pure noise — 32
            spurious flags across 24 synthetic examples, in practice. Only the
            structural layer consults this: a *secret* is a secret wherever it
            appears, even inside something that looks like a fictional name.
    """
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
                    excerpt="",  # filled in below, once every span is known
                    slot=rule.slot,
                )
            )

    return with_excerpts(_resolve_overlaps(found), text)


def with_excerpts(detections: list[Detection], text: str) -> list[Detection]:
    """Attach excerpts that mask every detected span, not merely their own.

    Deliberately a second pass. An excerpt cannot be built while scanning,
    because the spans it needs to mask include ones not yet found.
    """
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


#: Rules whose *surrogate* is deliberately shaped like the thing it replaced.
#:
#: A redacted ``@dana`` becomes ``@rowan.baxter`` — which is still a handle, and
#: the detector will flag it again on the next scan. That is not a defect in the
#: surrogate: an @handle has to look like an @handle or the text stops reading
#: naturally, which is the whole reason surrogates exist rather than placeholders.
#:
#: So these rules consult the known-name set too. A match whose components are
#: all names this repository generated is not personal data.
_SURROGATE_SHAPED_KINDS: frozenset[str] = frozenset(
    {"social_handle", "email", "home_path", "person_name", "org_name"}
)

#: Characters that join name components inside a surrogate.
_NAME_SEPARATORS = re.compile(r"[@._\-\s/]+")

#: Fixed scaffolding in the surrogate templates — ``/Users/example/<name>``,
#: ``<name>@example.invalid``. Not names, so they are not in any pool, but a
#: surrogate built from a template is still entirely ours.
_TEMPLATE_LITERALS: frozenset[str] = frozenset(
    {"users", "home", "example", "invalid", "id", "https", "http"}
)


def _is_known_name(matched: str, known_names: frozenset[str] | None) -> bool:
    """Whether a match is entirely composed of names we generated.

    Three passes, widening:

    * the whole string ("Blue Harbor" is a pool entry outright);
    * word-wise ("Silverbrook Brightwater" is two entries the bigram rule
      bridged);
    * component-wise after splitting on surrogate punctuation and folding case,
      so ``@rowan.baxter`` and ``/Users/rowan.baxter`` resolve to the pool
      entries "Rowan" and "Baxter".

    The third pass is what stops sanitization's own output re-triggering the rule
    that produced it.

    The accepted tradeoff: a real person who happens to share a pool name would
    be allowlisted. The pools are committed, deliberately generic, and reviewable
    precisely so that tradeoff is visible rather than hidden.
    """
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
    """Keep the most severe, then longest, detection for any overlapping span."""
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
    """Yield ``(field_path, text)`` for every string a candidate carries.

    Covers the variation axes as well as the messages. An axis value is
    free-form text, and ``workspace: "acme-internal"`` is exactly the kind of
    identifying detail that reaches a release unnoticed because nobody thinks of
    axes as content.
    """
    for index, message in enumerate(payload.get("messages") or []):
        content = message.get("content")
        if isinstance(content, str):
            yield f"messages[{index}].content", content
        name = message.get("name")
        if isinstance(name, str) and name:
            yield f"messages[{index}].name", name

    for key, value in sorted((payload.get("variation_axes") or {}).items()):
        if isinstance(value, str):
            yield f"variation_axes.{key}", value


def scan_payload(
    payload: dict[str, Any],
    *,
    rules: tuple[Rule, ...] = ALL_RULES,
    known_names: frozenset[str] | None = None,
) -> list[Detection]:
    """Scan every text field of a candidate."""
    detections: list[Detection] = []
    for field_path, text in iter_text_fields(payload):
        detections.extend(
            scan_text(text, field_path=field_path, rules=rules, known_names=known_names)
        )
    return detections


@dataclass
class ScanSummary:
    """Counts by severity and layer, for reports and gates."""

    detections: list[Detection] = field(default_factory=list)

    @property
    def blocking(self) -> list[Detection]:
        """Secrets. These are never redacted — they reject the candidate."""
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
