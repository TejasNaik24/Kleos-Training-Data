"""Layered privacy detection, redaction and private-fact assessment.

Four layers, deterministic and authoritative. An LLM may assist review; it never
decides whether something is private.

See PRIVACY.md for the reasoning, and note the division this package enforces:
PII is a string problem with a mechanical fix, and a private fact is a semantic
problem with none. The first is redacted here; the second is only ever flagged,
because a heuristic confident enough to auto-reject would be confident enough to
auto-approve, and neither is warranted.
"""

from __future__ import annotations

from kleos_training_data.privacy.detect import (
    Detection,
    ScanSummary,
    iter_text_fields,
    redacted_excerpt,
    scan_payload,
    scan_text,
)
from kleos_training_data.privacy.entities import VAULT_SLOTS, EntityVault
from kleos_training_data.privacy.facts import FactRiskAssessment, FactRiskSignal, assess
from kleos_training_data.privacy.redaction import (
    PlaceholderMap,
    has_placeholder_residue,
    redact,
    rehydrate,
)
from kleos_training_data.privacy.rules import (
    ALL_RULES,
    PII_RULES,
    RULESET_VERSION,
    SECRET_RULES,
    STRUCTURAL_RULES,
    Rule,
)
from kleos_training_data.privacy.sanitize import (
    STATUS_BLOCKED,
    STATUS_CLEAN,
    STATUS_NEEDS_REVIEW,
    STATUS_SANITIZED,
    SanitizationResult,
    sanitize,
    scan_release_text,
    verify_sanitized,
)

__all__ = [
    "ALL_RULES",
    "PII_RULES",
    "RULESET_VERSION",
    "SECRET_RULES",
    "STATUS_BLOCKED",
    "STATUS_CLEAN",
    "STATUS_NEEDS_REVIEW",
    "STATUS_SANITIZED",
    "STRUCTURAL_RULES",
    "VAULT_SLOTS",
    "Detection",
    "EntityVault",
    "FactRiskAssessment",
    "FactRiskSignal",
    "PlaceholderMap",
    "Rule",
    "SanitizationResult",
    "ScanSummary",
    "assess",
    "has_placeholder_residue",
    "iter_text_fields",
    "redact",
    "redacted_excerpt",
    "rehydrate",
    "sanitize",
    "scan_payload",
    "scan_release_text",
    "scan_text",
    "verify_sanitized",
]
