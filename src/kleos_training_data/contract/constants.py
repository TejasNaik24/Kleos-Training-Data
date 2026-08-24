"""The public contract's vocabulary, mirrored.

Every tuple here must match ``kleos_models.constants`` **element for element and
in the same order**. Order matters even where it looks cosmetic: it determines
how ``holdout_values`` sorts and how a coverage report enumerates cells, so a
reordered tuple changes an artifact without changing a single value.

``tests/test_contract_pin.py`` compares all of it against the pinned public
package. That test is the early-warning detector for contract drift — it fails
on a newly registered task long before any byte-level test would notice.

Only the vocabulary needed to *produce* a dataset is mirrored. The public repo's
evaluation vocabulary (research arms, its grading rubric) is deliberately absent:
this repository has its own review rubric, and mirroring things we do not use
would invite them to drift unnoticed.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Schema versioning
# ---------------------------------------------------------------------------

#: Version of the training-example schema. A dataset records the version it was
#: written against so old artifacts stay interpretable.
DATASET_SCHEMA_VERSION: Final[str] = "1.0"

#: Version of the public repo's preprocessing/formatting logic.
PREPROCESSING_VERSION: Final[str] = "1.0"

#: Version of *this* repository's sanitization and promotion behaviour. Not part
#: of the public contract — it is recorded in provenance.json so a release can be
#: re-derived. Bump when a change would produce different output from identical
#: source material.
PIPELINE_VERSION: Final[str] = "0.1.0"


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

SUPPORTED_TASKS: Final[tuple[str, ...]] = (
    "notification_prioritization",
    "tool_routing",
    "mission_control_briefing",
    "context_prioritization",
    "recommendation_generation",
    "memory_conflict_resolution",
    "workspace_reasoning",
)

#: Short codes used in generated example ids, so an id is triageable in a
#: leakage report without a lookup. Not part of the public contract.
TASK_CODES: Final[dict[str, str]] = {
    "notification_prioritization": "npr",
    "tool_routing": "trt",
    "mission_control_briefing": "mcb",
    "context_prioritization": "ctx",
    "recommendation_generation": "rec",
    "memory_conflict_resolution": "mcr",
    "workspace_reasoning": "wsr",
}


# ---------------------------------------------------------------------------
# Variation axes
# ---------------------------------------------------------------------------

#: Registered axes. Note that ``task`` appears here but is NOT a field on the
#: VariationAxes model — the public coverage report special-cases it to read
#: ``example.task``. Mirroring the quirk rather than fixing it is deliberate.
VARIATION_AXES: Final[tuple[str, ...]] = (
    "domain",
    "entities",
    "urgency",
    "deadlines",
    "evidence_quality",
    "conflicting_evidence",
    "context_length",
    "presentation_order",
    "format",
    "source_type",
    "workspace",
    "task",
    "difficulty",
    "ambiguity",
)

#: Axes that must be present on every example.
REQUIRED_VARIATION_AXES: Final[tuple[str, ...]] = ("domain",)


# ---------------------------------------------------------------------------
# Provenance and quality
# ---------------------------------------------------------------------------

SOURCE_TYPES: Final[tuple[str, ...]] = (
    "synthetic",
    "synthetic_seeded",
    "real_sanitized",
    "expert_authored",
    "development_fixture",
)

QUALITY_STATUSES: Final[tuple[str, ...]] = (
    "draft",
    "auto_generated",
    "reviewed",
    "rejected",
)

#: The only status a promoted example may carry.
#:
#: The public loader's default filter is ``quality_statuses=["reviewed"]``, so
#: anything else is silently dropped at train time — producing a release that
#: validates cleanly and then trains on nothing. Promotion gate G09 enforces
#: this, and verify_release asserts it on the written bytes.
PROMOTED_QUALITY_STATUS: Final[str] = "reviewed"

MESSAGE_ROLES: Final[tuple[str, ...]] = ("system", "user", "assistant", "tool")


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

SPLIT_STRATEGIES: Final[tuple[str, ...]] = (
    "random",
    "group",
    "entity_holdout",
    "domain_holdout",
    "format_holdout",
    "scenario_family_holdout",
)

SPLIT_NAMES: Final[tuple[str, ...]] = ("train", "validation", "test")

#: Exact filenames a release directory may contain.
#:
#: The public ``validate_dataset.py`` also accepts ``synthetic_train.jsonl`` and
#: ``valid.jsonl``, but ``train.py`` does not — so a release using an alias
#: passes validation and then fails to train. Only these names are ever written,
#: and verify_release asserts no other ``*.jsonl`` is present.
SPLIT_FILENAMES: Final[dict[str, str]] = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
}

#: The only split that must exist for a release to be loadable.
REQUIRED_SPLITS: Final[tuple[str, ...]] = ("train",)


# ---------------------------------------------------------------------------
# Consistency and OOD
# ---------------------------------------------------------------------------

#: Perturbation kinds. A group of examples sharing a scenario family and
#: differing only by perturbation is what makes consistency measurable; without
#: at least two members the public consistency report is silently empty.
PERTURBATION_KINDS: Final[tuple[str, ...]] = (
    "paraphrase",
    "evidence_order",
    "context_order",
    "irrelevant_context",
    "formatting",
    "schema",
    "length",
)

OOD_SHIFT_KINDS: Final[tuple[str, ...]] = (
    "unseen_entities",
    "unseen_domains",
    "unseen_formats",
    "unseen_source_types",
    "context_length_shift",
    "reordered_evidence",
    "conflicting_evidence",
)


# ---------------------------------------------------------------------------
# Filesystem conventions
# ---------------------------------------------------------------------------

MANIFEST_FILENAME: Final[str] = "manifest.json"

#: Private sidecar carrying everything the manifest cannot hold.
#:
#: DatasetManifest is ``extra="forbid"``, so one added key makes the public
#: loader raise DatasetIntegrityError at train time. The loader ignores unknown
#: *files* in a release directory, so a sibling is free.
PROVENANCE_FILENAME: Final[str] = "provenance.json"

#: Written last when a release is sealed, and the thing verify_release checks
#: against. File permissions are a speed bump; this is the actual guarantee.
RELEASE_LOCK_FILENAME: Final[str] = "RELEASE.lock"


# ---------------------------------------------------------------------------
# Metadata extras allowlist
# ---------------------------------------------------------------------------

#: ExampleMetadata is ``extra="allow"``, so anything placed there ships inside
#: train.jsonl. A stray ``capture_id`` or ``input_hash`` would be a pointer back
#: into staging/ that survives into whatever the release is shared with.
#:
#: Promotion gate G14 rejects any extra key not listed here.
ALLOWED_METADATA_EXTRAS: Final[frozenset[str]] = frozenset(
    {
        "ruleset_version",
        "rubric_version",
        "scenario_catalog_version",
        "policy_claim_id",
        "pipeline_version",
    }
)
