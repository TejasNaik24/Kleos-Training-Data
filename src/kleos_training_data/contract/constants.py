from __future__ import annotations

from typing import Final

DATASET_SCHEMA_VERSION: Final[str] = "1.0"

PREPROCESSING_VERSION: Final[str] = "1.0"

PIPELINE_VERSION: Final[str] = "0.1.0"


SUPPORTED_TASKS: Final[tuple[str, ...]] = (
    "notification_prioritization",
    "tool_routing",
    "mission_control_briefing",
    "context_prioritization",
    "recommendation_generation",
    "memory_conflict_resolution",
    "workspace_reasoning",
)

TASK_CODES: Final[dict[str, str]] = {
    "notification_prioritization": "npr",
    "tool_routing": "trt",
    "mission_control_briefing": "mcb",
    "context_prioritization": "ctx",
    "recommendation_generation": "rec",
    "memory_conflict_resolution": "mcr",
    "workspace_reasoning": "wsr",
}


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

REQUIRED_VARIATION_AXES: Final[tuple[str, ...]] = ("domain",)


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

PROMOTED_QUALITY_STATUS: Final[str] = "reviewed"

MESSAGE_ROLES: Final[tuple[str, ...]] = ("system", "user", "assistant", "tool")


SPLIT_STRATEGIES: Final[tuple[str, ...]] = (
    "random",
    "group",
    "entity_holdout",
    "domain_holdout",
    "format_holdout",
    "scenario_family_holdout",
)

SPLIT_NAMES: Final[tuple[str, ...]] = ("train", "validation", "test")

SPLIT_FILENAMES: Final[dict[str, str]] = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
}

REQUIRED_SPLITS: Final[tuple[str, ...]] = ("train",)


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


MANIFEST_FILENAME: Final[str] = "manifest.json"

PROVENANCE_FILENAME: Final[str] = "provenance.json"

RELEASE_LOCK_FILENAME: Final[str] = "RELEASE.lock"


ALLOWED_METADATA_EXTRAS: Final[frozenset[str]] = frozenset(
    {
        "ruleset_version",
        "rubric_version",
        "scenario_catalog_version",
        "policy_claim_id",
        "pipeline_version",
    }
)
