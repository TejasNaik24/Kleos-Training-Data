from __future__ import annotations

from typing import Final

CONTRACT_SOURCE_REPO: Final[str] = "https://github.com/TejasNaik24/Kleos-Models"

CONTRACT_SOURCE_COMMIT: Final[str] = "12361d53cf329c897ec14daede6d14112a3e8f20"

CONTRACT_VERIFIED_AT: Final[str] = "2026-08-22"

MIRRORED_CONSTANTS: Final[tuple[str, ...]] = (
    "DATASET_SCHEMA_VERSION",
    "PREPROCESSING_VERSION",
    "SUPPORTED_TASKS",
    "VARIATION_AXES",
    "REQUIRED_VARIATION_AXES",
    "SOURCE_TYPES",
    "QUALITY_STATUSES",
    "MESSAGE_ROLES",
    "SPLIT_STRATEGIES",
    "SPLIT_NAMES",
    "PERTURBATION_KINDS",
    "OOD_SHIFT_KINDS",
    "MANIFEST_FILENAME",
)

MIRRORED_BEHAVIOURS: Final[dict[str, str]] = {
    "jsonl_writer_bytes": "kleos_models.data.loaders.write_jsonl (loaders.py:412-424)",
    "split_stable_rank": "kleos_models.data.splitting._stable_rank (splitting.py:86-94)",
    "split_holdout_routing": "kleos_models.data.splitting._split_holdout (splitting.py:192-292)",
    "leakage_normalize_text": "kleos_models.data.leakage.normalize_text (leakage.py:95-108)",
    "conversation_text": "kleos_models.data.schemas.TrainingExample.conversation_text",
    "manifest_content_hash": "kleos_models.data.schemas.DatasetManifest.compute_content_hash",
}

DELIBERATE_DELTAS: Final[dict[str, str]] = {
    "split_filenames": (
        "The public validate_dataset.py accepts synthetic_train.jsonl and "
        "valid.jsonl; train.py does not. We write only the canonical names."
    ),
    "eval_leakage_symmetry": (
        "conversation_text() includes the assistant turn for a TrainingExample "
        "but is prompt-only for an EvaluationExample. Comparing training "
        "candidates against public eval fixtures therefore weakens near-duplicate "
        "recall exactly where it matters. Gate G13 runs the comparison twice: "
        "once matching public semantics, once prompt-only on both sides."
    ),
    "metadata_extras": (
        "ExampleMetadata is extra='allow' upstream. Gate G14 enforces a closed "
        "allowlist, because an extra key ships inside train.jsonl."
    ),
    "quality_status": (
        "The public loader silently drops non-reviewed examples. We refuse to "
        "promote them at all, so a release cannot look valid and train on nothing."
    ),
}

KNOWN_UPSTREAM_ISSUES: Final[dict[str, str]] = {
    "redact_reprints_short_lines": (
        "scripts/check_no_private_data.py:234-239 truncates the matched secret "
        "to six characters but then prints line.strip()[:60], so any line shorter "
        "than 60 characters reprints the full credential into stdout and from "
        "there into a CI log. Our port excises the matched span instead."
    ),
    "broken_console_script": (
        "pyproject.toml:69-70 declares a `kleos` console script pointing at "
        "kleos_models.cli:main, which does not exist. Not mirrored."
    ),
    "split_fraction_defaults_disagree": (
        "scripts/split_dataset.py defaults to 0.7/0.15/0.15 while SplitConfig "
        "defaults to 0.8/0.1/0.1, so a config-driven and a CLI-driven split of "
        "the same data disagree unless fractions are stated explicitly. We always "
        "state them explicitly and record them in the manifest."
    ),
}
