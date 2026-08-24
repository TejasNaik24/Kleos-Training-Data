"""Manifest and provenance.

Two files, and the split between them is forced by the contract rather than
chosen: ``DatasetManifest`` is ``extra="forbid"``, so one added key makes the
public ``load_manifest`` raise ``DatasetIntegrityError`` at train time.

Everything this repository wants to record and the manifest cannot hold — ruleset
versions, gate report digests, the scenario catalog fingerprint, review counts,
the holdout rationale — goes in a sibling ``provenance.json``. The public loader
ignores unknown *files* in a release directory, so the sidecar is free.

``file_hashes`` covers only the shipped split JSONLs. Adding ``provenance.json``
to it would make our ``content_hash`` differ from what the public
``split_dataset.py`` computes for identical splits, and that hash is the cheapest
way to confirm two repositories are talking about the same data.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kleos_training_data.contract.constants import (
    DATASET_SCHEMA_VERSION,
    PIPELINE_VERSION,
    PREPROCESSING_VERSION,
    SPLIT_FILENAMES,
)
from kleos_training_data.contract.schemas import DatasetManifest, SplitCounts, TrainingExample
from kleos_training_data.contract.splitting import SplitResult
from kleos_training_data.datasets.holdouts import HoldoutPlan
from kleos_training_data.hashing import canonical_hash, file_sha256
from kleos_training_data.privacy.rules import RULESET_VERSION
from kleos_training_data.review.rubric import RUBRIC_VERSION


def _distribution(examples: list[TrainingExample], attribute: str) -> dict[str, int]:
    """Count examples by one attribute, sorted for a stable manifest."""
    getters = {
        "task": lambda e: e.task,
        "domain": lambda e: e.variation_axes.domain,
        "source": lambda e: e.metadata.source,
        "quality_status": lambda e: e.metadata.quality_status,
    }
    counts = Counter(getters[attribute](e) for e in examples)
    return dict(sorted(counts.items()))


def build_manifest(
    *,
    version: str,
    split: SplitResult,
    directory: Path,
    description: str = "",
    contains_private_data: bool = False,
    notes: str | None = None,
    created_at: str | None = None,
) -> DatasetManifest:
    """Build the public manifest for a written release.

    ``file_hashes`` is computed from the files that were actually written, not
    from what was intended, so the manifest describes bytes on disk.
    """
    all_examples = split.train + split.validation + split.test

    file_hashes = {
        filename: file_sha256(directory / filename)
        for filename in SPLIT_FILENAMES.values()
        if (directory / filename).is_file()
    }

    manifest = DatasetManifest(
        version=version,
        schema_version=DATASET_SCHEMA_VERSION,
        preprocessing_version=PREPROCESSING_VERSION,
        created_at=created_at or datetime.now(UTC).isoformat(),
        source="synthetic",
        description=description,
        example_count=len(all_examples),
        splits=SplitCounts(**split.counts),
        task_distribution=_distribution(all_examples, "task"),
        domain_distribution=_distribution(all_examples, "domain"),
        source_distribution=_distribution(all_examples, "source"),
        quality_distribution=_distribution(all_examples, "quality_status"),
        split_strategy=split.strategy,
        split_seed=split.seed,
        holdout_values=sorted(split.holdout_values),
        file_hashes=file_hashes,
        contains_private_data=contains_private_data,
        notes=notes
        or (
            "Reproduction details are in provenance.json alongside this file. "
            "The manifest schema forbids extra keys, so they cannot live here."
        ),
    )
    return manifest.finalize()


def build_provenance(
    *,
    version: str,
    split: SplitResult,
    holdout: HoldoutPlan,
    manifest: DatasetManifest,
    examples: list[TrainingExample],
    scenario_fingerprints: dict[str, str],
    gate_report_hash: str | None = None,
    contract_commit: str | None = None,
) -> dict[str, Any]:
    """Everything needed to re-derive this release, and nothing the loader reads.

    Deliberately excludes anything that points back into ``staging/``. A release
    is shared; a pointer into the staging zone would survive into wherever it
    goes.
    """
    from kleos_training_data.contract.pin import CONTRACT_SOURCE_COMMIT, CONTRACT_SOURCE_REPO

    families = Counter(e.metadata.scenario_family or "(none)" for e in examples)
    perturbations = Counter(e.metadata.perturbation_kind or "(base)" for e in examples)
    groups = {e.group_key() for e in examples}

    return {
        "version": version,
        "content_hash": manifest.content_hash,
        "pipeline_version": PIPELINE_VERSION,
        "versions": {
            "dataset_schema": DATASET_SCHEMA_VERSION,
            "preprocessing": PREPROCESSING_VERSION,
            "privacy_ruleset": RULESET_VERSION,
            "review_rubric": RUBRIC_VERSION,
        },
        "contract": {
            "source_repo": (contract_commit and CONTRACT_SOURCE_REPO) or CONTRACT_SOURCE_REPO,
            "pinned_commit": contract_commit or CONTRACT_SOURCE_COMMIT,
        },
        "split": {
            **split.to_dict(),
            "group_count": len(groups),
            "fractions_stated_explicitly": True,
        },
        "holdout": holdout.to_dict(),
        "composition": {
            "scenario_families": dict(sorted(families.items())),
            "perturbation_kinds": dict(sorted(perturbations.items())),
            "distinct_groups": len(groups),
        },
        "scenario_fingerprints": dict(sorted(scenario_fingerprints.items())),
        "gate_report_hash": gate_report_hash,
        "note": (
            "This file is not read by kleos-models. It exists because "
            "DatasetManifest forbids extra keys, and a release still has to be "
            "reproducible from its source material."
        ),
    }


def provenance_hash(provenance: dict[str, Any]) -> str:
    """Digest over the provenance sidecar, for the release lock."""
    return canonical_hash(provenance)
