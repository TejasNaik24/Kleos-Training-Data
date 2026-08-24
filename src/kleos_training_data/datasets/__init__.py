"""Dataset assembly: holdouts, splitting, manifests, and the sealed release.

A release is the interface to kleos-models — a directory, consumed by path. It
is immutable by construction: there is no --force, and sealing refuses an
existing version.
"""

from __future__ import annotations

from kleos_training_data.datasets.holdouts import (
    STRATEGY_BY_ATTRIBUTE,
    HoldoutPlan,
    collect_declarations,
    resolve_holdouts,
)
from kleos_training_data.datasets.manifest import (
    build_manifest,
    build_provenance,
    provenance_hash,
)
from kleos_training_data.datasets.release import (
    ReleaseWriter,
    SealedRelease,
    open_for_rewrite,
)
from kleos_training_data.datasets.verify import VerificationReport, verify_release

__all__ = [
    "STRATEGY_BY_ATTRIBUTE",
    "HoldoutPlan",
    "ReleaseWriter",
    "SealedRelease",
    "VerificationReport",
    "build_manifest",
    "build_provenance",
    "collect_declarations",
    "open_for_rewrite",
    "provenance_hash",
    "resolve_holdouts",
    "verify_release",
]
