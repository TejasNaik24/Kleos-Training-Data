"""What the contract mirror is pinned to, and why it is a mirror at all.

The public repository's ``docs/privacy.md`` states: *"Neither repository imports
the other."* This module is how that is honoured without guessing at the
contract.

The mirror in :mod:`kleos_training_data.contract` is a faithful port of the
public models, the JSONL writer, the splitter and the duplicate detectors. It
exists so this repository builds, tests and produces a release with the public
package absent from the machine entirely.

The obvious objection is that a mirror drifts. It does — on day ninety, not day
one. The answer is not to couple the repositories but to *test* the mirror:
``tests/test_differential_*.py`` install the pinned public package and assert our
output is identical to it, byte for byte and assignment for assignment. Coupling
would prevent drift by making this repository unbuildable whenever the public one
is mid-refactor. Testing catches drift without that cost, and gives a named
failure instead of a silent divergence.

Updating the pin is a deliberate act:

1. Read the public repo's diff between the old and new commit.
2. Update the mirror to match.
3. Run ``pytest -m requires_kleos_models`` — all four differential suites must
   pass, not merely the ones you thought were affected.
4. Update :data:`CONTRACT_SOURCE_COMMIT` and :data:`CONTRACT_VERIFIED_AT` here.
5. Record the decision in ``docs/compatibility.md``.

Never update the pin to make a failing test pass. A failing differential test
means a release built today may not load in the public repo tomorrow, which is
exactly what the pin exists to surface.
"""

from __future__ import annotations

from typing import Final

#: The repository the contract is mirrored from.
CONTRACT_SOURCE_REPO: Final[str] = "https://github.com/TejasNaik24/Kleos-Models"

#: The exact commit the mirror was ported from and verified against.
#:
#: A commit, never a branch or a tag. A moving reference would let the public
#: contract change underneath a release without any test failing — which is the
#: single failure mode this whole arrangement exists to prevent.
CONTRACT_SOURCE_COMMIT: Final[str] = "12361d53cf329c897ec14daede6d14112a3e8f20"

#: When the differential suites last passed against that commit.
CONTRACT_VERIFIED_AT: Final[str] = "2026-08-22"

#: Public constants asserted to be identical, by name. Compared element-wise
#: *and* order-wise by tests/test_contract_pin.py.
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

#: Behaviours the mirror reproduces exactly, and the public source of each.
#: Named here so a reviewer can find the original without searching.
MIRRORED_BEHAVIOURS: Final[dict[str, str]] = {
    "jsonl_writer_bytes": "kleos_models.data.loaders.write_jsonl (loaders.py:412-424)",
    "split_stable_rank": "kleos_models.data.splitting._stable_rank (splitting.py:86-94)",
    "split_holdout_routing": "kleos_models.data.splitting._split_holdout (splitting.py:192-292)",
    "leakage_normalize_text": "kleos_models.data.leakage.normalize_text (leakage.py:95-108)",
    "conversation_text": "kleos_models.data.schemas.TrainingExample.conversation_text",
    "manifest_content_hash": "kleos_models.data.schemas.DatasetManifest.compute_content_hash",
}

#: Places where this repository is deliberately STRICTER than the public repo.
#:
#: A mirror that is stricter is safe — anything we accept, the public repo
#: accepts. A mirror that is *looser* is how a release passes here and fails
#: there, so the differential tests assert the direction of each delta rather
#: than merely asserting equality.
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

#: Known defects in the pinned public repo that this mirror does not reproduce.
#: Reproducing a bug for fidelity's sake would be the wrong kind of faithful.
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
