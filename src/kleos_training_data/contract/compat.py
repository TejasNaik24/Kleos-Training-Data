"""The compatibility handshake with the pinned public repository.

This is what makes the contract mirror defensible rather than merely asserted.
The mirror exists so this repository builds and tests with kleos-models absent;
these checks exist so "we ported it faithfully" is a property somebody verified
rather than a claim in a docstring.

**A skip is a degraded run, not a pass.** ``kleos_models_available()`` returning
``False`` is the normal local state and the correct one — the offline pipeline
must be workable without the public repo checked out. But CI installs the pinned
extra and runs with ``--strict``, where absence is a failure. A compatibility
check that silently passes when it could not run is worse than no check, because
it produces a green build that means nothing.

What is compared, in increasing cost:

1. **Vocabulary** — every mirrored constant, element-wise *and* order-wise. The
   early-warning detector: a newly registered task fails here long before any
   byte-level test notices, because the byte tests only exercise values already
   in use.
2. **Behaviour** — writer bytes, split assignments, ``normalize_text``, and
   accept/reject decisions over a shared corpus.
3. **Artifact** — a real release round-tripped through the public loader and
   validator.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kleos_training_data.contract import constants as mirror
from kleos_training_data.contract.pin import (
    CONTRACT_SOURCE_COMMIT,
    CONTRACT_SOURCE_REPO,
    MIRRORED_CONSTANTS,
)
from kleos_training_data.errors import CompatibilityDriftError

#: Environment variable pointing at a local checkout of the public repo.
ENV_MODELS_PATH = "KLEOS_MODELS_PATH"


def _add_local_checkout_to_path() -> Path | None:
    """Make a local kleos-models checkout importable, if one is configured.

    Development convenience only. CI installs the SHA-pinned package, so what it
    verifies against is exactly the commit in :mod:`.pin` rather than whatever a
    working copy happens to contain.
    """
    configured = os.environ.get(ENV_MODELS_PATH, "").strip()
    if not configured:
        return None
    src = Path(configured).expanduser() / "src"
    if not src.is_dir():
        return None
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def kleos_models_available() -> bool:
    """Whether the public package can be imported."""
    _add_local_checkout_to_path()
    try:
        import kleos_models  # noqa: F401
    except ImportError:
        return False
    return True


def public_version() -> str | None:
    """The installed public package's version, if any."""
    if not kleos_models_available():
        return None
    import kleos_models

    return getattr(kleos_models, "__version__", None)


@dataclass
class CompatCheck:
    """One comparison and its outcome."""

    name: str
    ok: bool
    detail: str = ""

    def render(self) -> str:
        return f"    {'✓' if self.ok else '✗'} {self.name:<34} {self.detail}"


@dataclass
class CompatReport:
    """The full handshake."""

    available: bool
    checks: list[CompatCheck] = field(default_factory=list)
    public_version: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[CompatCheck]:
        return [c for c in self.checks if not c.ok]

    @property
    def ok(self) -> bool:
        return self.available and not self.failures

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(CompatCheck(name, ok, detail))

    def render(self) -> str:
        lines = [
            f"  pinned commit : {CONTRACT_SOURCE_COMMIT[:12]}",
            f"  source repo   : {CONTRACT_SOURCE_REPO}",
            f"  installed     : {self.public_version or '(not installed)'}",
            "",
        ]
        lines += [check.render() for check in self.checks]
        lines += [f"    · {note}" for note in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "ok": self.ok,
            "pinned_commit": CONTRACT_SOURCE_COMMIT,
            "public_version": self.public_version,
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.checks],
            "notes": self.notes,
        }


def compare_vocabularies() -> list[CompatCheck]:
    """Compare every mirrored constant, element-wise and order-wise.

    Order matters even where it looks cosmetic: it decides how ``holdout_values``
    sorts and how a coverage report enumerates cells, so a reordered tuple
    changes an artifact without changing a single value.
    """
    from kleos_models import constants as public

    checks: list[CompatCheck] = []
    for name in MIRRORED_CONSTANTS:
        ours = getattr(mirror, name, None)
        theirs = getattr(public, name, None)
        if theirs is None:
            checks.append(CompatCheck(name, False, "absent from the public package"))
            continue
        if ours != theirs:
            checks.append(CompatCheck(name, False, f"mirror={ours!r} public={theirs!r}"))
            continue
        if isinstance(ours, tuple) and list(ours) != list(theirs):
            checks.append(CompatCheck(name, False, "same members, different order"))
            continue
        checks.append(CompatCheck(name, True, "identical"))
    return checks


def compare_behaviour() -> list[CompatCheck]:
    """Compare the ported functions against the public originals."""
    import random

    from kleos_models.data.leakage import normalize_text as public_normalize
    from kleos_models.data.splitting import _stable_rank as public_rank

    from kleos_training_data.hashing import normalize_text, stable_rank

    checks: list[CompatCheck] = []

    rng = random.Random(1712)
    alphabet = "abcdefghijklmnop-_.:0123456789 "
    keys = ["".join(rng.choices(alphabet, k=rng.randint(1, 60))) for _ in range(500)]
    mismatched = [
        (key, seed)
        for seed in (0, 1, 42, 1712, 2**31 - 1)
        for key in keys
        if stable_rank(key, seed) != public_rank(key, seed)
    ]
    checks.append(
        CompatCheck(
            "stable_rank",
            not mismatched,
            f"{len(keys) * 5} comparisons"
            if not mismatched
            else f"{len(mismatched)} disagreement(s)",
        )
    )

    torture = [
        "Deadline: Mar 3",
        "deadline mar 7",
        "Café RÉSUMÉ — naïve",
        "  multi   space\t\ttabs  ",
        "",
        "!!!???...",
        "2026-08-22T10:00:00Z",
        "日本語のテキスト",
        "mixed 123 numbers 4567 here",
    ]
    bad = [t for t in torture if normalize_text(t) != public_normalize(t)]
    checks.append(
        CompatCheck(
            "normalize_text",
            not bad,
            f"{len(torture)} inputs" if not bad else f"{len(bad)} disagreement(s)",
        )
    )
    return checks


def compare_writer() -> list[CompatCheck]:
    """Compare our JSONL bytes against the public writer's."""
    import tempfile

    from kleos_models.data.loaders import write_jsonl as public_write
    from kleos_models.data.schemas import TrainingExample as PublicExample

    from kleos_training_data.contract.schemas import TrainingExample
    from kleos_training_data.contract.writer import dumps_example

    payload = {
        "id": "kx-npr-3f9a1c8e2b7d0456",
        "task": "notification_prioritization",
        "messages": [
            {"role": "system", "content": "Rank the items."},
            {"role": "user", "content": "Café résumé — 日本語. Which first?"},
            {"role": "assistant", "content": "- Item A first.\n- Item B second."},
        ],
        "variation_axes": {"domain": "career", "urgency": "high", "format": "bullets"},
        "metadata": {"source": "synthetic", "quality_status": "reviewed"},
    }
    ours = dumps_example(TrainingExample.model_validate(payload))
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "x.jsonl"
        public_write([PublicExample.model_validate(payload)], target)
        theirs = target.read_text(encoding="utf-8")

    return [
        CompatCheck(
            "jsonl_writer_bytes",
            ours == theirs,
            "byte-identical" if ours == theirs else f"mirror={ours[:60]!r}",
        )
    ]


def verify_release_against_public(directory: Path | str) -> list[CompatCheck]:
    """Round-trip a real release through the public loader and validator."""
    from kleos_models.config import DatasetConfig
    from kleos_models.data.loaders import load_dataset_bundle
    from kleos_models.data.validation import validate_examples

    checks: list[CompatCheck] = []
    path = Path(directory)

    try:
        bundle = load_dataset_bundle(DatasetConfig(path=path), strict=True)
    except Exception as exc:
        return [CompatCheck("public_loader", False, str(exc).split("\n")[0])]

    checks.append(
        CompatCheck("public_loader", True, f"loaded {len(bundle.train)} train example(s)")
    )

    report = validate_examples(bundle.train, split_name="train", require_reviewed=True)
    checks.append(
        CompatCheck(
            "public_validator",
            report.ok,
            "no errors" if report.ok else "; ".join(f.code for f in report.errors),
        )
    )

    if bundle.manifest is not None:
        expected = bundle.manifest.compute_content_hash()
        matches = bundle.manifest.content_hash == expected
        checks.append(
            CompatCheck(
                "manifest_content_hash",
                matches,
                "public computation agrees" if matches else "disagrees",
            )
        )
    return checks


def run_handshake(release: Path | str | None = None) -> CompatReport:
    """Run every available comparison.

    Returns a report rather than raising, so a caller can decide whether an
    unavailable public package is acceptable. ``--strict`` says it is not.
    """
    available = kleos_models_available()
    report = CompatReport(available=available, public_version=public_version())

    if not available:
        report.notes.append(
            "kleos-models is not installed, so nothing was verified. This is the "
            "normal local state — the offline pipeline is meant to work without "
            "it. Install with: make install-compat"
        )
        return report

    checkout = _add_local_checkout_to_path()
    if checkout is not None:
        report.notes.append(
            f"Using the local checkout at {checkout}, which may differ from the "
            f"pinned commit. CI verifies against the pin itself."
        )

    report.checks.extend(compare_vocabularies())
    report.checks.extend(compare_behaviour())
    report.checks.extend(compare_writer())
    if release is not None:
        report.checks.extend(verify_release_against_public(release))

    return report


def assert_compatible(release: Path | str | None = None) -> CompatReport:
    """Run the handshake and raise on drift.

    Raises:
        CompatibilityDriftError: If the public package is absent or any
            comparison disagrees.
    """
    report = run_handshake(release)
    if not report.available:
        raise CompatibilityDriftError(
            "Cannot verify compatibility: kleos-models is not installed.",
            suggestions=[
                'Install the pinned extra: pip install -e ".[dev,compat]"',
                "A compatibility check that silently passes when it could not run "
                "produces a green build that means nothing.",
            ],
        )
    if report.failures:
        raise CompatibilityDriftError(
            f"{len(report.failures)} compatibility check(s) failed.",
            details={c.name: c.detail for c in report.failures},
            suggestions=[
                "The contract mirror has drifted from the pinned public commit. A "
                "release built now may not load in kleos-models.",
                "Read the public repo's diff, update the mirror, re-run every "
                "differential suite, then move the pin in contract/pin.py.",
                "Never move the pin to make a failing check pass.",
            ],
        )
    return report
