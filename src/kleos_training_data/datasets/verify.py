"""Verify a sealed release by re-deriving everything from its bytes.

Deliberately shares no code with the writer beyond parsing. A verifier that
reuses the writer's computations proves the writer is self-consistent, which is
not the question — the question is whether the files on disk say what the
manifest claims.

Every count, distribution and hash is recomputed from the JSONL, then compared
against the manifest and the lock. Anything the writer got wrong, or anything
that changed afterwards, shows up as a mismatch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from kleos_training_data.contract.constants import (
    MANIFEST_FILENAME,
    PROMOTED_QUALITY_STATUS,
    PROVENANCE_FILENAME,
    RELEASE_LOCK_FILENAME,
    REQUIRED_SPLITS,
    SPLIT_FILENAMES,
)
from kleos_training_data.contract.schemas import DatasetManifest, TrainingExample
from kleos_training_data.contract.writer import read_examples
from kleos_training_data.hashing import file_sha256
from kleos_training_data.privacy.sanitize import scan_release_text


@dataclass
class VerificationReport:
    """What verification found."""

    version: str
    path: Path
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_run: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    def add(self, problem: str) -> None:
        self.problems.append(problem)

    def render(self) -> str:
        lines = [f"  release : {self.version}", f"  path    : {self.path}"]
        if self.counts:
            lines.append(f"  counts  : {self.counts}")
        lines.append(f"  checks  : {self.checks_run}")
        for warning in self.warnings:
            lines.append(f"    ! {warning}")
        for problem in self.problems:
            lines.append(f"    ✗ {problem}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "ok": self.ok,
            "checks_run": self.checks_run,
            "counts": self.counts,
            "problems": self.problems,
            "warnings": self.warnings,
        }


def verify_release(directory: Path | str, *, strict: bool = False) -> VerificationReport:
    """Re-derive a release from its bytes and compare against its own claims.

    Args:
        directory: The sealed release.
        strict: Treat warnings as problems.
    """
    path = Path(directory)
    report = VerificationReport(version=path.name, path=path)

    if not path.is_dir():
        report.add(f"{path} is not a directory")
        return report

    # --- structure ---------------------------------------------------------
    report.checks_run += 1
    manifest_path = path / MANIFEST_FILENAME
    if not manifest_path.is_file():
        report.add(f"{MANIFEST_FILENAME} is missing")
        return report

    try:
        manifest = DatasetManifest.model_validate(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        report.add(f"{MANIFEST_FILENAME} is not a valid manifest: {exc}")
        return report

    report.checks_run += 1
    if manifest.version != path.name:
        report.add(f"manifest version {manifest.version!r} does not match directory {path.name!r}")

    # --- filenames ---------------------------------------------------------
    report.checks_run += 1
    allowed = set(SPLIT_FILENAMES.values())
    stray = sorted(p.name for p in path.glob("*.jsonl") if p.name not in allowed)
    if stray:
        report.add(
            f"unexpected JSONL file(s): {stray}. train.py finds splits by exact "
            f"filename, so an alias validates and then fails to train."
        )

    for split in REQUIRED_SPLITS:
        report.checks_run += 1
        if not (path / SPLIT_FILENAMES[split]).is_file():
            report.add(f"{SPLIT_FILENAMES[split]} is required and missing")

    # --- content -----------------------------------------------------------
    by_split: dict[str, list[TrainingExample]] = {}
    for split, filename in SPLIT_FILENAMES.items():
        target = path / filename
        if not target.is_file():
            continue
        report.checks_run += 1
        try:
            by_split[split] = read_examples(target)
        except Exception as exc:
            report.add(f"{filename} does not parse: {exc}")

    if not by_split:
        report.add("the release contains no readable examples")
        return report

    every = [example for examples in by_split.values() for example in examples]
    report.counts = {name: len(examples) for name, examples in by_split.items()}

    # --- counts and distributions, recomputed ------------------------------
    report.checks_run += 1
    if manifest.example_count != len(every):
        report.add(
            f"manifest example_count is {manifest.example_count}, the files hold {len(every)}"
        )

    for name in SPLIT_FILENAMES:
        report.checks_run += 1
        claimed = getattr(manifest.splits, name)
        actual = len(by_split.get(name, []))
        if claimed != actual:
            report.add(f"manifest claims {claimed} {name} example(s), the file holds {actual}")

    from collections import Counter

    recomputed = {
        "task_distribution": dict(sorted(Counter(e.task for e in every).items())),
        "domain_distribution": dict(
            sorted(Counter(e.variation_axes.domain for e in every).items())
        ),
        "source_distribution": dict(sorted(Counter(e.metadata.source for e in every).items())),
        "quality_distribution": dict(
            sorted(Counter(e.metadata.quality_status for e in every).items())
        ),
    }
    for field_name, recomputed_value in recomputed.items():
        report.checks_run += 1
        if getattr(manifest, field_name) != recomputed_value:
            report.add(f"manifest {field_name} does not match the content")

    # --- integrity ---------------------------------------------------------
    report.checks_run += 1
    ids = [e.id for e in every]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        report.add(f"{len(duplicates)} example id(s) appear more than once: {duplicates[:5]}")

    report.checks_run += 1
    for left in by_split:
        for right in by_split:
            if left >= right:
                continue
            overlap = {e.id for e in by_split[left]} & {e.id for e in by_split[right]}
            if overlap:
                report.add(f"{len(overlap)} id(s) appear in both {left} and {right}")

    report.checks_run += 1
    placement: dict[str, str] = {}
    straddled: set[str] = set()
    for name, examples in by_split.items():
        for example in examples:
            group = example.group_key()
            if placement.setdefault(group, name) != name:
                straddled.add(group)
    if straddled:
        report.add(
            f"{len(straddled)} group(s) straddle a split boundary: {sorted(straddled)[:5]}. "
            f"A perturbation pair on both sides makes consistency testing meaningless."
        )

    report.checks_run += 1
    unreviewed = [e.id for e in every if e.metadata.quality_status != PROMOTED_QUALITY_STATUS]
    if unreviewed:
        report.add(
            f"{len(unreviewed)} example(s) are not {PROMOTED_QUALITY_STATUS!r}. The public "
            f"loader filters to reviewed by default, so these would be silently dropped."
        )

    # --- hashes ------------------------------------------------------------
    for filename, claimed_hash in sorted(manifest.file_hashes.items()):
        report.checks_run += 1
        target = path / filename
        if not target.is_file():
            report.add(f"manifest hashes {filename}, which is missing")
            continue
        actual_hash = file_sha256(target)
        if actual_hash != claimed_hash:
            report.add(
                f"{filename} has changed since it was sealed "
                f"(manifest {claimed_hash[:12]}…, actual {actual_hash[:12]}…)"
            )

    report.checks_run += 1
    if manifest.content_hash != manifest.compute_content_hash():
        report.add("manifest content_hash does not match its own file_hashes")

    # --- lock --------------------------------------------------------------
    lock_path = path / RELEASE_LOCK_FILENAME
    if lock_path.is_file():
        report.checks_run += 1
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("content_hash") != manifest.content_hash:
            report.add("RELEASE.lock content_hash disagrees with the manifest")
        for filename, claimed in sorted((lock.get("all_file_hashes") or {}).items()):
            report.checks_run += 1
            target = path / filename
            if not target.is_file():
                report.add(f"RELEASE.lock names {filename}, which is missing")
            elif file_sha256(target) != claimed:
                report.add(f"{filename} has changed since it was sealed (per RELEASE.lock)")
    else:
        report.warnings.append(f"{RELEASE_LOCK_FILENAME} is absent; drift cannot be detected")

    if not (path / PROVENANCE_FILENAME).is_file():
        report.warnings.append(
            f"{PROVENANCE_FILENAME} is absent; the release is not reproducible from itself"
        )

    # --- privacy -----------------------------------------------------------
    for filename in sorted(manifest.file_hashes):
        target = path / filename
        if not target.is_file():
            continue
        report.checks_run += 1
        detections = [
            d
            for d in scan_release_text(target.read_text(encoding="utf-8"))
            if d.severity in {"block", "redact"}
        ]
        if detections:
            kinds = sorted({d.kind for d in detections})
            if manifest.contains_private_data:
                report.warnings.append(
                    f"{filename} carries {kinds}, and the manifest declares private data"
                )
            else:
                report.add(
                    f"{filename} carries privacy detections {kinds} but the manifest "
                    f"declares contains_private_data=false"
                )

    if strict and report.warnings:
        report.problems.extend(f"(strict) {w}" for w in report.warnings)

    return report
