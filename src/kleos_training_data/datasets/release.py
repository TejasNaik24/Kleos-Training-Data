"""Writing a release, and sealing it so it stays what it was.

Assembly happens in a scratch directory and is moved into place with a single
``os.replace``. On one filesystem that is atomic, so a crash mid-write can never
leave a partial release visible under its version name — the failure mode where
a directory exists, looks plausible, and is missing three examples.

There is deliberately **no ``--force``**. Dataset versions are immutable: every
comparison, every experiment manifest and every trained checkpoint that named
``kleos-policy-v0.1.0`` meant one specific set of bytes. If the content needs to
change, the version string needs to change.

File permissions (``0o444``/``0o555``) are a speed bump, not a guarantee —
anyone can ``chmod``. ``RELEASE.lock`` plus :mod:`.verify` is the actual
guarantee, and CI re-verifies every release directory on every run so post-hoc
drift fails a build rather than surviving quietly.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kleos_training_data.contract.constants import (
    MANIFEST_FILENAME,
    PROVENANCE_FILENAME,
    RELEASE_LOCK_FILENAME,
    SPLIT_FILENAMES,
)
from kleos_training_data.contract.schemas import DatasetManifest
from kleos_training_data.contract.splitting import SplitResult
from kleos_training_data.contract.writer import read_examples, write_jsonl
from kleos_training_data.errors import PrivacyViolationError, ReleaseImmutabilityError
from kleos_training_data.hashing import file_sha256
from kleos_training_data.logging_utils import get_logger
from kleos_training_data.paths import Workspace
from kleos_training_data.privacy.sanitize import scan_release_text

logger = get_logger(__name__)

#: Read-only for everyone once sealed.
FILE_MODE = 0o444
DIR_MODE = 0o555


@dataclass
class SealedRelease:
    """A finished release directory."""

    version: str
    path: Path
    manifest: DatasetManifest
    provenance: dict[str, Any]
    file_hashes: dict[str, str]

    @property
    def content_hash(self) -> str:
        return self.manifest.content_hash or ""


class ReleaseWriter:
    """Assemble, verify and seal one dataset version."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def seal(
        self,
        *,
        version: str,
        split: SplitResult,
        provenance_builder,
        description: str = "",
        declare_private: bool = False,
    ) -> SealedRelease:
        """Write, verify and seal a release. Refuses to overwrite an existing one.

        Raises:
            ReleaseImmutabilityError: If the version already exists.
            PrivacyViolationError: If the final byte scan finds anything.
        """
        from kleos_training_data.datasets.manifest import build_manifest

        target = self.workspace.release(version)
        if target.exists():
            raise ReleaseImmutabilityError(
                f"Release {version} already exists.",
                details={"path": str(target)},
                suggestions=[
                    "Dataset versions are immutable, and there is deliberately no "
                    "--force. Every comparison and every trained checkpoint that "
                    "named this version meant one specific set of bytes.",
                    "If the content changed, the version string changes: bump to "
                    "the next patch or minor version.",
                ],
            )

        staging = self.workspace.release_staging(uuid.uuid4().hex[:12])
        staging.mkdir(parents=True, exist_ok=False)

        try:
            written = self._write_splits(staging, split)
            manifest = build_manifest(
                version=version,
                split=split,
                directory=staging,
                description=description,
                contains_private_data=declare_private,
            )

            # Re-read what was written, not what was intended. A truncated write
            # produces a file that parses as fewer examples, and the manifest
            # built from memory would not notice.
            self._verify_written(staging, split, written)

            # Byte-level privacy scan of the final artifact. Everything upstream
            # scanned payloads; this scans the file that ships.
            self._scan_bytes(staging, written, declare_private=declare_private)

            provenance = provenance_builder(manifest)
            (staging / PROVENANCE_FILENAME).write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (staging / MANIFEST_FILENAME).write_text(
                json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
                encoding="utf-8",
            )

            lock = self._build_lock(staging, version, manifest)
            (staging / RELEASE_LOCK_FILENAME).write_text(
                json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        self._make_read_only(target)
        logger.info("Sealed release %s at %s", version, target)

        return SealedRelease(
            version=version,
            path=target,
            manifest=manifest,
            provenance=provenance,
            file_hashes=dict(manifest.file_hashes),
        )

    # --- internals ---------------------------------------------------------

    def _write_splits(self, directory: Path, split: SplitResult) -> list[str]:
        """Write each non-empty split under its canonical filename.

        Only the canonical names are ever emitted. The public
        ``validate_dataset.py`` also accepts ``synthetic_train.jsonl`` and
        ``valid.jsonl``, but ``train.py`` does not — so an alias produces a
        release that validates and then fails to train.
        """
        written: list[str] = []
        for name, filename in SPLIT_FILENAMES.items():
            examples = split.split(name)
            if not examples:
                continue
            write_jsonl(examples, directory / filename)
            written.append(filename)
        return written

    def _verify_written(self, directory: Path, split: SplitResult, written: list[str]) -> None:
        """Re-parse every file and confirm it holds what it should."""
        for name, filename in SPLIT_FILENAMES.items():
            if filename not in written:
                continue
            reparsed = read_examples(directory / filename)
            expected = split.split(name)
            if len(reparsed) != len(expected):
                raise ReleaseImmutabilityError(
                    f"{filename} holds {len(reparsed)} example(s), expected {len(expected)}.",
                    suggestions=["A truncated write. The release was not sealed."],
                )
            if [e.id for e in reparsed] != [e.id for e in expected]:
                raise ReleaseImmutabilityError(
                    f"{filename} does not hold the examples it was given.",
                )

    def _scan_bytes(self, directory: Path, written: list[str], *, declare_private: bool) -> None:
        """Scan the written files themselves for anything sensitive."""
        for filename in written:
            text = (directory / filename).read_text(encoding="utf-8")
            detections = [d for d in scan_release_text(text) if d.severity in {"block", "redact"}]
            if detections and not declare_private:
                raise PrivacyViolationError(
                    f"{len(detections)} privacy detection(s) in the written {filename}.",
                    details={"rules": ", ".join(sorted({d.rule_id for d in detections}))},
                    suggestions=[
                        "Every promoted example passed the privacy gates, so this "
                        "means something was introduced during assembly, or a rule "
                        "changed since promotion.",
                        "The release was not sealed. See docs/incident-response.md.",
                    ],
                )

    def _build_lock(
        self, directory: Path, version: str, manifest: DatasetManifest
    ) -> dict[str, Any]:
        """The lock file. This, not the permissions, is the actual guarantee."""
        from kleos_training_data.contract.pin import CONTRACT_SOURCE_COMMIT

        every_file = {
            path.name: file_sha256(path)
            for path in sorted(directory.iterdir())
            if path.is_file() and path.name != RELEASE_LOCK_FILENAME
        }
        return {
            "version": version,
            "sealed_at": datetime.now(UTC).isoformat(),
            "content_hash": manifest.content_hash,
            "manifest_file_hashes": dict(sorted(manifest.file_hashes.items())),
            # Covers provenance.json too, which manifest.file_hashes deliberately
            # does not — so drift in the sidecar is detectable without changing
            # the hash the public repo computes.
            "all_file_hashes": every_file,
            "pinned_contract_commit": CONTRACT_SOURCE_COMMIT,
        }

    def _make_read_only(self, directory: Path) -> None:
        for path in sorted(directory.iterdir()):
            if path.is_file():
                path.chmod(FILE_MODE)
        directory.chmod(DIR_MODE)


def open_for_rewrite(path: Path) -> None:
    """Restore write permissions on a sealed release.

    Exists for tests and for a deliberate operator action such as deleting a
    quarantined release. It is not part of any pipeline path — nothing in this
    repository calls it to *modify* a release, because modifying one is the thing
    immutability forbids.
    """
    path.chmod(stat.S_IMODE(path.stat().st_mode) | 0o700)
    for child in path.iterdir():
        if child.is_file():
            child.chmod(0o644)
