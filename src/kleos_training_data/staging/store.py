"""Atomic, integrity-checked record storage.

Writes go to a temporary file in the destination directory and are then
``os.replace``d into place. On the same filesystem that is atomic, so a crash
mid-write leaves either the old record or the new one — never half of either.
A truncated JSON file in ``staging/`` would fail to parse later, at a point far
from the cause.

Reads recompute the record hash and refuse a mismatch. See
:mod:`kleos_training_data.staging.records` for why that matters.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from kleos_training_data.errors import StagingIntegrityError
from kleos_training_data.staging.records import StagingRecord

RecordT = TypeVar("RecordT", bound=StagingRecord)


def write_record(record: StagingRecord, path: Path | str) -> Path:
    """Seal a record with its hash and write it atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sealed = record.sealed()
    body = json.dumps(sealed.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"

    # Same directory as the target, so os.replace stays on one filesystem and is
    # therefore atomic.
    #
    # The file deliberately outlives its context manager. It is written and
    # closed inside the `with` below, then moved into place by os.replace — a
    # context manager around the whole thing would delete it before the move.
    # The except arm cleans up on any failure.
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return target


def read_record(path: Path | str, model: type[RecordT], *, verify: bool = True) -> RecordT:
    """Read and validate one record.

    Args:
        path: File to read.
        model: Record class to parse into.
        verify: Recompute and check ``record_hash``. Leave this on. Turning it
            off is only appropriate when deliberately inspecting a record known
            to have been hand-edited.

    Raises:
        StagingIntegrityError: If the file is missing, unparseable, or its hash
            no longer matches its content.
    """
    target = Path(path)
    if not target.is_file():
        raise StagingIntegrityError(
            f"Staged record {target.name} is missing.",
            details={"path": str(target)},
            suggestions=["Re-run the stage that produces it, or check --batch."],
        )

    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StagingIntegrityError(
            f"Staged record {target.name} is not valid JSON.",
            details={"path": str(target), "error": str(exc)},
            suggestions=[
                "A truncated file usually means a crash mid-write. Re-run the stage.",
            ],
        ) from exc

    try:
        record = model.model_validate(payload)
    except ValidationError as exc:
        raise StagingIntegrityError(
            f"Staged record {target.name} does not match {model.__name__}.",
            details={"errors": str(exc)},
        ) from exc

    if verify and not record.hash_matches():
        raise StagingIntegrityError(
            f"Staged record {target.name} was modified outside the pipeline.",
            details={"path": str(target), "stored_hash": record.record_hash or "(none)"},
            suggestions=[
                "Every signature and gate result downstream is bound to the exact "
                "bytes that were reviewed. An edited record invalidates them.",
                "Re-run the stage that produced this record rather than editing it.",
                "If the edit was intentional, the candidate needs a fresh review — "
                "its content hash, and therefore its id, has changed.",
            ],
        )
    return record


def iter_records(
    directory: Path | str, model: type[RecordT], *, pattern: str = "*.json", verify: bool = True
) -> Iterator[RecordT]:
    """Yield every record in a directory, in sorted filename order.

    Sorted so a pipeline run over a batch is reproducible — filesystem
    enumeration order is not stable across platforms, and a release built from
    it would not be either.
    """
    root = Path(directory)
    if not root.is_dir():
        return
    for path in sorted(root.glob(pattern)):
        if not is_record_file(path):
            continue
        yield read_record(path, model, verify=verify)


def is_record_file(path: Path) -> bool:
    """Whether a file in a staging directory is a record.

    Two conventions, both needed:

    * A leading underscore marks a directory-level file — ``_batch.json``,
      ``_capture_authorization.json`` — which is metadata about the batch rather
      than a record in it.
    * A second dot marks a sidecar — ``<id>.privacy.json`` — read by its own
      loader alongside the record it annotates.

    Without these, an iterator over a batch tries to parse its own manifest as a
    record and fails with a wall of validation errors that name every field of
    the manifest.
    """
    return not path.name.startswith("_") and path.name.count(".") == 1


def count_records(directory: Path | str, *, pattern: str = "*.json") -> int:
    """How many records a directory holds, without parsing them."""
    root = Path(directory)
    if not root.is_dir():
        return 0
    return sum(1 for path in root.glob(pattern) if is_record_file(path))


def append_index(path: Path | str, entry: dict[str, object]) -> None:
    """Append one line to an append-only ledger.

    Not atomic, and deliberately so: the ledger is an audit trail whose value is
    that it only ever grows. A rewrite-in-place would let a failed run erase the
    record that it happened.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def read_index(path: Path | str) -> list[dict[str, object]]:
    """Read an append-only ledger. Malformed lines are skipped, not fatal."""
    target = Path(path)
    if not target.is_file():
        return []
    entries: list[dict[str, object]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entries.append(json.loads(stripped))
        except json.JSONDecodeError:
            continue
    return entries
