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
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sealed = record.sealed()
    body = json.dumps(sealed.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"

    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def read_record(path: Path | str, model: type[RecordT], *, verify: bool = True) -> RecordT:
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
    root = Path(directory)
    if not root.is_dir():
        return
    for path in sorted(root.glob(pattern)):
        if not is_record_file(path):
            continue
        yield read_record(path, model, verify=verify)


def is_record_file(path: Path) -> bool:
    return not path.name.startswith("_") and path.name.count(".") == 1


def count_records(directory: Path | str, *, pattern: str = "*.json") -> int:
    root = Path(directory)
    if not root.is_dir():
        return 0
    return sum(1 for path in root.glob(pattern) if is_record_file(path))


def append_index(path: Path | str, entry: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def read_index(path: Path | str) -> list[dict[str, object]]:
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
