"""JSONL serialization, byte-identical to the public writer.

The release artifact is bytes on disk, and this is the only thing that produces
them. Ported from ``kleos_models.data.loaders.write_jsonl`` (loaders.py:412-424).

Three details are load-bearing and none of them look it:

``exclude_none=False``
    Every optional field is emitted explicitly as ``null`` — ``"name": null`` on
    every message, and a null for every unset variation axis. Dropping them
    produces smaller, cleaner, *different* files.

``sort_keys=True``
    Keys are alphabetical, not field-declaration order.

``ensure_ascii=False``
    Non-ASCII survives as itself rather than as an escape sequence, so a
    scenario written with real typography round-trips.

A "cleaner" reimplementation changes every line of ``train.jsonl`` and therefore
every file hash and the manifest's content hash.
``tests/test_differential_writer.py`` compares our bytes against the public
writer's over the whole fixture corpus.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from kleos_training_data.contract.schemas import TrainingExample
from kleos_training_data.errors import ContractViolationError


def dumps_example(example: BaseModel | dict[str, Any]) -> str:
    """Serialize one example to a single JSONL line, including its newline."""
    payload = (
        example.model_dump(mode="json", exclude_none=False)
        if isinstance(example, BaseModel)
        else example
    )
    return json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"


def write_jsonl(examples: Iterable[BaseModel | dict[str, Any]], path: Path | str) -> Path:
    """Write examples to JSONL, creating parent directories.

    Returns:
        The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(dumps_example(example))
    return target


def iter_jsonl(path: Path | str) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, object)`` for each non-blank line.

    Blank lines are skipped silently, matching the public loader. A
    pretty-printed JSON array is not valid JSONL and raises here rather than
    producing one enormous "example".
    """
    target = Path(path)
    with target.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ContractViolationError(
                    f"Line {number} of {target} is not valid JSON.",
                    details={"error": str(exc)},
                    suggestions=[
                        "JSONL is one complete JSON object per line.",
                        "A pretty-printed array is not JSONL — re-emit with write_jsonl().",
                    ],
                ) from exc
            if not isinstance(payload, dict):
                raise ContractViolationError(
                    f"Line {number} of {target} is not a JSON object.",
                    details={"found_type": type(payload).__name__},
                )
            yield number, payload


def read_examples(path: Path | str) -> list[TrainingExample]:
    """Parse a JSONL file into validated training examples.

    Strict by design: one invalid line fails the read. This is used to verify a
    sealed release, where "most of the file parsed" is not a useful outcome.
    """
    examples: list[TrainingExample] = []
    for number, payload in iter_jsonl(path):
        try:
            examples.append(TrainingExample.model_validate(payload))
        except Exception as exc:
            raise ContractViolationError(
                f"Line {number} of {path} is not a valid training example.",
                details={"error": str(exc)},
                suggestions=[
                    "The release was written by this pipeline, so this means the "
                    "file changed after it was sealed, or the contract mirror "
                    "drifted. Run: python scripts/verify_release.py --release <dir>",
                ],
            ) from exc
    return examples


def round_trips(example: TrainingExample) -> bool:
    """Whether an example survives serialization and re-parsing unchanged.

    Promotion gate G14 uses this. It catches the case where a value is
    representable in memory but not in JSON — a float that loses precision, a
    dict key that is not a string — which would otherwise surface as a release
    that verifies on the way out and fails on the way in.
    """
    reparsed = TrainingExample.model_validate(json.loads(dumps_example(example)))
    return dumps_example(reparsed) == dumps_example(example)
