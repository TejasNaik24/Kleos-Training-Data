from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from kleos_training_data.contract.schemas import TrainingExample
from kleos_training_data.errors import ContractViolationError


def dumps_example(example: BaseModel | dict[str, Any]) -> str:
    payload = (
        example.model_dump(mode="json", exclude_none=False)
        if isinstance(example, BaseModel)
        else example
    )
    return json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"


def write_jsonl(examples: Iterable[BaseModel | dict[str, Any]], path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(dumps_example(example))
    return target


def iter_jsonl(path: Path | str) -> Iterator[tuple[int, dict[str, Any]]]:
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
    reparsed = TrainingExample.model_validate(json.loads(dumps_example(example)))
    return dumps_example(reparsed) == dumps_example(example)
