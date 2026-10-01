from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from kleos_training_data.contract.constants import DATASET_SCHEMA_VERSION, REASONING_SCHEMA_VERSION
from kleos_training_data.scenarios.reasoning import MAX_REASONING_CHARS

__all__ = [
    "DECLINE_LABELS",
    "MAX_REASONING_CHARS",
    "MIN_TRAIN_LABEL_COUNT",
    "MIN_VALIDATION_LABEL_COUNT",
    "ReasoningReleaseReport",
    "check_reasoning_release",
]

MIN_TRAIN_LABEL_COUNT: Final[int] = 20

MIN_VALIDATION_LABEL_COUNT: Final[int] = 5

DECLINE_LABELS: Final[tuple[str, ...]] = (
    "request_ambiguous",
    "missing_input",
    "stale_explicit_conflict",
    "insufficient_separation",
)

_LABEL_LINE: Final[re.Pattern[str]] = re.compile(r"What decided it: ([a-z_]+)\.")

_SPLITS: Final[tuple[str, ...]] = ("train", "validation", "test")


@dataclass
class ReasoningReleaseReport:
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    label_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    reasoning_lengths: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def render(self) -> str:
        lines = [f"  problems : {len(self.problems)}", f"  warnings : {len(self.warnings)}"]
        lines += [f"    ✗ {problem}" for problem in self.problems[:20]]
        lines += [f"    ! {warning}" for warning in self.warnings]
        if self.reasoning_lengths:
            ordered = sorted(self.reasoning_lengths)
            lines.append(
                f"  reasoning: {len(ordered)} traces, median {ordered[len(ordered) // 2]}, "
                f"max {ordered[-1]} chars"
            )
        lines.append("  decline labels (train / validation):")
        for label in DECLINE_LABELS:
            train = self.label_counts.get("train", {}).get(label, 0)
            validation = self.label_counts.get("validation", {}).get(label, 0)
            lines.append(f"    {label:<26} {train:>4} / {validation:>3}")
        return "\n".join(lines)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_record(
    split: str, number: int, record: dict[str, Any], report: ReasoningReleaseReport
) -> None:
    where = f"{split}.jsonl line {number}"
    assistant = record["messages"][-1]
    fmt = (record.get("variation_axes") or {}).get("format")
    reasoning = assistant.get("reasoning")

    if fmt == "json":
        if "reasoning" in assistant:
            report.problems.append(f"{where}: reasoning on a json-format example")
    elif not isinstance(reasoning, str) or not reasoning.strip():
        report.problems.append(f"{where}: no reasoning on a non-json example")
    else:
        report.reasoning_lengths.append(len(reasoning))
        if len(reasoning) > MAX_REASONING_CHARS:
            report.problems.append(
                f"{where}: reasoning is {len(reasoning)} chars (cap {MAX_REASONING_CHARS})"
            )

    expected_version = (
        REASONING_SCHEMA_VERSION if "reasoning" in assistant else DATASET_SCHEMA_VERSION
    )
    if record.get("version") != expected_version:
        report.problems.append(
            f"{where}: version {record.get('version')!r}, expected {expected_version!r}"
        )

    if split != "test":
        found = _LABEL_LINE.search(str(assistant.get("content", "")))
        if found is None:
            report.problems.append(f"{where}: no 'What decided it: <label>.' line")
        else:
            report.label_counts.setdefault(split, {})
            counts = report.label_counts[split]
            counts[found.group(1)] = counts.get(found.group(1), 0) + 1


def check_reasoning_release(release_dir: Path, reference_test: Path) -> ReasoningReleaseReport:
    report = ReasoningReleaseReport()
    test_path = release_dir / "test.jsonl"
    if _sha256(test_path) != _sha256(reference_test):
        report.problems.append(
            f"test.jsonl differs from the reference ({_sha256(test_path)[:16]} vs "
            f"{_sha256(reference_test)[:16]})"
        )

    for split in _SPLITS:
        path = release_dir / f"{split}.jsonl"
        if not path.exists():
            report.problems.append(f"{split}.jsonl is missing")
            continue
        report.label_counts.setdefault(split, {})
        for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if text.strip():
                _check_record(split, number, json.loads(text), report)

    minimums = {"train": MIN_TRAIN_LABEL_COUNT, "validation": MIN_VALIDATION_LABEL_COUNT}
    for label in DECLINE_LABELS:
        for split, minimum in minimums.items():
            count = report.label_counts.get(split, {}).get(label, 0)
            if count < minimum:
                report.warnings.append(
                    f"{label} has {count} {split} example(s), below the target of {minimum}"
                )
    return report
