from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.contract_cases import base

from kleos_training_data.contract.schemas import TrainingExample
from kleos_training_data.datasets.reasoning_checks import (
    DECLINE_LABELS,
    check_reasoning_release,
)
from kleos_training_data.quality.coverage import build_coverage_report


def line(fmt: str, *, label: str = "impact", reasoning: str | None = "Score each item.") -> str:
    assistant: dict[str, Any] = {"role": "assistant", "name": None}
    if fmt == "json":
        assistant["content"] = '{"ranking": ["A"], "deciding_factor": "impact"}'
    else:
        assistant["content"] = f"1. A — reason.\n\nWhat decided it: {label}. More."
        if reasoning is not None:
            assistant["reasoning"] = reasoning
    version = "1.1" if "reasoning" in assistant else "1.0"
    record = {
        "id": "kx-npr-0000000000000000",
        "version": version,
        "messages": [{"role": "user", "content": "Which first?", "name": None}, assistant],
        "variation_axes": {"format": fmt},
    }
    return json.dumps(record, sort_keys=True) + "\n"


def release(tmp_path: Path, **overrides: str) -> tuple[Path, Path]:
    files = {
        "train.jsonl": line("bullets") + line("prose", label="request_ambiguous"),
        "validation.jsonl": line("bullets", label="missing_input"),
        "test.jsonl": line("json"),
    }
    files.update(overrides)
    directory = tmp_path / "kleos-policy-v0.0.7"
    directory.mkdir()
    for name, text in files.items():
        (directory / name).write_text(text, encoding="utf-8")
    reference = tmp_path / "reference-test.jsonl"
    reference.write_text(line("json"), encoding="utf-8")
    return directory, reference


class TestReasoningReleaseCheck:
    def test_a_conforming_release_passes(self, tmp_path: Path) -> None:
        report = check_reasoning_release(*release(tmp_path))
        assert report.ok, report.problems
        assert report.label_counts["train"]["request_ambiguous"] == 1
        assert report.label_counts["validation"]["missing_input"] == 1

    def test_a_changed_test_file_fails(self, tmp_path: Path) -> None:
        report = check_reasoning_release(*release(tmp_path, **{"test.jsonl": line("json") + "\n"}))
        assert any("test.jsonl" in problem for problem in report.problems)

    def test_reasoning_on_a_json_line_fails(self, tmp_path: Path) -> None:
        bad = json.loads(line("json"))
        bad["messages"][-1]["reasoning"] = "x."
        directory, reference = release(tmp_path)
        (directory / "train.jsonl").write_text(json.dumps(bad) + "\n", encoding="utf-8")
        report = check_reasoning_release(directory, reference)
        assert any("json" in problem for problem in report.problems)

    def test_a_missing_reasoning_fails(self, tmp_path: Path) -> None:
        report = check_reasoning_release(
            *release(tmp_path, **{"train.jsonl": line("bullets", reasoning=None)})
        )
        assert any("reasoning" in problem for problem in report.problems)

    def test_an_overlong_reasoning_fails(self, tmp_path: Path) -> None:
        report = check_reasoning_release(
            *release(tmp_path, **{"train.jsonl": line("bullets", reasoning="x" * 1201)})
        )
        assert any("1200" in problem for problem in report.problems)

    def test_a_missing_label_line_fails(self, tmp_path: Path) -> None:
        unlabeled = json.loads(line("bullets"))
        unlabeled["messages"][-1]["content"] = "1. A — reason."
        directory, reference = release(tmp_path)
        (directory / "train.jsonl").write_text(json.dumps(unlabeled) + "\n", encoding="utf-8")
        report = check_reasoning_release(directory, reference)
        assert any("What decided it" in problem for problem in report.problems)

    def test_a_wrong_schema_version_fails(self, tmp_path: Path) -> None:
        wrong = json.loads(line("bullets"))
        wrong["version"] = "1.0"
        directory, reference = release(tmp_path)
        (directory / "train.jsonl").write_text(json.dumps(wrong) + "\n", encoding="utf-8")
        report = check_reasoning_release(directory, reference)
        assert any("version" in problem for problem in report.problems)

    def test_thin_decline_labels_warn_but_do_not_fail(self, tmp_path: Path) -> None:
        report = check_reasoning_release(*release(tmp_path))
        assert report.ok
        assert set(DECLINE_LABELS) <= {w.split()[0] for w in report.warnings}


class TestCoverageCountsReasoning:
    def test_coverage_reports_reasoning_lengths(self) -> None:
        payload = base()
        payload["messages"] = [dict(m) for m in payload["messages"]]
        payload["messages"][-1]["reasoning"] = "Score each item."
        payload["variation_axes"] = {**payload["variation_axes"], "format": "bullets"}
        report = build_coverage_report([TrainingExample.model_validate(payload)])
        assert report.message_lengths["reasoning_traces"] == 1
        assert report.message_lengths["reasoning_max_chars"] == len("Score each item.")
