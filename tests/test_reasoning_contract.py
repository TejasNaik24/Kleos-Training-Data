from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError
from tests.contract_cases import base

from kleos_training_data.contract.constants import REASONING_SCHEMA_VERSION
from kleos_training_data.contract.schemas import Message, TrainingExample
from kleos_training_data.contract.writer import dumps_example
from kleos_training_data.ids import canonicalize, content_hash


def with_reasoning(text: str = "Rank by support first.", fmt: str = "bullets") -> dict[str, Any]:
    payload = base()
    payload["messages"] = [dict(m) for m in payload["messages"]]
    payload["messages"][-1]["reasoning"] = text
    payload["variation_axes"] = {**payload["variation_axes"], "format": fmt}
    return payload


class TestReasoningField:
    def test_an_example_without_reasoning_serializes_as_before(self) -> None:
        line = dumps_example(TrainingExample.model_validate(base()))
        assert "reasoning" not in line
        assert '"name": null' in line
        assert '"version": "1.0"' in line

    def test_reasoning_is_written_and_marks_schema_1_1(self) -> None:
        example = TrainingExample.model_validate(with_reasoning())
        data = json.loads(dumps_example(example))
        assert data["messages"][-1]["reasoning"] == "Rank by support first."
        assert data["messages"][-1]["name"] is None
        assert data["version"] == REASONING_SCHEMA_VERSION == "1.1"

    def test_a_written_example_with_reasoning_reads_back_identically(self) -> None:
        example = TrainingExample.model_validate(with_reasoning())
        line = dumps_example(example)
        again = TrainingExample.model_validate(json.loads(line))
        assert dumps_example(again) == line

    def test_reasoning_is_assistant_only(self) -> None:
        with pytest.raises(ValidationError, match="assistant"):
            Message(role="user", content="x", reasoning="y")

    def test_reasoning_is_never_blank(self) -> None:
        with pytest.raises(ValidationError, match="reasoning"):
            Message(role="assistant", content="x", reasoning="   ")

    def test_reasoning_is_forbidden_on_json_format(self) -> None:
        with pytest.raises(ValidationError, match="json"):
            TrainingExample.model_validate(with_reasoning(fmt="json"))

    def test_schema_1_1_without_reasoning_is_refused(self) -> None:
        with pytest.raises(ValidationError, match=r"1\.1"):
            TrainingExample.model_validate({**base(), "version": "1.1"})

    def test_reasoning_changes_the_content_hash(self) -> None:
        assert content_hash(with_reasoning("a.")) != content_hash(with_reasoning("b."))

    def test_an_example_without_reasoning_hashes_exactly_as_before(self) -> None:
        assert all("reasoning" not in m for m in canonicalize(base())["messages"])

    def test_think_span_stripping_leaves_the_reasoning_field_alone(self) -> None:
        payload = with_reasoning("Rank by support first.")
        payload["messages"][-1]["content"] = "<think>draft</think>Answer."
        example = TrainingExample.model_validate(payload).strip_reasoning_spans()
        assert example.messages[-1].reasoning == "Rank by support first."
        assert example.messages[-1].content == "Answer."
