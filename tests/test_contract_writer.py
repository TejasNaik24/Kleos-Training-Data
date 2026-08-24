"""The JSONL writer emits the exact bytes the public loader expects.

A release is bytes on disk. Everything downstream — the file hashes, the
manifest content hash, whether the public trainer can read it at all — is
determined here, and every property below is one a "cleaner" reimplementation
would quietly break.
"""

from __future__ import annotations

import json

import pytest
from tests.contract_cases import VALID_CASES, base

from kleos_training_data.contract.schemas import TrainingExample
from kleos_training_data.contract.writer import (
    dumps_example,
    iter_jsonl,
    read_examples,
    round_trips,
    write_jsonl,
)
from kleos_training_data.errors import ContractViolationError


def _example(**overrides) -> TrainingExample:
    return TrainingExample.model_validate(base(**overrides))


class TestByteFormat:
    def test_keys_are_sorted_alphabetically(self) -> None:
        line = dumps_example(_example())
        keys = list(json.loads(line).keys())
        assert keys == sorted(keys)

    def test_nulls_are_emitted_explicitly(self) -> None:
        """`exclude_none=False` is not incidental — it is most of the file.

        Every message carries "name": null and every unset axis carries a null.
        Dropping them produces a smaller, cleaner, *different* file whose hash
        matches nothing.
        """
        payload = json.loads(dumps_example(_example()))
        assert payload["messages"][0]["name"] is None
        assert "urgency" in payload["variation_axes"]
        assert payload["variation_axes"]["urgency"] is None

    def test_non_ascii_survives_unescaped(self) -> None:
        line = dumps_example(
            _example(
                messages=[
                    {"role": "user", "content": "Café résumé 日本語"},
                    {"role": "assistant", "content": "ok"},
                ]
            )
        )
        assert "Café résumé 日本語" in line
        assert "\\u" not in line

    def test_every_line_ends_with_exactly_one_newline(self) -> None:
        line = dumps_example(_example())
        assert line.endswith("\n")
        assert not line.endswith("\n\n")

    def test_a_line_is_a_single_json_object(self) -> None:
        line = dumps_example(_example())
        assert line.count("\n") == 1
        assert isinstance(json.loads(line), dict)


class TestRoundTrip:
    @pytest.mark.parametrize("name", sorted(VALID_CASES))
    def test_every_valid_example_round_trips(self, name: str) -> None:
        example = TrainingExample.model_validate(VALID_CASES[name])
        assert round_trips(example)

    def test_reparsing_yields_identical_bytes(self) -> None:
        example = _example()
        once = dumps_example(example)
        twice = dumps_example(TrainingExample.model_validate(json.loads(once)))
        assert once == twice

    def test_writing_then_reading_preserves_the_corpus(self, tmp_path) -> None:
        examples = [TrainingExample.model_validate(p) for p in VALID_CASES.values()]
        # Ids must be unique for a real release; here we only need parse fidelity.
        path = write_jsonl(examples, tmp_path / "train.jsonl")
        assert [e.model_dump() for e in read_examples(path)] == [e.model_dump() for e in examples]


class TestFileWriting:
    def test_parent_directories_are_created(self, tmp_path) -> None:
        path = write_jsonl([_example()], tmp_path / "deep" / "nested" / "train.jsonl")
        assert path.is_file()

    def test_the_file_has_one_line_per_example(self, tmp_path) -> None:
        examples = [
            _example(
                messages=[
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": "a"},
                ]
            )
            for i in range(5)
        ]
        path = write_jsonl(examples, tmp_path / "train.jsonl")
        assert len(path.read_text(encoding="utf-8").splitlines()) == 5

    def test_writing_is_deterministic(self, tmp_path) -> None:
        """Two writes of the same examples must hash identically."""
        examples = [_example()]
        first = write_jsonl(examples, tmp_path / "a.jsonl").read_bytes()
        second = write_jsonl(examples, tmp_path / "b.jsonl").read_bytes()
        assert first == second


class TestReading:
    def test_blank_lines_are_skipped(self, tmp_path) -> None:
        path = tmp_path / "train.jsonl"
        path.write_text(
            dumps_example(_example()) + "\n\n" + dumps_example(_example()), encoding="utf-8"
        )
        assert len(list(iter_jsonl(path))) == 2

    def test_a_pretty_printed_array_is_rejected(self, tmp_path) -> None:
        """JSONL is one object per line. An array is a common, silent mistake."""
        path = tmp_path / "train.jsonl"
        path.write_text(json.dumps([base()], indent=2), encoding="utf-8")
        with pytest.raises(ContractViolationError, match="not valid JSON"):
            list(iter_jsonl(path))

    def test_a_non_object_line_is_rejected(self, tmp_path) -> None:
        path = tmp_path / "train.jsonl"
        path.write_text('"just a string"\n', encoding="utf-8")
        with pytest.raises(ContractViolationError, match="not a JSON object"):
            list(iter_jsonl(path))

    def test_an_invalid_example_names_its_line(self, tmp_path) -> None:
        path = tmp_path / "train.jsonl"
        path.write_text(
            dumps_example(_example()) + json.dumps({"id": "x", "task": "nope"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ContractViolationError, match="Line 2"):
            read_examples(path)
