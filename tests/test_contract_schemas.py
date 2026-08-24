"""The contract mirror accepts and rejects what the public repo does.

These tests run everywhere. Their differential counterpart —
``test_differential_schema_decisions.py`` — asserts the pinned public models
agree with every verdict here, and runs only where kleos-models is installed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from tests.contract_cases import INVALID_CASES, VALID_CASES, base

from kleos_training_data.contract.constants import (
    QUALITY_STATUSES,
    SOURCE_TYPES,
    SUPPORTED_TASKS,
)
from kleos_training_data.contract.schemas import (
    DatasetManifest,
    Message,
    TrainingExample,
    VariationAxes,
    contains_reasoning,
    strip_reasoning,
)


class TestValidExamples:
    @pytest.mark.parametrize("name", sorted(VALID_CASES))
    def test_accepted(self, name: str) -> None:
        example = TrainingExample.model_validate(VALID_CASES[name])
        assert example.id
        assert example.messages[-1].role == "assistant"

    def test_version_defaults_to_the_schema_version(self) -> None:
        assert TrainingExample.model_validate(base()).version == "1.0"

    def test_domain_is_mirrored_from_the_axes_when_absent(self) -> None:
        """The public writer emits `domain` explicitly, so the auto-fill changes bytes."""
        example = TrainingExample.model_validate(base(variation_axes={"domain": "research"}))
        assert example.domain == "research"

    def test_metadata_defaults_are_the_conservative_ones(self) -> None:
        metadata = TrainingExample.model_validate(base()).metadata
        assert metadata.source == "synthetic"
        assert metadata.quality_status == "draft", (
            "An example must not be born 'reviewed'. The public loader filters to "
            "reviewed by default, so a permissive default would let unreviewed "
            "data train silently."
        )


class TestRejectedExamples:
    @pytest.mark.parametrize("name", sorted(INVALID_CASES))
    def test_rejected(self, name: str) -> None:
        payload, _expected = INVALID_CASES[name]
        with pytest.raises(ValidationError):
            TrainingExample.model_validate(payload)

    @pytest.mark.parametrize("name", sorted(INVALID_CASES))
    def test_the_error_names_the_problem(self, name: str) -> None:
        """An error a reviewer cannot act on is barely better than no error."""
        payload, expected = INVALID_CASES[name]
        with pytest.raises(ValidationError) as caught:
            TrainingExample.model_validate(payload)
        assert expected.lower() in str(caught.value).lower(), (
            f"{name}: error did not mention {expected!r}:\n{caught.value}"
        )


class TestExtraFieldPolicy:
    """Closed where the public repo is closed, open where it is open.

    Getting either backwards is silent: a closed model that should be open
    rejects a legitimate release, and an open model that should be closed ships
    fields the public loader then refuses.
    """

    def test_example_is_closed(self) -> None:
        with pytest.raises(ValidationError):
            TrainingExample.model_validate(base(capture_id="cap-123"))

    def test_message_is_closed(self) -> None:
        with pytest.raises(ValidationError):
            Message.model_validate({"role": "user", "content": "hi", "ts": 1})

    def test_variation_axes_is_open(self) -> None:
        axes = VariationAxes.model_validate({"domain": "career", "pilot_axis": "v1"})
        assert axes.as_dict()["pilot_axis"] == "v1"

    def test_metadata_is_open(self) -> None:
        """Open upstream — which is exactly why promotion gate G14 closes it.

        Anything here ships inside train.jsonl, so a stray capture_id would be a
        pointer back into staging/ surviving into a shared artifact.
        """
        example = TrainingExample.model_validate(
            base(metadata={"source": "synthetic", "leaked": "capture-123"})
        )
        assert example.metadata.model_extra == {"leaked": "capture-123"}

    def test_manifest_is_closed(self) -> None:
        """One extra key makes the public loader raise at train time."""
        with pytest.raises(ValidationError):
            DatasetManifest.model_validate({"version": "v1", "provenance": {}})


class TestVocabularies:
    @pytest.mark.parametrize("task", SUPPORTED_TASKS)
    def test_every_registered_task_is_accepted(self, task: str) -> None:
        assert TrainingExample.model_validate(base(task=task)).task == task

    @pytest.mark.parametrize("source", SOURCE_TYPES)
    def test_every_registered_source_is_accepted(self, source: str) -> None:
        example = TrainingExample.model_validate(base(metadata={"source": source}))
        assert example.metadata.source == source

    @pytest.mark.parametrize("status", QUALITY_STATUSES)
    def test_every_registered_status_is_accepted(self, status: str) -> None:
        example = TrainingExample.model_validate(base(metadata={"quality_status": status}))
        assert example.metadata.quality_status == status


class TestDerivedViews:
    def test_group_key_falls_back_through_group_id_then_family_then_id(self) -> None:
        """The fallback chain is what guarantees every example is in exactly one group."""
        payload = base(metadata={"group_id": "g1", "scenario_family": "fam"})
        assert TrainingExample.model_validate(payload).group_key() == "g1"

        payload = base(metadata={"scenario_family": "fam"})
        assert TrainingExample.model_validate(payload).group_key() == "fam"

        assert TrainingExample.model_validate(base()).group_key() == base()["id"]

    def test_group_key_can_name_a_variation_axis(self) -> None:
        payload = base(variation_axes={"domain": "career", "entities": "set_a"})
        assert TrainingExample.model_validate(payload).group_key("entities") == "set_a"

    def test_conversation_text_can_exclude_the_assistant_turn(self) -> None:
        example = TrainingExample.model_validate(base())
        assert "assistant:" in example.conversation_text()
        assert "assistant:" not in example.conversation_text(include_assistant=False)

    def test_content_hash_ignores_id_and_metadata(self) -> None:
        """The public notion of content identity, used for dedup parity."""
        a = TrainingExample.model_validate(base(id="kx-npr-1111111111111111"))
        b = TrainingExample.model_validate(
            base(id="kx-npr-2222222222222222", metadata={"notes": "different"})
        )
        assert a.content_hash() == b.content_hash()

    def test_content_hash_changes_with_the_conversation(self) -> None:
        a = TrainingExample.model_validate(base())
        b = TrainingExample.model_validate(
            base(
                messages=[
                    {"role": "user", "content": "Two items are open. Which first?"},
                    {"role": "assistant", "content": "Item B."},
                ]
            )
        )
        assert a.content_hash() != b.content_hash()


class TestReasoningStripping:
    """Reasoning spans never become training targets."""

    def test_a_well_formed_block_is_removed(self) -> None:
        assert strip_reasoning("<think>weighing options</think>Item A.") == "Item A."

    def test_a_dangling_close_tag_is_removed(self) -> None:
        """Qwen Thinking templates pre-open <think>, so only the close tag appears."""
        assert strip_reasoning("weighing options</think>Item A.") == "Item A."

    def test_plain_text_is_untouched(self) -> None:
        assert strip_reasoning("Item A.") == "Item A."

    def test_detection_is_case_insensitive(self) -> None:
        assert contains_reasoning("x</THINK>y")

    def test_stripping_an_example_leaves_other_turns_alone(self) -> None:
        example = TrainingExample.model_validate(
            base(
                messages=[
                    {"role": "user", "content": "<think>not mine</think>Which first?"},
                    {"role": "assistant", "content": "<think>weighing</think>Item A."},
                ]
            )
        )
        stripped = example.strip_reasoning_spans()
        assert stripped.messages[0].content == "<think>not mine</think>Which first?"
        assert stripped.messages[1].content == "Item A."


class TestDatasetManifest:
    def test_only_version_is_required(self) -> None:
        assert DatasetManifest.model_validate({"version": "kleos-policy-v0.1.0"}).example_count == 0

    def test_content_hash_covers_file_hashes_only(self) -> None:
        """Covering anything else would diverge from the public split_dataset.py."""
        a = DatasetManifest(version="v1", file_hashes={"train.jsonl": "abc"})
        b = DatasetManifest(
            version="v2", description="different", file_hashes={"train.jsonl": "abc"}
        )
        assert a.compute_content_hash() == b.compute_content_hash()

    def test_content_hash_changes_when_a_file_hash_does(self) -> None:
        a = DatasetManifest(version="v1", file_hashes={"train.jsonl": "abc"})
        b = DatasetManifest(version="v1", file_hashes={"train.jsonl": "def"})
        assert a.compute_content_hash() != b.compute_content_hash()

    def test_finalize_populates_the_content_hash(self) -> None:
        manifest = DatasetManifest(version="v1", file_hashes={"train.jsonl": "abc"}).finalize()
        assert manifest.content_hash == manifest.compute_content_hash()

    def test_contains_private_data_defaults_to_false(self) -> None:
        assert DatasetManifest(version="v1").contains_private_data is False
