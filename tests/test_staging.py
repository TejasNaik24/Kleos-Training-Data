from __future__ import annotations

import json

import pytest

from kleos_training_data.errors import ContractViolationError, StagingIntegrityError
from kleos_training_data.staging.normalize import normalize_capture
from kleos_training_data.staging.reasons import (
    PRIVACY_REASONS,
    RETRYABLE_REASONS,
    RejectionReason,
)
from kleos_training_data.staging.records import (
    CaptureLane,
    NormalizedCandidate,
    RawCapture,
    RejectionRecord,
    ScenarioRef,
    TransportInfo,
)
from kleos_training_data.staging.store import (
    append_index,
    count_records,
    is_record_file,
    iter_records,
    read_index,
    read_record,
    write_record,
)


def make_ref(**overrides) -> ScenarioRef:
    payload = {
        "family": "test.family",
        "task": "notification_prioritization",
        "point_index": 0,
        "catalog_version": "scenarios-v1",
        "policy_claim": "rank by expected cost of delay",
        "policy": "rank_by_deadline_then_evidence",
        "scenario_fingerprint": "abc123",
    }
    payload.update(overrides)
    return ScenarioRef(**payload)


def make_capture(**overrides) -> RawCapture:
    payload = {
        "capture_id": "cap-0001",
        "batch_id": "b1",
        "lane": CaptureLane.MOCK_BACKEND,
        "adapter": "mock",
        "scenario": make_ref(),
        "endpoint": "mock://x",
        "answer_text": "1. Northwind — nearest deadline.",
        "answer_sha256": "0" * 64,
        "transport": TransportInfo(status=200),
    }
    payload.update(overrides)
    return RawCapture(**payload)


class TestRecordIntegrity:
    def test_a_sealed_record_verifies(self) -> None:
        assert make_capture().sealed().hash_matches()

    def test_an_unsealed_record_does_not_verify(self) -> None:
        assert not make_capture().hash_matches()

    def test_changing_any_field_breaks_the_hash(self) -> None:
        sealed = make_capture().sealed()
        tampered = sealed.model_copy(update={"answer_text": "1. Something else."})
        assert not tampered.hash_matches()

    def test_the_hash_excludes_itself(self) -> None:
        record = make_capture()
        assert record.compute_hash() == record.sealed().compute_hash()

    def test_a_hand_edited_record_is_rejected_on_read(self, tmp_path) -> None:
        path = write_record(make_capture(), tmp_path / "cap.json")

        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["answer_text"] = "1. Edited by hand."
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(StagingIntegrityError, match="modified outside the pipeline"):
            read_record(path, RawCapture)

    def test_verification_can_be_disabled_for_deliberate_inspection(self, tmp_path) -> None:
        path = write_record(make_capture(), tmp_path / "cap.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["answer_text"] = "edited"
        path.write_text(json.dumps(payload), encoding="utf-8")

        assert read_record(path, RawCapture, verify=False).answer_text == "edited"


class TestAtomicWrites:
    def test_a_record_round_trips(self, tmp_path) -> None:
        original = make_capture()
        path = write_record(original, tmp_path / "cap.json")
        assert read_record(path, RawCapture).answer_text == original.answer_text

    def test_no_temporary_files_survive(self, tmp_path) -> None:
        write_record(make_capture(), tmp_path / "cap.json")
        assert [p.name for p in tmp_path.iterdir()] == ["cap.json"]

    def test_a_failed_write_leaves_no_partial_file(self, tmp_path, monkeypatch) -> None:
        import os

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            write_record(make_capture(), tmp_path / "cap.json")

        assert list(tmp_path.iterdir()) == []

    def test_parent_directories_are_created(self, tmp_path) -> None:
        path = write_record(make_capture(), tmp_path / "a" / "b" / "cap.json")
        assert path.is_file()

    def test_a_missing_record_names_the_path(self, tmp_path) -> None:
        with pytest.raises(StagingIntegrityError, match="is missing"):
            read_record(tmp_path / "absent.json", RawCapture)

    def test_unparseable_json_is_reported_as_a_crash(self, tmp_path) -> None:
        path = tmp_path / "cap.json"
        path.write_text('{"truncated": ', encoding="utf-8")
        with pytest.raises(StagingIntegrityError, match="not valid JSON"):
            read_record(path, RawCapture)


class TestRecordDiscovery:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("cap-0001.json", True),
            ("_batch.json", False),
            ("_capture_authorization.json", False),
            ("kx-npr-0000.privacy.json", False),
        ],
    )
    def test_directory_metadata_and_sidecars_are_not_records(
        self, tmp_path, name: str, expected: bool
    ) -> None:
        assert is_record_file(tmp_path / name) is expected

    def test_iteration_skips_non_records(self, tmp_path) -> None:
        write_record(make_capture(), tmp_path / "cap-0001.json")
        (tmp_path / "_batch.json").write_text('{"batch_id": "b1"}', encoding="utf-8")
        (tmp_path / "cap-0001.privacy.json").write_text("{}", encoding="utf-8")

        assert len(list(iter_records(tmp_path, RawCapture))) == 1
        assert count_records(tmp_path) == 1

    def test_iteration_is_sorted(self, tmp_path) -> None:
        for index in (3, 1, 2):
            write_record(make_capture(capture_id=f"cap-{index}"), tmp_path / f"cap-{index}.json")
        ids = [record.capture_id for record in iter_records(tmp_path, RawCapture)]
        assert ids == sorted(ids)

    def test_an_absent_directory_yields_nothing(self, tmp_path) -> None:
        assert list(iter_records(tmp_path / "nope", RawCapture)) == []


class TestCaptureLane:
    def test_production_observation_is_never_promotable(self) -> None:
        assert not CaptureLane.PRODUCTION_OBSERVATION.promotable

    @pytest.mark.parametrize("lane", [CaptureLane.SYNTHETIC, CaptureLane.MOCK_BACKEND])
    def test_offline_lanes_are_promotable(self, lane: CaptureLane) -> None:
        assert lane.promotable

    def test_offline_lanes_claim_a_synthetic_source(self) -> None:
        assert CaptureLane.MOCK_BACKEND.contract_source == "synthetic"
        assert CaptureLane.SYNTHETIC.contract_source == "synthetic"

    def test_a_production_capture_would_claim_real_sanitized(self) -> None:
        assert CaptureLane.PRODUCTION_OBSERVATION.contract_source == "real_sanitized"


class TestNormalization:
    def test_reasoning_spans_are_stripped(self) -> None:
        capture = make_capture(answer_text="<think>weighing</think>1. Northwind.")
        candidate = normalize_capture(
            capture,
            system_prompt="Rank them.",
            user_message="Which first?",
            variation_axes={"domain": "career"},
            group_id="g1",
        )
        assert "<think>" not in candidate.payload["messages"][2]["content"]
        assert "stripped_reasoning_span" in candidate.transformations

    def test_crlf_is_normalized(self) -> None:
        capture = make_capture(answer_text="1. A.\r\n2. B.")
        candidate = normalize_capture(
            capture,
            system_prompt="Rank them.",
            user_message="Which first?",
            variation_axes={"domain": "career"},
            group_id="g1",
        )
        assert "\r" not in candidate.payload["messages"][2]["content"]
        assert "normalized_line_endings" in candidate.transformations

    def test_every_change_is_recorded(self) -> None:
        capture = make_capture(answer_text="<think>x</think>1. A.  \r\n2. B.  ")
        candidate = normalize_capture(
            capture,
            system_prompt="Rank them.",
            user_message="Which first?",
            variation_axes={"domain": "career"},
            group_id="g1",
        )
        assert set(candidate.transformations) >= {
            "stripped_reasoning_span",
            "normalized_line_endings",
        }

    def test_clean_text_records_no_transformation(self) -> None:
        capture = make_capture(answer_text="1. Northwind.")
        candidate = normalize_capture(
            capture,
            system_prompt="Rank them.",
            user_message="Which first?",
            variation_axes={"domain": "career"},
            group_id="g1",
        )
        assert candidate.transformations == []

    def test_an_answer_that_was_only_reasoning_is_rejected(self) -> None:
        capture = make_capture(answer_text="<think>still thinking</think>")
        with pytest.raises(ContractViolationError, match="no assistant content"):
            normalize_capture(
                capture,
                system_prompt="Rank them.",
                user_message="Which first?",
                variation_axes={"domain": "career"},
                group_id="g1",
            )

    def test_the_candidate_id_is_derived_from_content(self) -> None:
        from kleos_training_data.ids import example_id

        capture = make_capture()
        candidate = normalize_capture(
            capture,
            system_prompt="Rank them.",
            user_message="Which first?",
            variation_axes={"domain": "career"},
            group_id="g1",
        )
        assert candidate.candidate_id == example_id(candidate.payload)

    def test_the_candidate_is_marked_provisional(self) -> None:
        candidate = normalize_capture(
            make_capture(),
            system_prompt="Rank them.",
            user_message="Which first?",
            variation_axes={"domain": "career"},
            group_id="g1",
        )
        assert candidate.provisional

    def test_the_lane_survives_normalization(self) -> None:
        candidate = normalize_capture(
            make_capture(lane=CaptureLane.PRODUCTION_OBSERVATION),
            system_prompt="Rank them.",
            user_message="Which first?",
            variation_axes={"domain": "career"},
            group_id="g1",
        )
        assert candidate.lane is CaptureLane.PRODUCTION_OBSERVATION


class TestRejectionReasons:
    def test_the_vocabulary_is_closed(self) -> None:
        with pytest.raises(ValueError, match="not a valid"):
            RejectionReason("it seemed bad")

    def test_privacy_reasons_are_a_subset_of_the_vocabulary(self) -> None:
        assert set(RejectionReason) >= PRIVACY_REASONS

    def test_retryable_reasons_are_a_subset_of_the_vocabulary(self) -> None:
        assert set(RejectionReason) >= RETRYABLE_REASONS

    def test_a_duplicate_is_not_retryable(self) -> None:
        assert RejectionReason.CORPUS_DUPLICATE not in RETRYABLE_REASONS

    def test_a_secret_is_not_retryable(self) -> None:
        assert RejectionReason.SECRET_DETECTED not in RETRYABLE_REASONS

    def test_a_rejection_record_keeps_the_reason_not_the_content(self) -> None:
        record = RejectionRecord(
            candidate_id="kx-npr-0000",
            content_hash="a" * 64,
            stage="promotion",
            reason_codes=[RejectionReason.PRIVATE_FACT],
            detail="unsupported entity in the assistant turn",
        )
        serialized = json.dumps(record.sealed().model_dump(mode="json"))
        assert "PRIVATE_FACT" in serialized


class TestAppendOnlyIndex:
    def test_entries_accumulate(self, tmp_path) -> None:
        path = tmp_path / "index.jsonl"
        append_index(path, {"id": "a"})
        append_index(path, {"id": "b"})
        assert [e["id"] for e in read_index(path)] == ["a", "b"]

    def test_a_missing_index_reads_as_empty(self, tmp_path) -> None:
        assert read_index(tmp_path / "absent.jsonl") == []

    def test_a_malformed_line_does_not_lose_the_rest(self, tmp_path) -> None:
        path = tmp_path / "index.jsonl"
        append_index(path, {"id": "a"})
        with path.open("a", encoding="utf-8") as handle:
            handle.write("{ broken\n")
        append_index(path, {"id": "c"})
        assert [e["id"] for e in read_index(path)] == ["a", "c"]


class TestNormalizedCandidateModel:
    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"[Ee]xtra"):
            NormalizedCandidate(
                candidate_id="kx-npr-0000",
                batch_id="b1",
                lane=CaptureLane.SYNTHETIC,
                scenario=make_ref(),
                payload={},
                scenario_family="f",
                group_id="g",
                content_hash="a" * 64,
                surprise="value",
            )
