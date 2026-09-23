from __future__ import annotations

import pytest

from kleos_training_data.collection.adapters import (
    MockBackendAdapter,
    ScenarioRequest,
    collect_answer,
    parse_sse_stream,
    resolve_adapter,
)
from kleos_training_data.collection.guard import (
    CONFIRM_PHRASE,
    ENV_ALLOW,
    assert_capture_allowed,
    is_local,
)
from kleos_training_data.errors import CaptureError, ProductionGuardError
from kleos_training_data.staging.records import CaptureLane, ScenarioRef

PROD_URL = "https://kleos.example.com"


def make_request(**overrides) -> ScenarioRequest:
    payload = {
        "scenario": ScenarioRef(
            family="test.family",
            task="notification_prioritization",
            point_index=0,
            catalog_version="scenarios-v1",
            policy_claim="rank by expected cost of delay",
            policy="rank_by_deadline_then_evidence",
            scenario_fingerprint="abc123",
        ),
        "system_prompt": "Rank the items.",
        "user_message": "Which first?",
        "variation_axes": {"domain": "career"},
        "group_id": "g1",
        "expected_answer": "1. Northwind — nearest deadline.",
    }
    payload.update(overrides)
    return ScenarioRequest(**payload)


def sse(*lines: str):
    return iter(lines)


class TestSSEParsing:
    def test_a_simple_stream_parses(self) -> None:
        frames = list(sse_frames('data: {"type": "done"}', ""))
        assert [f.type for f in frames] == ["done"]

    def test_comments_are_ignored(self) -> None:
        frames = list(sse_frames(": keepalive", "", 'data: {"type": "done"}', ""))
        assert [f.type for f in frames] == ["done"]

    def test_multi_line_data_is_joined(self) -> None:
        frames = list(sse_frames('data: {"type":', 'data: "done"}', ""))
        assert frames[0].type == "done"

    def test_a_trailing_frame_without_a_blank_line_is_kept(self) -> None:
        assert [f.type for f in sse_frames('data: {"type": "done"}')] == ["done"]

    def test_a_malformed_frame_raises(self) -> None:
        with pytest.raises(CaptureError, match="Malformed SSE frame"):
            list(sse_frames("data: not json", ""))

    def test_a_frame_without_a_type_raises(self) -> None:
        with pytest.raises(CaptureError, match="no 'type'"):
            list(sse_frames('data: {"text": "hi"}', ""))


def sse_frames(*lines: str):
    return parse_sse_stream(iter(lines))


class TestAnswerAssembly:
    def test_deltas_are_concatenated_in_order(self) -> None:
        answer, counts = collect_answer(
            sse_frames(
                'data: {"type": "answer_start"}',
                "",
                'data: {"type": "answer_delta", "text": "Hello "}',
                "",
                'data: {"type": "answer_delta", "text": "world"}',
                "",
                'data: {"type": "done"}',
                "",
            )
        )
        assert answer == "Hello world"
        assert counts == {"answer_start": 1, "answer_delta": 2, "done": 1}

    def test_a_stream_without_done_raises(self) -> None:
        with pytest.raises(CaptureError, match="without a 'done' frame"):
            collect_answer(sse_frames('data: {"type": "answer_delta", "text": "partial"}', ""))

    def test_an_error_frame_raises(self) -> None:
        with pytest.raises(CaptureError, match="error frame"):
            collect_answer(sse_frames('data: {"type": "error", "code": "rate_limit"}', ""))

    def test_frame_counts_record_activity_without_recording_content(self) -> None:
        _, counts = collect_answer(
            sse_frames(
                'data: {"type": "citation", "url": "https://private.example/doc"}',
                "",
                'data: {"type": "done"}',
                "",
            )
        )
        assert counts["citation"] == 1
        assert "https://private.example/doc" not in str(counts)


class TestMockAdapter:
    def test_it_returns_the_policy_derived_answer(self) -> None:
        capture = MockBackendAdapter().run(make_request(), batch_id="b1")
        assert "Northwind" in capture.answer_text

    def test_it_emits_artifacts_normalization_must_handle(self) -> None:
        capture = MockBackendAdapter().run(
            make_request(expected_answer="1. A.\n2. B."), batch_id="b1"
        )
        assert "\r\n" in capture.answer_text
        assert "<think>" in capture.answer_text

    def test_reasoning_emission_can_be_disabled(self) -> None:
        capture = MockBackendAdapter(emit_reasoning=False).run(make_request(), batch_id="b1")
        assert "<think>" not in capture.answer_text

    def test_it_is_deterministic(self) -> None:
        first = MockBackendAdapter().run(make_request(), batch_id="b1")
        second = MockBackendAdapter().run(make_request(), batch_id="b1")
        assert first.capture_id == second.capture_id
        assert first.answer_sha256 == second.answer_sha256

    def test_the_capture_id_depends_on_the_batch(self) -> None:
        a = MockBackendAdapter().run(make_request(), batch_id="b1")
        b = MockBackendAdapter().run(make_request(), batch_id="b2")
        assert a.capture_id != b.capture_id

    def test_chunking_does_not_change_the_answer(self) -> None:
        answer = "1. Northwind — nearest deadline and confirmed evidence."
        big = MockBackendAdapter(chunk_size=1000).run(
            make_request(expected_answer=answer), batch_id="b1"
        )
        small = MockBackendAdapter(chunk_size=3).run(
            make_request(expected_answer=answer), batch_id="b1"
        )
        assert big.answer_text == small.answer_text

    def test_it_marks_the_mock_lane(self) -> None:
        assert MockBackendAdapter().run(make_request(), batch_id="b1").lane is (
            CaptureLane.MOCK_BACKEND
        )

    def test_integrations_are_recorded_as_disabled(self) -> None:
        assert MockBackendAdapter().run(make_request(), batch_id="b1").integrations_disabled

    def test_it_refuses_without_a_policy_derived_answer(self) -> None:
        with pytest.raises(CaptureError, match="policy-derived answer"):
            MockBackendAdapter().run(make_request(expected_answer=None), batch_id="b1")

    def test_an_unknown_adapter_names_the_available_ones(self) -> None:
        with pytest.raises(CaptureError, match="Unknown adapter"):
            resolve_adapter("kleos_chat_v2")


class TestLocalDetection:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:8000",
            "http://localhost:8000/api/v1",
            "mock://local",
            "http://kleos.local:3000",
            "http://api.test",
        ],
    )
    def test_local_urls_need_no_authorization(self, url: str) -> None:
        assert is_local(url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://kleos.example.com",
            "https://api.kleos.app/api/v1",
            "http://10.0.0.5:8000",
            "https://localhost.evil.example.com",
        ],
    )
    def test_everything_else_is_production(self, url: str) -> None:
        assert not is_local(url)


class TestProductionGuard:
    def _env(self, **overrides) -> dict[str, str]:
        env = {ENV_ALLOW: "1"}
        env.update(overrides)
        return env

    def test_a_local_url_needs_no_authorization(self) -> None:
        assert (
            assert_capture_allowed(
                "http://127.0.0.1:8000", allow_production=False, confirm=None, env={}
            )
            is None
        )

    def test_all_four_conditions_together_authorize(self) -> None:
        authorization = assert_capture_allowed(
            PROD_URL,
            allow_production=True,
            confirm=CONFIRM_PHRASE,
            env=self._env(),
            scenario_families=("f1",),
            expected_captures=10,
        )
        assert authorization is not None
        assert authorization.base_url_host == "kleos.example.com"

    def test_missing_the_flag_refuses(self) -> None:
        with pytest.raises(ProductionGuardError, match="allow-production"):
            assert_capture_allowed(
                PROD_URL, allow_production=False, confirm=CONFIRM_PHRASE, env=self._env()
            )

    def test_missing_the_environment_variable_refuses(self) -> None:
        with pytest.raises(ProductionGuardError, match=ENV_ALLOW):
            assert_capture_allowed(PROD_URL, allow_production=True, confirm=CONFIRM_PHRASE, env={})

    def test_a_wrong_confirmation_phrase_refuses(self) -> None:
        with pytest.raises(ProductionGuardError, match="confirmation phrase"):
            assert_capture_allowed(PROD_URL, allow_production=True, confirm="yes", env=self._env())

    def test_ci_refuses_even_when_everything_else_is_satisfied(self) -> None:
        with pytest.raises(ProductionGuardError, match="CI is set"):
            assert_capture_allowed(
                PROD_URL,
                allow_production=True,
                confirm=CONFIRM_PHRASE,
                env=self._env(CI="true"),
            )

    def test_the_error_never_explains_how_to_disable_the_guard(self) -> None:
        with pytest.raises(ProductionGuardError) as caught:
            assert_capture_allowed(PROD_URL, allow_production=False, confirm=None, env={})
        rendered = caught.value.render().lower()
        assert "mock" in rendered, "the error should point at the safe alternative"
        assert "skip" not in rendered
        assert "bypass" not in rendered
        assert "disable" not in rendered

    def test_the_audit_record_stores_a_host_not_a_url(self) -> None:
        authorization = assert_capture_allowed(
            f"{PROD_URL}/api?access_token=supersecretvalue",
            allow_production=True,
            confirm=CONFIRM_PHRASE,
            env=self._env(),
        )
        assert authorization is not None
        serialized = str(authorization.to_dict())
        assert "supersecretvalue" not in serialized

    def test_the_audit_record_states_the_promotion_ban(self) -> None:
        authorization = assert_capture_allowed(
            PROD_URL, allow_production=True, confirm=CONFIRM_PHRASE, env=self._env()
        )
        assert authorization is not None
        assert "never be promoted" in authorization.to_dict()["note"]
