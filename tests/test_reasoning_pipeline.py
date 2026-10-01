from __future__ import annotations

from typing import Any

import pytest

from kleos_training_data.collection.runner import to_request
from kleos_training_data.errors import ContractViolationError
from kleos_training_data.privacy.detect import iter_text_fields
from kleos_training_data.privacy.facts import assess
from kleos_training_data.privacy.sanitize import STATUS_CLEAN, sanitize, verify_sanitized
from kleos_training_data.review.packets import PacketItem, ReviewPacket
from kleos_training_data.scenarios.generator import Candidate, generate
from kleos_training_data.scenarios.loader import load_catalog
from kleos_training_data.scenarios.surrogates import SurrogatePool, load_pools
from kleos_training_data.staging.normalize import candidate_from_payload, normalize_capture
from kleos_training_data.staging.records import CaptureLane, RawCapture, ScenarioRef, TransportInfo


def three_messages(reasoning: str) -> dict[str, Any]:
    return {
        "task": "notification_prioritization",
        "messages": [
            {"role": "system", "content": "Rank the items."},
            {"role": "user", "content": "- Redpine: due today.\n- Pennfold: due tomorrow."},
            {"role": "assistant", "content": "Redpine first.", "reasoning": reasoning},
        ],
        "variation_axes": {"format": "bullets"},
    }


def ref() -> ScenarioRef:
    return ScenarioRef(
        family="test.family",
        task="notification_prioritization",
        point_index=0,
        catalog_version="scenarios-v1",
        policy_claim="rank by expected cost of delay",
        policy="rank_by_deadline_then_evidence",
        scenario_fingerprint="abc123",
    )


def capture() -> RawCapture:
    return RawCapture(
        capture_id="cap-0001",
        batch_id="b1",
        lane=CaptureLane.MOCK_BACKEND,
        adapter="mock",
        scenario=ref(),
        endpoint="mock://x",
        answer_text="1. Redpine — nearest deadline.",
        answer_sha256="0" * 64,
        transport=TransportInfo(status=200),
    )


def normalize(reasoning: str | None) -> dict[str, Any]:
    candidate = normalize_capture(
        capture(),
        system_prompt="Rank the items.",
        user_message="- Redpine: due today.",
        variation_axes={"format": "bullets"},
        group_id="test.family:0000",
        reasoning=reasoning,
    )
    payload: dict[str, Any] = candidate.payload
    return payload


@pytest.fixture(scope="module")
def candidate() -> tuple[Candidate, str]:
    for scenario in load_catalog():
        pool = SurrogatePool.load(scenario.entities.pool)
        for generated in generate(scenario, pool):
            if generated.variation_axes.get("format") != "json":
                return generated, scenario.family
    raise AssertionError("the catalog has no non-json candidate")


class TestCarryingReasoning:
    def test_a_request_carries_the_generators_reasoning(self) -> None:
        for scenario in load_catalog():
            pool = SurrogatePool.load(scenario.entities.pool)
            generated = generate(scenario, pool)[0]
            request = to_request(scenario, generated)
            assert request.expected_reasoning == generated.messages[2].get("reasoning")

    def test_normalize_attaches_reasoning_with_clean_line_endings(self) -> None:
        payload = normalize("Rank by deadline first.\r\nSo: Redpine.  \r\n")
        assert payload["messages"][2]["reasoning"] == "Rank by deadline first.\nSo: Redpine."

    def test_normalize_without_reasoning_adds_no_field(self) -> None:
        assert "reasoning" not in normalize(None)["messages"][2]

    def test_think_tags_in_reasoning_are_rejected(self) -> None:
        with pytest.raises(ContractViolationError, match="think"):
            normalize("draft</think>Rank by deadline first.")

    def test_candidate_from_payload_normalizes_reasoning(self) -> None:
        normalized = candidate_from_payload(
            three_messages("Rank first.  \r\nSo: Redpine."),
            batch_id="b1",
            lane=CaptureLane.MOCK_BACKEND,
            scenario=ref(),
            group_id="test.family:0000",
        )
        assert normalized.payload["messages"][2]["reasoning"] == "Rank first.\nSo: Redpine."


class TestScanningReasoning:
    def test_privacy_scanning_sees_reasoning(self) -> None:
        fields = dict(iter_text_fields(three_messages("contact jane.doe@mail.org.")))
        assert fields["messages[2].reasoning"] == "contact jane.doe@mail.org."

    def test_sanitize_redacts_private_data_in_reasoning(self) -> None:
        result = sanitize(
            three_messages("contact jane.doe@mail.org about it."),
            scenario_family="notif.deadline_vs_evidence",
            pools=load_pools(),
        )
        assert "jane.doe@mail.org" not in result.payload["messages"][2]["reasoning"]
        assert result.status != STATUS_CLEAN

    def test_verify_sanitized_catches_residue_in_reasoning(self) -> None:
        problems = verify_sanitized(three_messages("ask [[PERSON_1]] about it."))
        assert "placeholder residue survived rehydration" in problems

    def test_fact_checks_cover_reasoning(self) -> None:
        signals = assess(three_messages("Redpine leads, says Marlowe.")).signals
        found = [s for s in signals if s.rule_id == "fact.unsupported_entity"]
        assert found
        assert found[0].field == "reasoning"
        assert found[0].to_dict()["field"] == "reasoning"

    def test_a_generated_candidate_sanitizes_clean_and_keeps_its_reasoning(
        self, candidate: tuple[Candidate, str]
    ) -> None:
        generated, family = candidate
        payload = generated.to_payload()
        result = sanitize(payload, scenario_family=family, pools=load_pools())
        assert result.status == STATUS_CLEAN
        assert result.payload["messages"][2]["reasoning"] == payload["messages"][2]["reasoning"]

    def test_the_review_packet_shows_reasoning(self) -> None:
        item = PacketItem(
            candidate_id="kx-npr-0001",
            content_hash="a" * 64,
            task="notification_prioritization",
            scenario_family="notif.deadline_vs_evidence",
            policy_claim="Rank by expected cost of delay.",
            anti_claim="Do not rank by list position.",
            payload=three_messages("Rank by deadline first."),
            variation_axes={"format": "bullets"},
        )
        rendered = ReviewPacket("pk-1", "b1", [item]).render_markdown()
        assert "**reasoning:**" in rendered
        assert "Rank by deadline first." in rendered
