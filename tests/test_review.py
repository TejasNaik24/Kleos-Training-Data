from __future__ import annotations

import itertools
import json

import pytest
from pydantic import ValidationError

from kleos_training_data.errors import ReviewError
from kleos_training_data.review.llm_schema import (
    REVIEW_RESPONSE_SCHEMA,
    LLMReview,
    parse_review,
    validate_against_schema,
)
from kleos_training_data.review.packets import (
    PacketItem,
    ReviewPacket,
    nearest_neighbours,
)
from kleos_training_data.review.records import HumanDecision, combine
from kleos_training_data.review.reviewers import MockReviewer, resolve_reviewer
from kleos_training_data.review.rubric import (
    DIMENSIONS,
    HARD_GATES,
    MIN_DIMENSION_SCORE,
    UNOVERRIDABLE_GATES,
    GateResults,
    ReviewVerdict,
    RubricScores,
)
from kleos_training_data.staging.reasons import RejectionReason


def scores(value: int = 4, **overrides: int) -> RubricScores:
    return RubricScores(**{**dict.fromkeys(DIMENSIONS, value), **overrides})


def gates(**overrides: str) -> GateResults:
    return GateResults(**{**dict.fromkeys(HARD_GATES, "PASS"), **overrides})


def payload(assistant: str = "Start with Northwind — nearest deadline, confirmed.") -> dict:
    return {
        "task": "notification_prioritization",
        "messages": [
            {"role": "system", "content": "Rank the items."},
            {"role": "user", "content": "Northwind is due Friday. Fieldstone is due next month."},
            {"role": "assistant", "content": assistant},
        ],
        "variation_axes": {"domain": "career"},
    }


class TestGatesDominate:
    @pytest.mark.parametrize(
        "failing",
        [
            subset
            for size in range(1, len(HARD_GATES) + 1)
            for subset in itertools.combinations(sorted(HARD_GATES), size)
        ],
    )
    @pytest.mark.parametrize("decision", ["approved", "needs_revision"])
    def test_no_failing_subset_can_be_approved(
        self, failing: tuple[str, ...], decision: str
    ) -> None:
        with pytest.raises(ValidationError, match="FAILED"):
            ReviewVerdict(
                scores=scores(4),
                gates=gates(**dict.fromkeys(failing, "FAIL")),
                decision=decision,
            )

    def test_a_failing_gate_with_rejected_is_constructible(self) -> None:
        verdict = ReviewVerdict(
            scores=scores(4), gates=gates(no_private_data="FAIL"), decision="rejected"
        )
        assert verdict.decision == "rejected"

    def test_decide_rejects_on_a_failing_gate_despite_perfect_scores(self) -> None:
        verdict = ReviewVerdict.decide(scores(4), gates(policy_not_facts="FAIL"))
        assert verdict.decision == "rejected"
        assert "policy_not_facts" in verdict.explain()

    def test_json_deserialization_cannot_bypass_the_invariant(self) -> None:
        crafted = {
            "scores": scores(4).model_dump(),
            "gates": gates(no_private_data="FAIL").model_dump(),
            "decision": "approved",
        }
        with pytest.raises(ValidationError):
            ReviewVerdict.model_validate(crafted)

    def test_model_copy_cannot_bypass_the_invariant(self) -> None:
        verdict = ReviewVerdict(
            scores=scores(4), gates=gates(no_private_data="FAIL"), decision="rejected"
        )
        with pytest.raises(ValidationError):
            ReviewVerdict.model_validate({**verdict.model_dump(), "decision": "approved"})

    def test_a_verdict_is_frozen(self) -> None:
        verdict = ReviewVerdict.decide(scores(4), gates())
        with pytest.raises(ValidationError):
            verdict.decision = "rejected"


class TestScoreThresholds:
    def test_all_gates_passing_and_high_scores_approves(self) -> None:
        assert ReviewVerdict.decide(scores(4), gates()).decision == "approved"

    def test_a_single_dimension_below_the_floor_blocks_approval(self) -> None:
        verdict = ReviewVerdict.decide(scores(4, decision_correctness=1), gates())
        assert verdict.decision == "needs_revision"
        assert "decision_correctness" in verdict.explain()

    def test_a_low_mean_blocks_approval(self) -> None:
        verdict = ReviewVerdict.decide(scores(2), gates(), min_mean_score=3.0)
        assert verdict.decision == "needs_revision"

    def test_the_floor_is_inclusive(self) -> None:
        assert (
            ReviewVerdict.decide(scores(MIN_DIMENSION_SCORE), gates(), min_mean_score=0.0).decision
            == "approved"
        )

    def test_scores_outside_the_range_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            scores(4, actionability=5)


class TestLLMSchema:
    def _valid(self, **overrides) -> dict:
        base = {
            "scores": dict.fromkeys(DIMENSIONS, 4),
            "gates": dict.fromkeys(HARD_GATES, "PASS"),
            "gate_evidence": {},
            "rationale": "The example teaches a transferable prioritization policy.",
            "suggested_revision": None,
            "reviewer_confidence": 0.9,
        }
        base.update(overrides)
        return base

    def test_a_valid_review_parses(self) -> None:
        assert parse_review(self._valid()).reviewer_confidence == 0.9

    @pytest.mark.parametrize(
        ("name", "payload_override"),
        [
            (
                "score_out_of_range",
                {"scores": {**dict.fromkeys(DIMENSIONS, 4), "actionability": 7}},
            ),
            ("missing_dimension", {"scores": dict.fromkeys(list(DIMENSIONS)[:-1], 4)}),
            (
                "unknown_gate_value",
                {"gates": {**dict.fromkeys(HARD_GATES, "PASS"), "no_private_data": "MAYBE"}},
            ),
            ("missing_gate", {"gates": dict.fromkeys(list(HARD_GATES)[:-1], "PASS")}),
            ("short_rationale", {"rationale": "fine"}),
            ("confidence_out_of_range", {"reviewer_confidence": 1.5}),
            ("extra_field", {"vibes": "good"}),
        ],
    )
    def test_both_layers_reject_the_same_payloads(self, name: str, payload_override: dict) -> None:
        crafted = self._valid(**payload_override)

        schema_errors = validate_against_schema(crafted)
        pydantic_ok = True
        try:
            LLMReview.model_validate(crafted)
        except ValidationError:
            pydantic_ok = False

        assert schema_errors, f"{name}: JSON Schema accepted it"
        assert not pydantic_ok, f"{name}: pydantic accepted it"

    def test_both_layers_accept_the_valid_payload(self) -> None:
        assert validate_against_schema(self._valid()) == []
        assert LLMReview.model_validate(self._valid())

    def test_a_failing_gate_without_evidence_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="no evidence"):
            LLMReview.model_validate(
                self._valid(gates={**dict.fromkeys(HARD_GATES, "PASS"), "no_private_data": "FAIL"})
            )

    def test_a_failing_gate_with_evidence_is_accepted(self) -> None:
        review = LLMReview.model_validate(
            self._valid(
                gates={**dict.fromkeys(HARD_GATES, "PASS"), "no_private_data": "FAIL"},
                gate_evidence={"no_private_data": "an email address survives in message 1"},
            )
        )
        assert review.gates.failed_ids() == ["no_private_data"]

    def test_evidence_for_an_unknown_gate_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown gate"):
            LLMReview.model_validate(self._valid(gate_evidence={"vibes": "off"}))

    def test_the_schema_covers_every_dimension_and_gate(self) -> None:
        assert set(REVIEW_RESPONSE_SCHEMA["properties"]["scores"]["properties"]) == set(DIMENSIONS)
        assert set(REVIEW_RESPONSE_SCHEMA["properties"]["gates"]["properties"]) == set(HARD_GATES)

    def test_the_schema_forbids_extra_fields(self) -> None:
        assert REVIEW_RESPONSE_SCHEMA["additionalProperties"] is False


class TestMockReviewer:
    def _item(self, **overrides) -> dict:
        base = {"candidate_id": "kx-npr-0001", "payload": payload(), "contract_valid": True}
        base.update(overrides)
        return base

    def test_a_clean_example_passes_every_gate(self) -> None:
        review = MockReviewer().review(self._item())
        assert not review.gates.any_failed()

    def test_it_is_deterministic(self) -> None:
        assert MockReviewer().review(self._item()) == MockReviewer().review(self._item())

    def test_private_data_fails_the_gate(self) -> None:
        item = self._item(payload=payload("Email j.doe@somecollege.edu about it."))
        review = MockReviewer().review(item)
        assert review.gates.no_private_data == "FAIL"
        assert review.gate_evidence["no_private_data"]

    def test_a_fact_teaching_answer_fails_the_policy_gate(self) -> None:
        item = self._item(
            payload=payload("Do it because you work at Initech and your advisor agreed.")
        )
        review = MockReviewer().review(item)
        assert review.gates.policy_not_facts == "FAIL"

    def test_an_invalid_contract_fails_its_gate(self) -> None:
        review = MockReviewer().review(self._item(contract_valid=False))
        assert review.gates.schema_and_contract_valid == "FAIL"

    def test_every_failure_carries_evidence(self) -> None:
        item = self._item(payload=payload("Email j.doe@somecollege.edu, you work at Initech."))
        review = MockReviewer().review(item)
        for gate in review.gates.failed_ids():
            assert review.gate_evidence.get(gate)

    def test_an_unknown_reviewer_names_the_available_ones(self) -> None:
        with pytest.raises(ReviewError, match="Unknown reviewer"):
            resolve_reviewer("gpt-9")


class TestHumanDecision:
    def _decision(self, **overrides) -> HumanDecision:
        base = {
            "candidate_id": "kx-npr-0001",
            "content_hash": "a" * 64,
            "decision": "approve",
            "gates": gates(),
        }
        base.update(overrides)
        return HumanDecision(**base)

    @pytest.mark.parametrize("gate", sorted(UNOVERRIDABLE_GATES))
    def test_a_human_cannot_approve_over_privacy(self, gate: str) -> None:
        with pytest.raises(ValidationError, match="cannot approve over failing gate"):
            self._decision(gates=gates(**{gate: "FAIL"}))

    def test_a_human_cannot_approve_over_any_failing_gate(self) -> None:
        with pytest.raises(ValidationError, match="cannot approve"):
            self._decision(gates=gates(no_unsupported_claims="FAIL"))

    def test_a_human_may_reject_over_a_failing_gate(self) -> None:
        decision = self._decision(
            decision="reject",
            gates=gates(no_private_data="FAIL"),
            reason_codes=[RejectionReason.PII_UNRESOLVED],
        )
        assert decision.decision == "reject"

    def test_a_rejection_needs_a_reason_code(self) -> None:
        with pytest.raises(ValidationError, match="reason code"):
            self._decision(decision="reject", gates=gates(no_private_data="FAIL"))

    def test_a_score_override_needs_a_justification(self) -> None:
        with pytest.raises(ValidationError, match="override_justification"):
            self._decision(score_overrides={"actionability": 2})

    def test_a_justified_override_is_accepted(self) -> None:
        decision = self._decision(
            score_overrides={"actionability": 2},
            override_justification="The answer needs a follow-up to be usable.",
        )
        assert decision.score_overrides == {"actionability": 2}

    def test_an_override_of_an_unknown_dimension_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown dimension"):
            self._decision(score_overrides={"vibes": 3}, override_justification="x")

    def test_an_override_outside_the_range_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="outside 0-4"):
            self._decision(score_overrides={"actionability": 9}, override_justification="x")

    def test_the_reviewer_is_a_role_not_a_name(self) -> None:
        assert self._decision().reviewer_role == "operator"


class TestDecisionSignatures:
    def _signed(self, **overrides) -> HumanDecision:
        base = {
            "candidate_id": "kx-npr-0001",
            "content_hash": "a" * 64,
            "decision": "approve",
            "gates": gates(),
        }
        base.update(overrides)
        return HumanDecision(**base).signed()

    def test_a_signed_decision_verifies(self) -> None:
        assert self._signed().signature_valid()

    def test_an_unsigned_decision_does_not_verify(self) -> None:
        assert not HumanDecision(
            candidate_id="x", content_hash="a" * 64, decision="approve", gates=gates()
        ).signature_valid()

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("decision", "reject"),
            ("content_hash", "b" * 64),
            ("reviewer_role", "someone_else"),
            ("notes", "changed my mind"),
        ],
    )
    def test_mutating_any_field_breaks_the_signature(self, field: str, value: str) -> None:
        signed = self._signed()
        tampered = signed.model_copy(update={field: value})
        assert not tampered.signature_valid()

    def test_a_decision_applies_only_to_the_hash_it_was_made_about(self) -> None:
        signed = self._signed()
        assert signed.applies_to("a" * 64)
        assert not signed.applies_to("b" * 64)

    def test_an_edited_candidate_loses_its_approval(self) -> None:
        signed = self._signed(content_hash="original" + "0" * 56)
        assert not signed.applies_to("edited" + "0" * 58)


class TestCombine:
    def _machine(self, **overrides):
        from kleos_training_data.review.records import MachineReviewRecord

        review = parse_review(
            {
                "scores": dict.fromkeys(DIMENSIONS, 4),
                "gates": dict.fromkeys(HARD_GATES, "PASS"),
                "gate_evidence": {},
                "rationale": "A transferable prioritization policy, well grounded.",
                "reviewer_confidence": 0.9,
            }
        )
        return MachineReviewRecord(
            candidate_id="kx-npr-0001",
            content_hash="a" * 64,
            packet_id="pk-1",
            reviewer_backend="mock",
            review=review,
            **overrides,
        )

    def test_the_human_gates_win(self) -> None:
        human = HumanDecision(
            candidate_id="kx-npr-0001",
            content_hash="a" * 64,
            decision="reject",
            gates=gates(policy_not_facts="FAIL"),
            reason_codes=[RejectionReason.PRIVATE_FACT],
        )
        verdict = combine(self._machine(), human, min_mean_score=3.0)
        assert verdict.decision == "rejected"

    def test_machine_scores_are_used_when_the_human_gives_none(self) -> None:
        human = HumanDecision(
            candidate_id="kx-npr-0001", content_hash="a" * 64, decision="approve", gates=gates()
        )
        verdict = combine(self._machine(), human, min_mean_score=3.0)
        assert verdict.decision == "approved"
        assert verdict.scores.mean == 4.0

    def test_a_human_override_wins_over_the_machine_score(self) -> None:
        human = HumanDecision(
            candidate_id="kx-npr-0001",
            content_hash="a" * 64,
            decision="approve",
            gates=gates(),
            score_overrides={"generalizability": 1},
            override_justification="Reasoning only works for these specific entities.",
        )
        verdict = combine(self._machine(), human, min_mean_score=3.0)
        assert verdict.scores.generalizability == 1
        assert verdict.decision == "needs_revision"

    def test_revise_produces_needs_revision(self) -> None:
        human = HumanDecision(
            candidate_id="kx-npr-0001", content_hash="a" * 64, decision="revise", gates=gates()
        )
        assert combine(self._machine(), human, min_mean_score=3.0).decision == "needs_revision"


class TestPackets:
    def _item(self, **overrides) -> PacketItem:
        base = {
            "candidate_id": "kx-npr-0001",
            "content_hash": "a" * 64,
            "task": "notification_prioritization",
            "scenario_family": "notif.deadline_vs_evidence",
            "policy_claim": "Rank by expected cost of delay.",
            "anti_claim": "Do not rank by list position.",
            "payload": payload(),
            "variation_axes": {"domain": "career", "urgency": "high"},
        }
        base.update(overrides)
        return PacketItem(**base)

    def test_the_packet_leads_with_the_claims(self) -> None:
        rendered = ReviewPacket("pk-1", "b1", [self._item()]).render_markdown()
        assert "Should teach" in rendered
        assert "Must not teach" in rendered
        assert "Rank by expected cost of delay." in rendered

    def test_the_packet_shows_the_conversation(self) -> None:
        rendered = ReviewPacket("pk-1", "b1", [self._item()]).render_markdown()
        assert "Northwind is due Friday" in rendered

    def test_the_packet_lists_every_dimension_and_gate(self) -> None:
        rendered = ReviewPacket("pk-1", "b1", [self._item()]).render_markdown()
        for name in list(DIMENSIONS) + list(HARD_GATES):
            assert name in rendered

    def test_the_packet_states_what_a_human_may_not_override(self) -> None:
        rendered = ReviewPacket("pk-1", "b1", [self._item()]).render_markdown()
        assert "may **not** approve" in rendered

    def test_writing_produces_all_three_files(self, tmp_path) -> None:
        directory = ReviewPacket("pk-1", "b1", [self._item()]).write(tmp_path / "pk-1")
        assert (directory / "packet.md").is_file()
        assert (directory / "packet.jsonl").is_file()
        assert (directory / "packet.meta.json").is_file()

    def test_the_machine_form_carries_the_response_schema(self, tmp_path) -> None:
        directory = ReviewPacket("pk-1", "b1", [self._item()]).write(tmp_path / "pk-1")
        meta = json.loads((directory / "packet.meta.json").read_text(encoding="utf-8"))
        assert meta["response_schema"]["properties"]["scores"]

    def test_the_machine_form_is_one_object_per_line(self, tmp_path) -> None:
        directory = ReviewPacket("pk-1", "b1", [self._item(), self._item()]).write(
            tmp_path / "pk-1"
        )
        lines = (directory / "packet.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert all(json.loads(line)["candidate_id"] for line in lines)


class TestNeighbours:
    def test_an_identical_example_scores_one(self) -> None:
        text = "user: rank these\nassistant: start with the nearest deadline"
        assert nearest_neighbours(text, {"other": text})[0][1] == pytest.approx(1.0)

    def test_an_unrelated_example_is_not_returned(self) -> None:
        assert (
            nearest_neighbours(
                "user: rank these items by deadline",
                {"other": "completely different subject matter entirely here"},
            )
            == []
        )

    def test_an_empty_corpus_is_handled(self) -> None:
        assert nearest_neighbours("anything", {}) == []

    def test_results_are_ordered_by_similarity(self) -> None:
        text = "user: rank these items\nassistant: start with the nearest deadline"
        corpus = {
            "close": text,
            "middling": "user: rank these items\nassistant: pick something else entirely",
        }
        results = nearest_neighbours(text, corpus)
        assert results[0][0] == "close"
        assert results[0][1] >= results[-1][1]
