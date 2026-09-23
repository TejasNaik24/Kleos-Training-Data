from __future__ import annotations

import pytest

from kleos_training_data.scenarios.generator import generate
from kleos_training_data.scenarios.loader import load_catalog
from kleos_training_data.scenarios.policies import (
    CLOSE_CALL_RELATIVE_MARGIN,
    STALE_AFTER_DAYS,
    STRONG_CONFLICT_MIN_EVIDENCE,
    ask_when_request_ambiguous,
    defer_to_explicit_statement,
    rank_or_abstain_when_close,
    relative_separation,
    verify_when_evidence_weak,
)
from kleos_training_data.scenarios.situations import Item, Situation
from kleos_training_data.scenarios.surrogates import load_pools


def sit(*items: Item, **kw) -> Situation:
    base = {"task": "notification_prioritization", "family": "t", "point_index": 0, "axes": {}}
    return Situation(**{**base, **kw}, items=items)


@pytest.fixture(scope="module")
def generated():
    pools = load_pools()
    return [(s, c) for s in load_catalog() for c in generate(s, pools[s.entities.pool])]


class TestCloseCallIsScaleInvariant:
    @pytest.mark.parametrize("scale", [1, 3, 7, 14, 30, 60])
    def test_the_same_relative_gap_gives_the_same_answer_at_every_scale(self, scale: int) -> None:
        a = Item("item_a", "A", scale, "confirmed", "high", "")
        b = Item("item_b", "B", scale * 2 + 1, "confirmed", "high", "")
        decisions = {rank_or_abstain_when_close(sit(a, b)).abstained}
        for other in (1, 3, 7, 14, 30, 60):
            x = Item("item_a", "A", other, "confirmed", "high", "")
            y = Item("item_b", "B", other * 2 + 1, "confirmed", "high", "")
            decisions.add(rank_or_abstain_when_close(sit(x, y)).abstained)
        assert len(decisions) == 1, (
            "identical relative separation produced different abstain decisions at "
            "different deadline scales; the margin has gone back to being absolute"
        )

    def test_relative_separation_is_invariant_under_uniform_scaling(self) -> None:
        near = relative_separation(
            Item("a", "A", 1, "confirmed", "high", ""),
            Item("b", "B", 2, "confirmed", "high", ""),
        )
        far = relative_separation(
            Item("a", "A", 19, "confirmed", "high", ""),
            Item("b", "B", 29, "confirmed", "high", ""),
        )
        assert near == pytest.approx(far, abs=0.02)

    def test_a_genuine_tie_still_abstains(self) -> None:
        a = Item("item_a", "A", 5, "reported", "medium", "")
        b = Item("item_b", "B", 5, "reported", "medium", "")
        assert rank_or_abstain_when_close(sit(a, b)).abstained

    def test_a_clear_separation_never_abstains(self) -> None:
        a = Item("item_a", "A", 1, "confirmed", "high", "")
        b = Item("item_b", "B", 40, "unverified", "low", "")
        assert not rank_or_abstain_when_close(sit(a, b)).abstained

    def test_the_margin_is_relative_not_absolute(self) -> None:
        from kleos_training_data.scenarios import policies

        assert policies.CLOSE_CALL_MARGIN is None, (
            "the absolute margin is back; it cannot be compared against a score "
            "whose scale varies with deadline distance"
        )
        assert 0.0 < CLOSE_CALL_RELATIVE_MARGIN < 1.0


class TestTheTwoUnderdeterminationConcepts:
    def test_referential_ambiguity_asks_regardless_of_evidence_strength(self) -> None:
        strong = (
            Item("item_a", "A", 1, "confirmed", "high", ""),
            Item("item_b", "B", 9, "confirmed", "high", ""),
        )
        d = ask_when_request_ambiguous(sit(*strong, request_ambiguous=True))
        assert d.abstained and d.deciding_factor == "request_ambiguous", (
            "the request is underspecified, so no amount of evidence strength makes ranking correct"
        )

    def test_a_determinate_request_is_answered(self) -> None:
        d = ask_when_request_ambiguous(
            sit(
                Item("item_a", "A", 1, "confirmed", "high", ""),
                Item("item_b", "B", 9, "unverified", "low", ""),
                request_ambiguous=False,
            )
        )
        assert not d.abstained

    def test_epistemic_weakness_is_about_evidence_not_the_request(self) -> None:
        weak = sit(
            Item("item_a", "A", 1, "unverified", "high", ""),
            Item("item_b", "B", 2, "single_source", "high", ""),
        )
        assert verify_when_evidence_weak(weak).deciding_factor == "missing_input"
        strong = sit(
            Item("item_a", "A", 1, "confirmed", "high", ""),
            Item("item_b", "B", 9, "reported", "low", ""),
        )
        assert verify_when_evidence_weak(strong).deciding_factor != "missing_input"

    def test_the_two_concepts_are_separate_policies(self) -> None:
        from kleos_training_data.scenarios.policies import POLICIES

        assert "ask_when_request_ambiguous" in POLICIES
        assert "verify_when_evidence_weak" in POLICIES
        assert "ask_when_underdetermined" not in POLICIES, "the overloaded policy is back"

    def test_no_example_answers_confidently_against_an_ambiguous_prompt(self, generated) -> None:
        bad = [
            (s.family, c.group_id)
            for s, c in generated
            if c.situation.request_ambiguous and not c.decision.abstained
        ]
        assert not bad, f"{len(bad)} example(s) rank despite a declared-ambiguous request"


class TestDecisionBThresholds:
    def test_the_thresholds_are_defined_and_deterministic(self) -> None:
        assert isinstance(STALE_AFTER_DAYS, int) and STALE_AFTER_DAYS > 0
        assert 0.0 < STRONG_CONFLICT_MIN_EVIDENCE <= 1.0

    def test_a_stale_explicit_statement_against_strong_evidence_is_surfaced(self) -> None:
        d = defer_to_explicit_statement(
            sit(
                Item("item_a", "Stated", STALE_AFTER_DAYS + 15, "confirmed", "high", ""),
                Item("item_b", "Newer", 3, "corroborated", "high", ""),
                framing="memory",
            )
        )
        assert d.deciding_factor == "stale_explicit_conflict" and d.abstained

    def test_a_fresh_explicit_statement_is_not_surfaced(self) -> None:
        d = defer_to_explicit_statement(
            sit(
                Item("item_a", "Stated", 2, "confirmed", "high", ""),
                Item("item_b", "Newer", 1, "corroborated", "high", ""),
                framing="memory",
            )
        )
        assert d.deciding_factor == "explicit_statement" and not d.abstained

    def test_weakly_conflicting_evidence_does_not_trigger_surfacing(self) -> None:
        d = defer_to_explicit_statement(
            sit(
                Item("item_a", "Stated", STALE_AFTER_DAYS + 15, "confirmed", "high", ""),
                Item("item_b", "Newer", 3, "single_source", "high", ""),
                framing="memory",
            )
        )
        assert d.deciding_factor == "explicit_statement" and not d.abstained

    def test_the_corpus_actually_exercises_the_case(self, generated) -> None:
        n = sum(1 for _s, c in generated if c.decision.deciding_factor == "stale_explicit_conflict")
        assert n > 0, "Decision B is described but never exercised, as in v0.0.5"


class TestExplanationsAreVerifiedNotAsserted:
    def test_no_answer_claims_comparability_it_has_not_checked(self, generated) -> None:
        bad = []
        for _s, c in generated:
            answer = c.messages[-1]["content"]
            d = c.decision
            if d.abstained or len(d.ranking) < 2:
                continue
            top = c.situation.item(d.ranking[0])
            runner = c.situation.item(d.ranking[1])
            if "of comparable strength" in answer and top.evidence != runner.evidence:
                bad.append((c.group_id, "evidence"))
            if "are comparable" in answer and top.evidence != runner.evidence:
                bad.append((c.group_id, "evidence"))
            if (
                "Deadlines are close" in answer
                and abs(top.deadline_days - runner.deadline_days) > 3
            ):
                bad.append((c.group_id, "deadline"))
        assert not bad, f"{len(bad)} answer(s) assert a comparability that does not hold"

    def test_no_resolver_asks_to_confirm_something_already_confirmed(self, generated) -> None:
        import re

        bad = []
        for _s, c in generated:
            m = re.search(
                r"confirm the status of ([^—]+)— it is ([^,]+)", c.messages[-1]["content"]
            )
            if m and "confirmed directly by the owner" in m.group(2):
                bad.append(c.group_id)
        assert not bad, (
            f"{len(bad)} answer(s) ask the user to confirm a record the prompt "
            f"already reports as confirmed"
        )
