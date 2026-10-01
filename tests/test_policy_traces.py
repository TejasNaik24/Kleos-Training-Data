from __future__ import annotations

from dataclasses import replace

import pytest

from kleos_training_data.scenarios.generator import Candidate, generate
from kleos_training_data.scenarios.loader import load_catalog
from kleos_training_data.scenarios.policies import (
    BASES,
    CHECK_NAMES,
    CLOSE_CALL_RELATIVE_MARGIN,
    RULES,
    relative_separation,
)
from kleos_training_data.scenarios.surrogates import SurrogatePool


@pytest.fixture(scope="module")
def candidates() -> list[Candidate]:
    out: list[Candidate] = []
    for scenario in load_catalog():
        pool = SurrogatePool.load(scenario.entities.pool)
        out.extend(generate(scenario, pool))
    return out


class TestPolicyTraces:
    def test_every_decision_carries_a_well_formed_trace(self, candidates: list[Candidate]) -> None:
        for candidate in candidates:
            trace = candidate.decision.trace
            assert trace is not None
            assert trace.rule in RULES
            assert trace.basis in BASES
            assert all(check.name in CHECK_NAMES for check in trace.checks)

    def test_the_trace_does_not_take_part_in_equality(self, candidates: list[Candidate]) -> None:
        decision = candidates[0].decision
        assert replace(decision, trace=None) == decision

    def test_largest_gap_names_the_decisions_factor(self, candidates: list[Candidate]) -> None:
        for candidate in candidates:
            trace, decision = candidate.decision.trace, candidate.decision
            assert trace is not None
            if trace.basis != "largest_gap":
                continue
            top, runner_up = (candidate.situation.item(k) for k in decision.ranking[:2])
            expected = (
                ("deadline", abs(top.deadline_score - runner_up.deadline_score)),
                ("evidence", abs(top.evidence_weight - runner_up.evidence_weight)),
                ("impact", abs(top.impact_weight - runner_up.impact_weight)),
            )
            gaps = dict(expected)
            assert trace.gaps == expected
            assert decision.deciding_factor == max(gaps, key=lambda factor: gaps[factor])

    def test_declines_and_only_declines_have_basis_decline(
        self, candidates: list[Candidate]
    ) -> None:
        for candidate in candidates:
            trace = candidate.decision.trace
            assert trace is not None
            assert (trace.basis == "decline") == candidate.decision.abstained

    def test_a_close_call_is_measured_like_the_policy(self, candidates: list[Candidate]) -> None:
        seen = 0
        for candidate in candidates:
            trace = candidate.decision.trace
            assert trace is not None
            for check in trace.checks:
                if check.name != "close_call":
                    continue
                seen += 1
                top, runner_up = (
                    candidate.situation.item(k) for k in candidate.decision.ranking[:2]
                )
                assert check.measured == relative_separation(top, runner_up)
                assert check.threshold == CLOSE_CALL_RELATIVE_MARGIN
                assert check.triggered == (check.measured < CLOSE_CALL_RELATIVE_MARGIN)
        assert seen > 0

    def test_every_check_names_only_items_in_the_situation(
        self, candidates: list[Candidate]
    ) -> None:
        for candidate in candidates:
            trace = candidate.decision.trace
            assert trace is not None
            keys = {item.key for item in candidate.situation.items}
            for check in trace.checks:
                assert set(check.subjects) <= keys
