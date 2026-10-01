from __future__ import annotations

import re

import pytest

from kleos_training_data.errors import ScenarioError
from kleos_training_data.scenarios.generator import Candidate, generate
from kleos_training_data.scenarios.loader import load_catalog
from kleos_training_data.scenarios.policies import (
    Decision,
    defer_to_explicit_statement,
    prefer_least_privilege_source,
    rank_by_deadline_then_evidence,
    rank_by_reliability_over_recency,
    rank_or_abstain_when_close,
)
from kleos_training_data.scenarios.reasoning import MAX_REASONING_CHARS, render_reasoning
from kleos_training_data.scenarios.situations import Item, Situation
from kleos_training_data.scenarios.surrogates import SurrogatePool


def sit(*items: Item) -> Situation:
    return Situation(
        task="notification_prioritization",
        family="notif.test",
        point_index=0,
        axes={"format": "bullets", "presentation_order": "as_given"},
        items=items,
    )


@pytest.fixture(scope="module")
def candidates() -> list[Candidate]:
    out: list[Candidate] = []
    for scenario in load_catalog():
        pool = SurrogatePool.load(scenario.entities.pool)
        out.extend(generate(scenario, pool))
    return out


class TestRenderReasoning:
    def test_it_shows_the_math_for_the_worked_example(self) -> None:
        situation = sit(
            Item("a", "Redpine", 2, "reported", "high", ""),
            Item("b", "Pennfold", 0, "single_source", "low", ""),
            Item("c", "Fieldstone", 2, "single_source", "low", ""),
        )
        text = render_reasoning(situation, rank_by_deadline_then_evidence(situation))
        assert "Redpine: due in 2 days (0.33), reported (0.55), high impact (1.00) → 0.183." in text
        assert "impact (1.00 vs 0.30)" in text
        assert text.endswith("So: Redpine, then Pennfold, then Fieldstone.")

    def test_a_close_call_decline_says_why_and_names_its_label(self) -> None:
        situation = sit(
            Item("a", "Redpine", 5, "reported", "medium", ""),
            Item("b", "Pennfold", 5, "reported", "medium", ""),
        )
        text = render_reasoning(situation, rank_or_abstain_when_close(situation))
        assert "below the 15% needed to call it" in text
        assert text.endswith("so I will ask instead of ranking: insufficient separation.")

    def test_equal_evidence_is_never_called_the_weaker_one(self) -> None:
        situation = sit(
            Item("a", "Redpine", 2, "reported", "high", ""),
            Item("b", "Pennfold", 2, "reported", "high", ""),
        )
        text = render_reasoning(situation, rank_or_abstain_when_close(situation))
        assert "weaker" not in text
        assert "equally supported (0.55)" in text

    def test_the_explicit_basis_does_not_call_a_stated_record_inferred(self) -> None:
        situation = Situation(
            task="memory_conflict_resolution",
            family="mem.test",
            point_index=0,
            axes={"format": "bullets", "presentation_order": "as_given"},
            items=(
                Item("a", "Fieldstone", 4, "confirmed", "high", ""),
                Item("b", "Grayling", 14, "confirmed", "high", ""),
                Item("c", "Hartwell", 7, "corroborated", "high", ""),
            ),
            framing="memory",
        )
        text = render_reasoning(situation, defer_to_explicit_statement(situation))
        assert "the rest are inferred" not in text
        assert "outranks every inferred record" in text

    def test_recency_is_named_as_recency_outside_due_dates(self) -> None:
        situation = Situation(
            task="memory_conflict_resolution",
            family="mem.test",
            point_index=0,
            axes={"format": "bullets", "presentation_order": "as_given"},
            items=(
                Item("a", "Ashgrove", 3, "corroborated", "high", ""),
                Item("b", "Wrenhaven", 9, "corroborated", "high", ""),
            ),
            framing="memory",
        )
        text = render_reasoning(situation, rank_by_reliability_over_recency(situation))
        assert "differ on recency (recorded 3 days ago vs recorded about 1 week ago)" in text
        assert "so deadline decides" in text

    def test_least_privilege_names_its_tiebreak(self) -> None:
        situation = Situation(
            task="tool_routing",
            family="route.test",
            point_index=0,
            axes={"format": "bullets", "presentation_order": "as_given"},
            items=(
                Item("a", "Basin", 2, "confirmed", "low", ""),
                Item("b", "Corvid", 2, "corroborated", "low", ""),
            ),
            framing="routing",
        )
        text = render_reasoning(situation, prefer_least_privilege_source(situation))
        assert "the narrowest wins, with stronger evidence breaking ties: Basin (" in text

    def test_a_decision_without_a_trace_is_refused(self) -> None:
        situation = sit(
            Item("a", "Redpine", 2, "reported", "high", ""),
            Item("b", "Pennfold", 0, "single_source", "low", ""),
        )
        with pytest.raises(ScenarioError, match="trace"):
            render_reasoning(situation, Decision(("a", "b"), "impact", ("x", "y")))

    def test_every_line_ends_a_sentence(self, candidates: list[Candidate]) -> None:
        for candidate in candidates:
            for line in render_reasoning(candidate.situation, candidate.decision).splitlines():
                assert line.endswith((".", ":")), line

    def test_it_stays_under_the_cap_with_no_five_digit_runs(
        self, candidates: list[Candidate]
    ) -> None:
        for candidate in candidates:
            text = render_reasoning(candidate.situation, candidate.decision)
            assert len(text) <= MAX_REASONING_CHARS
            assert not re.search(r"\d{5}", text)

    def test_a_committed_conclusion_lists_the_ranking_in_order(
        self, candidates: list[Candidate]
    ) -> None:
        for candidate in candidates:
            if candidate.decision.abstained:
                continue
            names = [candidate.situation.item(k).name for k in candidate.decision.ranking]
            text = render_reasoning(candidate.situation, candidate.decision)
            assert text.endswith("So: " + ", then ".join(names) + ".")

    def test_a_decline_ends_by_naming_its_label(self, candidates: list[Candidate]) -> None:
        for candidate in candidates:
            if not candidate.decision.abstained:
                continue
            label = candidate.decision.deciding_factor.replace("_", " ")
            text = render_reasoning(candidate.situation, candidate.decision)
            assert text.endswith(f"so I will ask instead of ranking: {label}.")

    def test_every_item_is_shown_in_prompt_order(self, candidates: list[Candidate]) -> None:
        for candidate in candidates:
            lines = render_reasoning(candidate.situation, candidate.decision).splitlines()
            shown = [line.split(":", 1)[0][2:] for line in lines if line.startswith("- ")]
            assert shown == [item.name for item in candidate.situation.presented()]

    def test_it_is_deterministic(self, candidates: list[Candidate]) -> None:
        for candidate in candidates[:50]:
            first = render_reasoning(candidate.situation, candidate.decision)
            assert first == render_reasoning(candidate.situation, candidate.decision)


class TestAnswersAndAttachment:
    def test_every_non_json_answer_states_its_label(self, candidates: list[Candidate]) -> None:
        for candidate in candidates:
            if candidate.variation_axes.get("format") == "json":
                continue
            label = f"What decided it: {candidate.decision.deciding_factor}."
            assert label in candidate.messages[-1]["content"]

    def test_json_answers_are_untouched(self, candidates: list[Candidate]) -> None:
        for candidate in candidates:
            if candidate.variation_axes.get("format") != "json":
                continue
            assert "What decided it" not in candidate.messages[-1]["content"]
            assert "reasoning" not in candidate.messages[-1]

    def test_reasoning_is_attached_to_every_non_json_answer(
        self, candidates: list[Candidate]
    ) -> None:
        for candidate in candidates:
            if candidate.variation_axes.get("format") == "json":
                continue
            assistant = candidate.messages[-1]
            assert assistant["reasoning"] == render_reasoning(
                candidate.situation, candidate.decision
            )

    @pytest.mark.requires_kleos_models
    def test_the_kleos_models_grader_reads_every_label(self, candidates: list[Candidate]) -> None:
        from kleos_models.evaluation.graders import extract_deciding_factor

        labels = sorted({c.decision.deciding_factor for c in candidates})
        for candidate in candidates:
            if candidate.variation_axes.get("format") == "json":
                continue
            found = extract_deciding_factor(candidate.messages[-1]["content"], allowed=labels)
            assert found == candidate.decision.deciding_factor


_PRODUCT = re.compile(r"^- (.+?): .* → (\d+\.\d{3})\.$")


def _products(text: str) -> dict[str, str]:
    found = {}
    for line in text.splitlines():
        match = _PRODUCT.match(line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def _favours(top: Item, runner_up: Item, factor: str, situation: Situation) -> str:
    if factor == "evidence":
        a, b = top.evidence_weight, runner_up.evidence_weight
    elif factor == "impact":
        a, b = top.impact_weight, runner_up.impact_weight
    elif situation.time_sense == "due":
        a, b = top.deadline_score, runner_up.deadline_score
    else:
        a, b = -top.deadline_days, -runner_up.deadline_days
    return "top" if a > b else "runner_up" if b > a else "neither"


class TestTraceFaithfulness:
    def test_scores_never_increase_down_the_ranking(self, candidates: list[Candidate]) -> None:
        for candidate in candidates:
            trace = candidate.decision.trace
            if candidate.variation_axes.get("format") == "json" or trace is None:
                continue
            if not trace.scored or candidate.decision.abstained:
                continue
            products = _products(render_reasoning(candidate.situation, candidate.decision))
            ranked = [candidate.situation.item(k) for k in candidate.decision.ranking]
            groups = [[i for i in ranked if i.in_scope], [i for i in ranked if not i.in_scope]]
            if trace.basis != "scope":
                groups = [ranked]
            for group in groups:
                values = [float(products[item.name]) for item in group]
                assert values == sorted(values, reverse=True), candidate.group_id

    def test_equal_printed_scores_at_the_top_are_called_a_tie(
        self, candidates: list[Candidate]
    ) -> None:
        for candidate in candidates:
            trace = candidate.decision.trace
            if candidate.variation_axes.get("format") == "json" or trace is None:
                continue
            if not trace.scored or candidate.decision.abstained:
                continue
            top, runner_up = (candidate.situation.item(k) for k in candidate.decision.ranking[:2])
            if top.in_scope != runner_up.in_scope:
                continue
            text = render_reasoning(candidate.situation, candidate.decision)
            products = _products(text)
            if products[top.name] == products[runner_up.name]:
                assert "tied" in text, candidate.group_id

    def test_a_factor_favouring_the_runner_up_is_never_said_to_decide(
        self, candidates: list[Candidate]
    ) -> None:
        for candidate in candidates:
            trace = candidate.decision.trace
            if candidate.variation_axes.get("format") == "json" or trace is None:
                continue
            if trace.basis != "first_difference":
                continue
            situation, decision = candidate.situation, candidate.decision
            top, runner_up = (situation.item(k) for k in decision.ranking[:2])
            if _favours(top, runner_up, decision.deciding_factor, situation) == "runner_up":
                text = render_reasoning(situation, decision)
                assert f"so {decision.deciding_factor} decides" not in text, candidate.group_id
