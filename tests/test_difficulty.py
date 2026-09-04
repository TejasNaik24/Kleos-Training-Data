"""Difficulty must describe the example, not decorate it.

The defect these tests exist to prevent is not "the numbers came out
non-monotonic". It is that a label shipped in `variation_axes.difficulty`
without anything checking it against the situation it labelled — and when
finally measured under each policy's own ordering key, the v0.0.3 labels were
*anti-correlated* with decision difficulty.

So these test the **property**, not the curve: the shipped label must be
recomputable from the example, gates must be distinguished from ranking
criteria, and the threshold must remain derived from the evidence scale rather
than chosen. A monotonicity check is kept at the end as a diagnostic, clearly
marked as such, because a distribution can shift for legitimate reasons while
the property still holds.
"""

from __future__ import annotations

import pytest

from kleos_training_data.scenarios.difficulty import (
    ARBITRARY_TIEBREAK,
    GATE_COMPONENTS,
    ONE_STEP,
    ORDERING_KEYS,
    derive_difficulty,
    evidence_scale_min_step,
    separation,
)
from kleos_training_data.scenarios.generator import generate
from kleos_training_data.scenarios.loader import load_catalog
from kleos_training_data.scenarios.policies import POLICIES
from kleos_training_data.scenarios.situations import Item, Situation
from kleos_training_data.scenarios.surrogates import load_pools


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


@pytest.fixture(scope="module")
def pools():
    return load_pools()


@pytest.fixture(scope="module")
def generated(catalog, pools):
    out = []
    for scenario in catalog:
        for candidate in generate(scenario, pools[scenario.entities.pool]):
            out.append((scenario, candidate))
    return out


def _situation(*items: Item, framing: str = "priority", workspace: str = "School") -> Situation:
    return Situation(
        task="notification_prioritization",
        family="t",
        point_index=0,
        axes={},
        items=items,
        framing=framing,
        workspace_name=workspace,
    )


class TestTheLabelDescribesTheExample:
    def test_every_shipped_label_recomputes_from_its_own_situation(self, generated) -> None:
        """The property the whole module exists for. A label that cannot be
        rederived from the example is a label nothing checked."""
        mismatches = []
        for scenario, candidate in generated:
            shipped = candidate.variation_axes.get("difficulty")
            if shipped is None:
                continue
            recomputed = derive_difficulty(candidate.situation, scenario.expected.policy)
            if shipped != recomputed:
                mismatches.append((scenario.family, shipped, recomputed))
        assert not mismatches, (
            f"{len(mismatches)} example(s) ship a difficulty that does not recompute "
            f"from their situation, e.g. {mismatches[:3]}"
        )

    def test_the_label_is_not_the_declared_value(self, catalog, pools) -> None:
        """Derived, not declared. If the shipped label merely echoed the axis
        declaration, every family declaring a single level would ship only that
        level — which is exactly how the old defect looked from outside."""
        single_level = [s for s in catalog if len(s.axes.get("difficulty", [])) == 1]
        assert single_level, "expected at least one family declaring one difficulty"
        differs = False
        for scenario in single_level:
            declared = scenario.axes["difficulty"][0]
            shipped = {
                c.variation_axes.get("difficulty")
                for c in generate(scenario, pools[scenario.entities.pool])
            }
            if shipped != {declared}:
                differs = True
        assert differs, (
            "no family ships a difficulty differing from its declaration; the label "
            "may have gone back to echoing the axis instead of measuring the situation"
        )

    def test_a_label_is_dropped_rather_than_guessed(self) -> None:
        """A policy with no registered ordering cannot have difficulty measured,
        so no difficulty is claimed."""
        situation = _situation(
            Item("item_a", "A", 1, "confirmed", "high", ""),
            Item("item_b", "B", 9, "unverified", "low", ""),
        )
        assert derive_difficulty(situation, "no_such_policy") is None

    def test_every_registered_policy_has_an_ordering_key(self) -> None:
        """Otherwise its families ship no difficulty at all, silently."""
        missing = sorted(set(POLICIES) - set(ORDERING_KEYS))
        assert not missing, f"policies with no registered ordering key: {missing}"


class TestGatesAreNotRankingCriteria:
    def test_a_differing_gate_is_the_clearest_separation(self) -> None:
        """An out-of-scope candidate is excluded by a stated rule. Nothing has to
        be weighed, so the decision is easy however close the rest looks."""
        situation = _situation(
            Item("item_a", "Inside", 5, "reported", "medium", "", scope="in"),
            Item("item_b", "Outside", 5, "reported", "medium", "", scope="out"),
            framing="workspace",
        )
        parted = separation(situation, "respect_workspace_scope")
        assert parted is not None and parted.is_gate
        assert derive_difficulty(situation, "respect_workspace_scope") == "easy"

    def test_a_shared_gate_value_is_skipped_not_treated_as_a_tie(self) -> None:
        """Two candidates both in scope have not tied on 'what matters' — the
        gate did not apply. Treating that as a tie labelled all four workspace
        families 100% hard, which was an artefact of the definition."""
        situation = _situation(
            Item("item_a", "A", 1, "confirmed", "high", "", scope="in"),
            Item("item_b", "B", 30, "unverified", "low", "", scope="in"),
            framing="workspace",
        )
        parted = separation(situation, "respect_workspace_scope")
        assert parted is not None
        assert not parted.is_gate, "a shared gate value was reported as the separator"
        assert parted.depth == 0, "the gate was counted as a ranking level"
        assert derive_difficulty(situation, "respect_workspace_scope") == "easy"

    @pytest.mark.parametrize("gate", sorted(GATE_COMPONENTS))
    def test_each_gate_is_declared_by_some_policy(self, gate: str) -> None:
        used = {name for comps in ORDERING_KEYS.values() for name, _ in comps}
        assert gate in used, f"{gate!r} is marked a gate but no policy orders by it"


class TestThresholdIsDerivedNotChosen:
    def test_one_step_equals_the_evidence_scales_smallest_step(self) -> None:
        """If this drifts, the threshold has become a tuning knob."""
        assert evidence_scale_min_step() == pytest.approx(ONE_STEP, abs=1e-9)

    def test_a_gap_below_one_step_is_never_easy(self) -> None:
        situation = _situation(
            Item("item_a", "A", 4, "confirmed", "high", ""),
            Item("item_b", "B", 4, "corroborated", "high", ""),
        )
        parted = separation(situation, "rank_by_deadline_then_evidence")
        assert parted is not None and parted.relative_gap < ONE_STEP
        assert derive_difficulty(situation, "rank_by_deadline_then_evidence") != "easy"

    def test_identical_candidates_fall_to_an_arbitrary_tiebreak(self) -> None:
        """Nothing stated separates them, so nobody reading the prompt could
        reproduce the order. That is the hardest case there is."""
        situation = _situation(
            Item("item_a", "A", 5, "reported", "medium", ""),
            Item("item_b", "B", 5, "reported", "medium", ""),
        )
        parted = separation(situation, "rank_by_deadline_then_evidence")
        assert parted is not None and parted.depth == ARBITRARY_TIEBREAK
        assert derive_difficulty(situation, "rank_by_deadline_then_evidence") == "hard"


class TestDepthRaisesDifficulty:
    def test_consulting_a_second_criterion_is_never_easy(self) -> None:
        """Tied on the primary criterion, separated on the secondary: the reader
        had to look further, so this cannot be easy however wide the second gap."""
        situation = _situation(
            Item("item_a", "A", 1, "confirmed", "high", ""),
            Item("item_b", "B", 40, "confirmed", "high", ""),
        )
        parted = separation(situation, "rank_by_reliability_over_recency")
        assert parted is not None
        assert parted.component == "recency" and parted.depth == 1
        assert derive_difficulty(situation, "rank_by_reliability_over_recency") == "medium"

    def test_primary_criterion_deciding_clearly_is_easy(self) -> None:
        situation = _situation(
            Item("item_a", "A", 5, "confirmed", "high", ""),
            Item("item_b", "B", 5, "unverified", "high", ""),
        )
        assert derive_difficulty(situation, "rank_by_reliability_over_recency") == "easy"


class TestPerturbationsPreserveDifficulty:
    def test_a_perturbation_never_changes_the_derived_label(self, generated) -> None:
        """Perturbations change presentation, not candidates. If one changed the
        derived difficulty, the equivalence group would straddle two labels and
        difficulty-sliced consistency analysis would compare unlike things."""
        by_group: dict[str, set[str | None]] = {}
        for _scenario, candidate in generated:
            by_group.setdefault(candidate.group_id, set()).add(
                candidate.variation_axes.get("difficulty")
            )
        split = {g: v for g, v in by_group.items() if len(v) > 1}
        assert not split, f"{len(split)} group(s) span two difficulty labels: {list(split)[:3]}"


class TestDiagnosticNotCorrectness:
    """Distribution checks. A shift here is a prompt to investigate, not proof of
    a defect — which is why these are separated from the property tests above."""

    def test_all_three_levels_are_represented(self, generated) -> None:
        levels = {c.variation_axes.get("difficulty") for _s, c in generated}
        levels.discard(None)
        assert levels == {"easy", "medium", "hard"}, (
            f"only {sorted(levels)} present; a level nothing produces cannot be evaluated against"
        )

    def test_no_single_level_swamps_the_corpus(self, generated) -> None:
        import collections

        counts = collections.Counter(c.variation_axes.get("difficulty") for _s, c in generated)
        counts.pop(None, None)
        total = sum(counts.values())
        top = counts.most_common(1)[0]
        assert top[1] / total < 0.85, (
            f"{top[0]!r} is {100 * top[1] / total:.0f}% of labelled examples; the axis "
            f"has stopped discriminating"
        )
