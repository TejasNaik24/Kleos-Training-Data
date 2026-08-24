"""Scenarios generate reproducible, policy-teaching examples.

The properties here are the research claim in executable form. A scenario system
that generates *plausible* examples is easy; one whose targets are derived from a
stated policy, whose perturbations provably preserve the decision, and whose
output is reproducible byte-for-byte is what makes the dataset defensible.
"""

from __future__ import annotations

import pytest

from kleos_training_data.contract.constants import PERTURBATION_KINDS, SUPPORTED_TASKS
from kleos_training_data.contract.schemas import TrainingExample
from kleos_training_data.errors import ConfigError, ScenarioError
from kleos_training_data.ids import example_id
from kleos_training_data.scenarios.generator import _perturb, generate, generation_fingerprint
from kleos_training_data.scenarios.loader import load_catalog, load_scenario
from kleos_training_data.scenarios.models import Scenario
from kleos_training_data.scenarios.policies import (
    POLICIES,
    decide,
    rank_by_deadline_then_evidence,
    rank_or_abstain_when_close,
    resolve_policy,
)
from kleos_training_data.scenarios.rendering import (
    PROMPT_FORMATS,
    render_answer,
    render_prompt,
    render_system_prompt,
)
from kleos_training_data.scenarios.situations import Item, Situation
from kleos_training_data.scenarios.surrogates import SurrogatePool, load_pools


def make_item(key: str, *, days: int, evidence: str = "confirmed", impact: str = "high") -> Item:
    return Item(
        key=key,
        name=key.replace("item_", "Item ").upper(),
        deadline_days=days,
        evidence=evidence,
        impact=impact,
        detail="A note.",
    )


def make_situation(*items: Item, **axes: str) -> Situation:
    return Situation(
        task="notification_prioritization",
        family="test.family",
        point_index=0,
        axes={"domain": "career", **axes},
        items=items or (make_item("item_a", days=1), make_item("item_b", days=10)),
    )


class TestPolicyCorrectness:
    """The policy encodes the research claim, so its edge cases are the claim."""

    def test_the_nearer_deadline_wins_when_evidence_matches(self) -> None:
        decision = rank_by_deadline_then_evidence(
            make_situation(
                make_item("item_a", days=10),
                make_item("item_b", days=2),
            )
        )
        assert decision.ranking[0] == "item_b"
        assert decision.deciding_factor == "deadline"

    def test_confirmed_beats_sooner_but_unverified(self) -> None:
        """The lesson the dataset exists to teach.

        Urgency alone is not a reason to act. An item due tomorrow that nobody
        has checked ranks below one due in three days that is confirmed, because
        acting on the unverified one risks doing the wrong work entirely.
        """
        decision = rank_by_deadline_then_evidence(
            make_situation(
                make_item("item_a", days=1, evidence="unverified"),
                make_item("item_b", days=3, evidence="confirmed"),
            )
        )
        assert decision.ranking[0] == "item_b"
        assert decision.deciding_factor == "evidence"

    def test_impact_decides_when_timing_and_evidence_match(self) -> None:
        decision = rank_by_deadline_then_evidence(
            make_situation(
                make_item("item_a", days=5, impact="low"),
                make_item("item_b", days=5, impact="high"),
            )
        )
        assert decision.ranking[0] == "item_b"
        assert decision.deciding_factor == "impact"

    def test_the_decision_ignores_presentation_order(self) -> None:
        """A policy that depended on ordering would be teaching a position bias."""
        items = (make_item("item_a", days=10), make_item("item_b", days=2))
        forward = rank_by_deadline_then_evidence(make_situation(*items))
        reverse = rank_by_deadline_then_evidence(
            make_situation(*reversed(items), presentation_order="reversed")
        )
        assert forward.ranking == reverse.ranking

    def test_a_close_call_abstains_rather_than_guessing(self) -> None:
        """A dataset of only clean decisions teaches that one is always available."""
        decision = rank_or_abstain_when_close(
            make_situation(
                make_item("item_a", days=30, evidence="single_source", impact="low"),
                make_item("item_b", days=31, evidence="single_source", impact="low"),
            )
        )
        assert decision.abstained
        assert decision.resolver, "abstaining without naming a next step is not useful"

    def test_a_clear_call_does_not_abstain(self) -> None:
        decision = rank_or_abstain_when_close(
            make_situation(
                make_item("item_a", days=1, evidence="confirmed", impact="high"),
                make_item("item_b", days=60, evidence="unverified", impact="low"),
            )
        )
        assert not decision.abstained

    def test_ranking_a_single_item_is_an_error(self) -> None:
        with pytest.raises(ScenarioError, match="at least two"):
            rank_by_deadline_then_evidence(make_situation(make_item("item_a", days=1)))

    @pytest.mark.parametrize("name", sorted(POLICIES))
    def test_every_policy_is_deterministic(self, name: str) -> None:
        situation = make_situation()
        assert decide(situation, name) == decide(situation, name)

    def test_an_unregistered_policy_names_the_alternatives(self) -> None:
        with pytest.raises(ScenarioError, match="Unknown policy"):
            resolve_policy("rank_by_vibes")


class TestRendering:
    @pytest.mark.parametrize("fmt", PROMPT_FORMATS)
    def test_every_prompt_format_renders(self, fmt: str) -> None:
        situation = make_situation(format=fmt)
        rendered = render_prompt(situation)
        assert rendered.strip()
        for item in situation.items:
            assert item.name in rendered, f"{fmt} dropped {item.name}"

    @pytest.mark.parametrize("fmt", PROMPT_FORMATS)
    def test_every_answer_format_renders(self, fmt: str) -> None:
        situation = make_situation(format=fmt)
        decision = rank_by_deadline_then_evidence(situation)
        assert render_answer(situation, decision).strip()

    def test_an_unknown_format_is_an_error(self) -> None:
        with pytest.raises(ScenarioError, match="Unknown prompt format"):
            render_prompt(make_situation(format="interpretive_dance"))

    def test_the_system_prompt_states_the_policy_not_the_answer(self) -> None:
        """A system prompt naming the winner makes every example trivial."""
        prompt = render_system_prompt(make_situation())
        for item in make_situation().items:
            assert item.name not in prompt

    def test_the_answer_names_the_deciding_factor(self) -> None:
        """The stated reason is derived from the same computation as the ranking,
        so it cannot drift into a plausible-sounding but wrong justification."""
        situation = make_situation(
            make_item("item_a", days=1, evidence="unverified"),
            make_item("item_b", days=3, evidence="confirmed"),
        )
        decision = rank_by_deadline_then_evidence(situation)
        answer = render_answer(situation, decision)
        assert "evidence" in answer.lower()

    def test_the_answer_asserts_nothing_absent_from_the_prompt(self) -> None:
        """What the `no_unsupported_claims` review gate checks for."""
        situation = make_situation()
        decision = rank_by_deadline_then_evidence(situation)
        answer = render_answer(situation, decision)
        prompt = render_prompt(situation)

        # Every entity named in the answer must appear in the prompt.
        for item in situation.items:
            if item.name in answer:
                assert item.name in prompt


class TestSurrogates:
    """Surrogate keying is what defeats memorization; the properties are the design."""

    def test_names_are_stable_within_a_family(self) -> None:
        """Coreference: "Northwind" in turn one is "Northwind" in turn three."""
        pool = SurrogatePool.load("generic_pool_a")
        first = pool.names("fam.a", point_index=0, count=3)
        second = pool.names("fam.a", point_index=0, count=3)
        assert first == second

    def test_names_differ_across_families(self) -> None:
        """No persistent pseudo-entity spans the corpus for a model to memorize."""
        pool = SurrogatePool.load("generic_pool_a")
        a = pool.names("fam.a", point_index=0, count=3)
        b = pool.names("fam.b", point_index=0, count=3)
        assert a != b

    def test_names_within_one_situation_are_distinct(self) -> None:
        """Two identically named items would make a ranking unreadable."""
        pool = SurrogatePool.load("generic_pool_a")
        names = pool.names("fam.a", point_index=3, count=4)
        assert len(set(names)) == 4

    def test_requesting_more_names_than_exist_is_an_error(self) -> None:
        pool = SurrogatePool.load("generic_pool_a")
        with pytest.raises(ConfigError, match="names but"):
            pool.names("fam.a", point_index=0, count=len(pool.values) + 1)

    def test_a_missing_pool_lists_the_available_ones(self) -> None:
        with pytest.raises(ConfigError, match="not found"):
            SurrogatePool.load("no_such_pool")

    def test_every_committed_pool_loads(self) -> None:
        pools = load_pools()
        assert pools, "no surrogate pools are committed"
        for name, pool in pools.items():
            assert len(pool.values) >= 8, f"{name} is too small to avoid recurrence"
            assert len(set(pool.values)) == len(pool.values)

    def test_the_holdout_pool_is_disjoint_from_the_training_pool(self) -> None:
        """A reserved entity that also appears in training is not held out at all."""
        pools = load_pools()
        train = set(pools["generic_pool_a"].values)
        held = set(pools["generic_pool_z"].values)
        assert not (train & held), f"overlap would break the OOD claim: {train & held}"


class TestPerturbations:
    """A perturbation must change the presentation and preserve the decision."""

    @pytest.mark.parametrize("kind", PERTURBATION_KINDS)
    def test_every_registered_kind_is_implemented(self, kind: str) -> None:
        """A kind a scenario may declare but the generator cannot produce would
        fail at generation time, long after the scenario was written."""
        situation = make_situation(format="bullets", presentation_order="as_given")
        assert _perturb(situation, kind, 0) is not None

    @pytest.mark.parametrize("kind", PERTURBATION_KINDS)
    def test_every_kind_actually_changes_something(self, kind: str) -> None:
        """A no-op perturbation is an exact duplicate wearing a label."""
        situation = make_situation(format="bullets", presentation_order="as_given")
        variant = _perturb(situation, kind, 0)
        changed = (
            variant.axes != situation.axes
            or variant.distractors != situation.distractors
            or variant.question != situation.question
        )
        assert changed, f"{kind} left the situation untouched"

    @pytest.mark.parametrize("kind", PERTURBATION_KINDS)
    def test_every_kind_preserves_the_decision(self, kind: str) -> None:
        situation = make_situation(
            make_item("item_a", days=2, evidence="confirmed"),
            make_item("item_b", days=9, evidence="reported"),
            format="bullets",
            presentation_order="as_given",
        )
        base = rank_by_deadline_then_evidence(situation)
        variant = rank_by_deadline_then_evidence(_perturb(situation, kind, 0))
        assert variant.comparable() == base.comparable(), (
            f"{kind} changed the decision, so it is not a perturbation — it is a different scenario"
        )

    @pytest.mark.parametrize(
        ("kind", "axis", "value"),
        [
            ("evidence_order", "presentation_order", "reversed"),
            ("formatting", "format", "prose"),
            ("schema", "format", "slack_thread"),
        ],
    )
    def test_a_kind_does_not_no_op_when_the_axis_already_holds_its_value(
        self, kind: str, axis: str, value: str
    ) -> None:
        """The bug this closes: setting an axis to a fixed value silently does
        nothing when the base already holds it, producing a duplicate."""
        situation = make_situation(**{axis: value})
        variant = _perturb(situation, kind, 0)
        assert variant.axes[axis] != value


class TestGeneration:
    def test_the_committed_catalog_generates(self) -> None:
        for scenario in load_catalog():
            pool = SurrogatePool.load(scenario.entities.pool)
            candidates = generate(scenario, pool)
            assert len(candidates) == scenario.expected_example_count

    def test_generated_examples_satisfy_the_public_contract(self) -> None:
        """Generation that produces something the trainer cannot load is wasted."""
        for scenario in load_catalog():
            pool = SurrogatePool.load(scenario.entities.pool)
            for candidate in generate(scenario, pool):
                payload = candidate.to_payload()
                payload["id"] = example_id(payload)
                TrainingExample.model_validate(payload)

    def test_generation_is_reproducible(self) -> None:
        """Same scenario, same seed, byte-identical output on any machine."""
        scenario = load_catalog()[0]
        pool = SurrogatePool.load(scenario.entities.pool)
        first = [example_id(c.to_payload()) for c in generate(scenario, pool)]
        second = [example_id(c.to_payload()) for c in generate(scenario, pool)]
        assert first == second

    def test_no_two_generated_candidates_are_identical(self) -> None:
        for scenario in load_catalog():
            pool = SurrogatePool.load(scenario.entities.pool)
            ids = [example_id(c.to_payload()) for c in generate(scenario, pool)]
            assert len(set(ids)) == len(ids), f"{scenario.family} generated duplicates"

    def test_a_perturbation_shares_its_bases_group(self) -> None:
        """A perturbation pair straddling the split boundary compares nothing."""
        scenario = load_catalog()[0]
        candidates = generate(scenario, SurrogatePool.load(scenario.entities.pool))
        by_group: dict[str, list] = {}
        for candidate in candidates:
            by_group.setdefault(candidate.group_id, []).append(candidate)

        for group, members in by_group.items():
            bases = [m for m in members if m.perturbation_of is None]
            assert len(bases) == 1, f"group {group} has {len(bases)} base examples"
            assert len(members) >= 2, f"group {group} cannot be consistency-tested"

    def test_perturbations_reference_their_base_by_derived_id(self) -> None:
        scenario = load_catalog()[0]
        candidates = generate(scenario, SurrogatePool.load(scenario.entities.pool))
        base_ids = {example_id(c.to_payload()) for c in candidates if c.perturbation_of is None}
        for candidate in candidates:
            if candidate.perturbation_of is not None:
                assert candidate.perturbation_of in base_ids

    def test_the_fingerprint_changes_when_the_scenario_does(self) -> None:
        scenario = load_catalog()[0]
        altered = scenario.model_copy(
            update={"generation": scenario.generation.model_copy(update={"seed": 999})}
        )
        assert generation_fingerprint(scenario) != generation_fingerprint(altered)


class TestCatalogLoading:
    def test_the_committed_catalog_loads(self) -> None:
        scenarios = load_catalog()
        assert scenarios
        for scenario in scenarios:
            assert scenario.task in SUPPORTED_TASKS

    def test_every_scenario_states_a_policy_claim_and_an_anti_claim(self) -> None:
        """A scenario that cannot state its anti-claim has usually not identified
        what it is testing."""
        for scenario in load_catalog():
            assert len(scenario.policy_claim.strip()) > 20
            assert len(scenario.anti_claim.strip()) > 20

    def test_a_typo_in_a_field_name_is_rejected(self, tmp_path) -> None:
        """Silently ignoring `pertubation_kinds` would produce a catalog that
        looks like it tests consistency and does not."""
        path = tmp_path / "broken.yaml"
        path.write_text(
            "family: x.y\ntask: tool_routing\n"
            "policy_claim: a claim long enough to pass\n"
            "anti_claim: an anti claim long enough\n"
            "axes: {domain: [career]}\n"
            "entities: {pool: generic_pool_a}\n"
            "expected: {policy: rank_by_deadline_then_evidence}\n"
            "pertubation_kinds: [paraphrase]\n",
            encoding="utf-8",
        )
        with pytest.raises(ScenarioError, match="not a valid scenario"):
            load_scenario(path)

    def test_a_scenario_without_a_domain_axis_is_rejected(self, tmp_path) -> None:
        path = tmp_path / "nodomain.yaml"
        path.write_text(
            "family: x.y\ntask: tool_routing\n"
            "policy_claim: a claim long enough to pass\n"
            "anti_claim: an anti claim long enough\n"
            "axes: {urgency: [high]}\n"
            "entities: {pool: generic_pool_a}\n"
            "expected: {policy: rank_by_deadline_then_evidence}\n",
            encoding="utf-8",
        )
        with pytest.raises(ScenarioError):
            load_scenario(path)

    def test_two_files_claiming_one_family_is_an_error(self, tmp_path) -> None:
        """Merging two research claims into one scenario_family group would make
        both unmeasurable."""
        body = (
            "family: dup.family\ntask: tool_routing\n"
            "policy_claim: a claim long enough to pass\n"
            "anti_claim: an anti claim long enough\n"
            "axes: {domain: [career]}\n"
            "entities: {pool: generic_pool_a}\n"
            "expected: {policy: rank_by_deadline_then_evidence}\n"
        )
        (tmp_path / "a.yaml").write_text(body, encoding="utf-8")
        (tmp_path / "b.yaml").write_text(body, encoding="utf-8")
        with pytest.raises(ScenarioError, match="claim the family"):
            load_catalog(tmp_path)

    def test_an_unknown_task_is_rejected(self, tmp_path) -> None:
        path = tmp_path / "badtask.yaml"
        path.write_text(
            "family: x.y\ntask: summarization\n"
            "policy_claim: a claim long enough to pass\n"
            "anti_claim: an anti claim long enough\n"
            "axes: {domain: [career]}\n"
            "entities: {pool: generic_pool_a}\n"
            "expected: {policy: rank_by_deadline_then_evidence}\n",
            encoding="utf-8",
        )
        with pytest.raises(ScenarioError):
            load_scenario(path)


class TestScenarioModel:
    def test_axis_space_is_the_product_of_the_axes(self) -> None:
        scenario = load_catalog()[0]
        expected = 1
        for values in scenario.axes.values():
            expected *= len(values)
        assert scenario.axis_space_size == expected

    def test_expected_count_includes_perturbations(self) -> None:
        scenario = load_catalog()[0]
        per_base = 1 + sum(g.count for g in scenario.generation.equivalence_groups)
        assert scenario.expected_example_count == scenario.generation.n_base * per_base

    def test_an_unregistered_perturbation_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown perturbation kind"):
            Scenario.model_validate(
                {
                    "family": "x.y",
                    "task": "tool_routing",
                    "policy_claim": "a claim long enough to pass",
                    "anti_claim": "an anti claim long enough",
                    "axes": {"domain": ["career"]},
                    "entities": {"pool": "generic_pool_a"},
                    "expected": {"policy": "rank_by_deadline_then_evidence"},
                    "generation": {"equivalence_groups": [{"kind": "vibes"}]},
                }
            )
