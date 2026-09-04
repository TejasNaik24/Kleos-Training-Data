"""The content layer: does an example teach what its label says it teaches?

Every test here corresponds to a defect that shipped in v0.0.1 and produced
examples that were schema-valid, privacy-clean, gate-passing and *wrong*. That
combination is the dangerous one: nothing in the pipeline objected, because
nothing in the pipeline was checking whether the rendered content matched the
task it was filed under.

The general shape of the failure is a policy or a renderer that ignores an input
it is given. A branch that cannot be reached, a parameter that is accepted and
dropped, an axis that measures the inverse of its name.
"""

from __future__ import annotations

import pytest

from kleos_training_data.scenarios.generator import (
    _DIFFICULTY_SPREAD,
    _DISTRACTORS,
    generate,
)
from kleos_training_data.scenarios.loader import load_catalog
from kleos_training_data.scenarios.policies import POLICIES
from kleos_training_data.scenarios.rendering import render_prompt, render_system_prompt
from kleos_training_data.scenarios.situations import FRAMINGS, Item, Situation
from kleos_training_data.scenarios.surrogates import load_pools


@pytest.fixture(scope="module")
def catalog():
    return {s.family: s for s in load_catalog()}


@pytest.fixture(scope="module")
def pools():
    return load_pools()


def _generate(scenario, pools):
    return generate(scenario, pools[scenario.entities.pool])


class TestFramingReachesTheOutput:
    def test_each_framing_has_its_own_system_prompt(self) -> None:
        """`render_system_prompt` took the situation and ignored it, so all seven
        tasks shipped the prioritization instruction — including tool_routing,
        whose examples were told to prioritize competing work and handed a list
        of candidate sources."""
        systems = {name: f.system for name, f in FRAMINGS.items()}
        assert len(set(systems.values())) == len(FRAMINGS), (
            "two framings share a system prompt, so the task they name is not "
            "reflected in what the model is instructed to do"
        )

    def test_the_catalog_uses_more_than_one_framing(self, catalog, pools) -> None:
        rendered = {
            render_system_prompt(_generate(s, pools)[0].situation) for s in catalog.values()
        }
        assert len(rendered) > 1, (
            "every family renders the same system prompt; the task labels are decoration"
        )

    @pytest.mark.parametrize("framing", sorted(FRAMINGS))
    def test_a_framing_names_what_its_number_measures(self, framing: str) -> None:
        assert FRAMINGS[framing].time_sense in ("due", "age", "staleness")

    def test_a_record_is_never_recorded_in_the_future(self) -> None:
        """`deadline_days` rendered as a due date in every framing produced
        "recorded as of in 4 days" for stored records."""
        item = Item(
            key="a", name="X", deadline_days=4, evidence="reported", impact="low", detail=""
        )
        assert item.age_phrase == "recorded 4 days ago"
        assert item.staleness_phrase == "last synced 4 days ago"
        assert item.deadline_phrase == "due in 4 days"

    def test_time_phrases_are_grammatical(self) -> None:
        """ "due in about 1 weeks"."""
        for days in range(0, 90):
            item = Item(
                key="a", name="X", deadline_days=days, evidence="reported", impact="low", detail=""
            )
            for phrase in (item.deadline_phrase, item.age_phrase, item.staleness_phrase):
                assert " 1 weeks" not in phrase, f"{days}d -> {phrase}"
                assert " 1 months" not in phrase, f"{days}d -> {phrase}"
                assert " 1 days" not in phrase, f"{days}d -> {phrase}"


class TestWorkspaceScopeIsReachable:
    def test_the_scope_branch_actually_fires(self, catalog, pools) -> None:
        """`respect_workspace_scope` matched the literal string "out of scope" in
        an item's detail. No generated detail ever contained it, so the branch was
        unreachable and every workspace_reasoning example silently fell through to
        plain deadline ranking under a workspace label."""
        scenario = catalog["wsp.scope_boundary"]
        decided = [c for c in _generate(scenario, pools) if c.decision.deciding_factor == "scope"]
        assert decided, "the scope branch is unreachable again"

    def test_an_out_of_scope_item_is_marked_in_the_prompt(self, catalog, pools) -> None:
        """The correct answer has to be derivable from the input. A prompt saying
        "only in the workspace I named" that names no workspace and marks no item
        has no stated correct answer."""
        scenario = catalog["wsp.scope_boundary"]
        candidate = _generate(scenario, pools)[0]
        prompt = render_prompt(candidate.situation)
        assert "Active workspace:" in prompt
        assert any(not i.in_scope for i in candidate.situation.items)
        assert "outside the one you named" in prompt or '"workspace": "other"' in prompt

    def test_scope_ranks_before_urgency(self) -> None:
        """An out-of-scope item ranks last however urgent it is."""
        from kleos_training_data.scenarios.policies import respect_workspace_scope

        situation = Situation(
            task="workspace_reasoning",
            family="t",
            point_index=0,
            axes={},
            framing="workspace",
            workspace_name="School",
            items=(
                Item("item_a", "Inside", 30, "single_source", "low", "", scope="in"),
                Item("item_b", "Outside", 0, "confirmed", "high", "", scope="out"),
            ),
        )
        decision = respect_workspace_scope(situation)
        assert decision.deciding_factor == "scope"
        assert decision.ranking[-1] == "item_b", (
            "an out-of-scope item that is due today with confirmed evidence still "
            "ranks last; scope is not a tiebreak"
        )

    def test_every_item_out_of_scope_leaves_nothing_to_answer_with(self) -> None:
        """Guarded at the scenario model, because a situation where everything is
        out of scope has no in-scope answer to give."""
        from pydantic import ValidationError

        from kleos_training_data.scenarios.models import Scenario

        with pytest.raises(ValidationError, match="leaves no item inside"):
            Scenario.model_validate(
                {
                    "family": "t.bad",
                    "task": "workspace_reasoning",
                    "policy_claim": "x" * 20,
                    "anti_claim": "y" * 20,
                    # The workspace axis is declared, so this reaches the
                    # out-of-scope check rather than stopping at the axis check.
                    "axes": {"domain": ["career"], "workspace": ["School"]},
                    "entities": {"pool": "generic_pool_a", "count": 3},
                    "prompt": {
                        "framing": "workspace",
                        "workspace_names": ["School"],
                        "out_of_scope_count": 3,
                    },
                    "expected": {"policy": "respect_workspace_scope"},
                }
            )


class TestGeneratorArtefacts:
    def test_no_prompt_repeats_a_distractor(self, catalog, pools) -> None:
        """Drawing each distractor slot by an independent hash collided, printing
        the same sentence twice in one prompt on 29 of 150 examples."""
        for scenario in catalog.values():
            for candidate in _generate(scenario, pools):
                distractors = candidate.situation.distractors
                assert len(distractors) == len(set(distractors)), (
                    f"{scenario.family} repeats a distractor: {distractors}"
                )

    def test_distractor_count_never_exceeds_the_pool(self, catalog, pools) -> None:
        for scenario in catalog.values():
            for candidate in _generate(scenario, pools):
                assert len(candidate.situation.distractors) <= len(_DISTRACTORS)

    def test_difficulty_is_not_inverted(self) -> None:
        """`spread` staggers each slot's deadline, so spread=0 leaves the field
        bunched. "easy" was the value getting spread=0, which made easy points the
        hardest to separate — the axis measured the opposite of its name."""
        assert _DIFFICULTY_SPREAD["easy"] > _DIFFICULTY_SPREAD["medium"], (
            "an easy point must separate its candidates more than a medium one"
        )
        assert _DIFFICULTY_SPREAD["medium"] > _DIFFICULTY_SPREAD["hard"]


class TestPolicyRegistry:
    def test_every_registered_policy_is_used_by_some_family(self, catalog) -> None:
        """A policy nobody generates from is a claim the dataset does not make."""
        used = {s.expected.policy for s in catalog.values()}
        unused = sorted(set(POLICIES) - used)
        assert not unused, f"registered but never generated from: {unused}"

    def test_every_family_names_a_registered_policy(self, catalog) -> None:
        for scenario in catalog.values():
            assert scenario.expected.policy in POLICIES

    def test_each_family_exhibits_its_distinctive_behaviour(self, catalog, pools) -> None:
        """A family whose special case never fires is filed under a policy it does
        not teach — which is exactly what workspace_reasoning was doing at 0/18."""
        distinctive = {
            "verify_when_evidence_weak": lambda d: d.deciding_factor == "missing_input",
            "ask_when_request_ambiguous": lambda d: d.deciding_factor == "request_ambiguous",
            # Two distinctive outcomes since v0.0.6: the plain explicit-over-inferred
            # case, and Decision B's stale-explicit conflict.
            "defer_to_explicit_statement": lambda d: (
                d.deciding_factor in ("explicit_statement", "stale_explicit_conflict")
            ),
            "prefer_least_privilege_source": lambda d: d.deciding_factor == "least_privilege",
            "respect_workspace_scope": lambda d: d.deciding_factor == "scope",
            "rank_or_abstain_when_close": lambda d: d.abstained,
        }
        for scenario in catalog.values():
            predicate = distinctive.get(scenario.expected.policy)
            if predicate is None:
                continue
            candidates = _generate(scenario, pools)
            hits = sum(1 for c in candidates if predicate(c.decision))
            assert hits, (
                f"{scenario.family} never exhibits the behaviour of "
                f"{scenario.expected.policy}; it is labelled for a policy it does "
                f"not demonstrate"
            )


class TestPersonNameRule:
    def test_a_name_does_not_span_a_line_break(self) -> None:
        """`\\s+` matched a newline, so "Active workspace: Research\\nBlue Harbor"
        matched "Research Blue" and redaction rewrote the prompt to "Active
        workspace: Sam Ridley Harbor" while the assistant still said Blue Harbor.
        97 candidates were corrupted this way and correctly rejected by the
        no_unsupported_claims gate."""
        from kleos_training_data.privacy.detect import scan_text

        detections = scan_text("Active workspace: Research\nBlue Harbor is due", field_path="t")
        matched_across_lines = [d for d in detections if d.kind == "person_name"]
        assert not matched_across_lines, (
            "a person-name bigram matched across a line break; redaction will "
            "corrupt the surrounding text"
        )

    def test_a_real_looking_name_on_one_line_is_still_detected(self) -> None:
        """Narrowing the separator must not weaken the rule."""
        from kleos_training_data.privacy.detect import scan_text

        detections = scan_text("Ask Priya Raghavan about it", field_path="t")
        assert any(d.kind == "person_name" for d in detections)


class TestCaptureIdentity:
    def test_two_paraphrases_of_one_point_get_different_capture_ids(self, catalog, pools) -> None:
        """The mock adapter seeded capture_id from the axes, which a paraphrase
        does not change — so both members of a `count: 2` paraphrase group
        produced one id and the second silently overwrote the first on disk. A
        1152-request batch landed 948 files while reporting 1152 captured."""
        from kleos_training_data.collection.adapters import MockBackendAdapter
        from kleos_training_data.collection.runner import to_request

        adapter = MockBackendAdapter()
        seen: dict[str, str] = {}
        for scenario in catalog.values():
            for candidate in _generate(scenario, pools):
                capture = adapter.run(to_request(scenario, candidate), batch_id="t")
                label = f"{scenario.family}:{candidate.situation.point_index}"
                assert capture.capture_id not in seen, (
                    f"capture_id collision between {seen[capture.capture_id]} and "
                    f"{label}; one would overwrite the other"
                )
                seen[capture.capture_id] = label


class TestV003Corrections:
    """Each defect the v0.0.2 human review surfaced, pinned so it cannot return."""

    def test_workspace_metadata_matches_the_rendered_prompt(self, catalog, pools) -> None:
        """v0.0.2 shipped 338 examples whose `variation_axes.workspace` named a
        different workspace than their own prompt, because the axis and the
        rendered name came from two independent mechanisms."""
        import re

        mismatches = []
        for scenario in catalog.values():
            if scenario.prompt.framing != "workspace":
                continue
            for candidate in _generate(scenario, pools):
                prompt = render_prompt(candidate.situation)
                found = re.search(r"Active workspace: (\S+)", prompt)
                axis = candidate.variation_axes.get("workspace")
                if found and axis and found.group(1) != axis:
                    mismatches.append((candidate.group_id, axis, found.group(1)))
        assert not mismatches, (
            f"{len(mismatches)} example(s) label a workspace their prompt does not "
            f"name, e.g. {mismatches[:3]}"
        )

    def test_a_workspace_scenario_must_declare_the_axis(self) -> None:
        from pydantic import ValidationError

        from kleos_training_data.scenarios.models import Scenario

        with pytest.raises(ValidationError, match="must declare a `workspace` axis"):
            Scenario.model_validate(
                {
                    "family": "t.noaxis",
                    "task": "workspace_reasoning",
                    "policy_claim": "x" * 20,
                    "anti_claim": "y" * 20,
                    "axes": {"domain": ["career"]},
                    "entities": {"pool": "generic_pool_a", "count": 3},
                    "prompt": {"framing": "workspace", "workspace_names": ["School"]},
                    "expected": {"policy": "respect_workspace_scope"},
                }
            )

    def test_no_answer_compares_a_phrase_with_itself(self, catalog, pools) -> None:
        """23 answers said "the nearer deadline wins, and X is due in about 1 week
        against due in about 1 week" — 8 and 9 days round to one phrase, so the
        stated reason compared a phrase with itself."""
        import re

        bad = []
        for scenario in catalog.values():
            for candidate in _generate(scenario, pools):
                answer = candidate.messages[-1]["content"]
                if re.search(r"\b(is|was) ([^,.]{4,60}) against \2\b", answer):
                    bad.append(candidate.group_id)
        assert not bad, f"{len(bad)} tautological comparison(s), e.g. {bad[:3]}"

    def test_absent_in_active_is_workspace_dependent(self, catalog, pools) -> None:
        """The family claimed to teach "ask before crossing a boundary" while
        generating every item in scope, so half its examples resolved as ordinary
        ranking and none mentioned a workspace."""
        from kleos_training_data.scenarios.policies import rank_by_deadline_then_evidence

        scenario = catalog["wsp.absent_in_active"]
        candidates = _generate(scenario, pools)
        crossing = [c for c in candidates if c.decision.deciding_factor == "ask_before_crossing"]
        assert len(crossing) == len(candidates), (
            f"only {len(crossing)}/{len(candidates)} ask before crossing"
        )

        # Counterfactual: neutralise scope and the decision must change.
        changed = 0
        for candidate in candidates:
            situation = candidate.situation
            neutral = tuple(
                Item(i.key, i.name, i.deadline_days, i.evidence, i.impact, i.detail, scope="in")
                for i in situation.items
            )
            without = rank_by_deadline_then_evidence(
                Situation(**{**situation.__dict__, "items": neutral, "framing": "priority"})
            )
            if (candidate.decision.ranking, candidate.decision.abstained) != (
                without.ranking,
                without.abstained,
            ):
                changed += 1
        assert changed == len(candidates), (
            f"only {changed}/{len(candidates)} change when workspace info is removed; "
            f"the rest are generic reasoning under a workspace label"
        )

    def test_tool_routing_candidates_are_sources_not_projects(self, catalog, pools) -> None:
        """v0.0.2 asked students to route between `Northwind` and `Marchwood`."""
        from kleos_training_data.scenarios.surrogates import load_pools as _pools

        project_names = set(_pools()["generic_pool_a"].values)
        for scenario in catalog.values():
            if scenario.task != "tool_routing":
                continue
            assert scenario.entities.pool.startswith("source_pool"), (
                f"{scenario.family} draws its candidate sources from {scenario.entities.pool!r}"
            )
            prompt = render_prompt(_generate(scenario, pools)[0].situation)
            assert not (project_names & set(prompt.split())), (
                f"{scenario.family} still renders project names as sources"
            )

    def test_no_source_name_predicts_the_answer(self, catalog, pools) -> None:
        """Realistic source names must not become the shortcut: properties are
        drawn independently of the name, so no name may dominate the winner."""
        import collections
        import re

        wins: collections.Counter[str] = collections.Counter()
        for scenario in catalog.values():
            if scenario.task != "tool_routing":
                continue
            for candidate in _generate(scenario, pools):
                answer = candidate.messages[-1]["content"]
                found = (
                    re.search(r'"ranking":\s*\[\s*"([^"]+)"', answer)
                    or re.search(r"^1\.\s+(.+?) —", answer, re.M)
                    or re.search(r"Start with (.+?) —", answer)
                )
                if found:
                    wins[found.group(1).strip()] += 1
        total = sum(wins.values())
        top_share = wins.most_common(1)[0][1] / total
        assert top_share < 0.25, (
            f"one source wins {top_share:.0%} of routing answers; the name has become the shortcut"
        )
