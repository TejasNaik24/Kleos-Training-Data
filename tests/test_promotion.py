"""Promotion: the gate table, and the four mechanisms behind "no bypass".

``TestForceCannotBypass`` is the heart of this module. "``--force`` cannot skip a
privacy gate" is the kind of claim that is true when written and false two
refactors later, so it is asserted against the mechanism rather than against the
current behaviour: the mandatory set is *derived*, the illegal policy cannot be
constructed, the flag maps to a constant, and the runner proves every mandatory
gate executed.
"""

from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError

from kleos_training_data.contract.dedup import (
    CorpusEntry,
    CorpusIndex,
    DuplicateKind,
    conversation_text,
)
from kleos_training_data.errors import PromotionIntegrityError
from kleos_training_data.ids import CollisionLedger, example_id
from kleos_training_data.privacy.entities import EntityVault
from kleos_training_data.promotion.context import PromotionContext
from kleos_training_data.promotion.gates import (
    BYPASSABLE_GATE_IDS,
    GATES,
    GATES_BY_ID,
    MANDATORY_GATE_IDS,
    PRIVACY_GATE_IDS,
    GateStatus,
)
from kleos_training_data.promotion.policy import PromotionPolicy
from kleos_training_data.promotion.runner import run_gates
from kleos_training_data.review.records import HumanDecision
from kleos_training_data.review.rubric import GateResults
from kleos_training_data.staging.records import CaptureLane, SanitizedCandidate, ScenarioRef


def make_payload(assistant: str = "Start with Northwind — nearest deadline, confirmed.") -> dict:
    return {
        "task": "notification_prioritization",
        "messages": [
            {"role": "system", "content": "Rank the items."},
            {"role": "user", "content": "Northwind is due Friday. Fieldstone is due next month."},
            {"role": "assistant", "content": assistant},
        ],
        "variation_axes": {"domain": "career", "urgency": "high"},
    }


def make_candidate(payload: dict | None = None, **overrides) -> SanitizedCandidate:
    payload = payload or make_payload()
    from kleos_training_data.ids import content_hash

    base = {
        "candidate_id": example_id(payload),
        "batch_id": "b1",
        "lane": CaptureLane.MOCK_BACKEND,
        "scenario": ScenarioRef(
            family="notif.deadline_vs_evidence",
            task="notification_prioritization",
            point_index=0,
            catalog_version="scenarios-v1",
            policy_claim="Rank by expected cost of delay.",
            policy="rank_by_deadline_then_evidence",
            scenario_fingerprint="abc123",
        ),
        "sanitization_status": "clean",
        "ruleset_version": "privacy-rules-v1",
        "payload": payload,
        "scenario_family": "notif.deadline_vs_evidence",
        "group_id": "notif.deadline_vs_evidence:0000",
        "content_hash": content_hash(payload),
    }
    base.update(overrides)
    return SanitizedCandidate(**base).sealed()


def make_decision(candidate: SanitizedCandidate, **overrides) -> HumanDecision:
    base = {
        "candidate_id": candidate.candidate_id,
        "content_hash": candidate.content_hash,
        "decision": "approve",
        "gates": GateResults.all_pass(),
    }
    base.update(overrides)
    return HumanDecision(**base).signed()


def make_context(candidate: SanitizedCandidate | None = None, **overrides) -> PromotionContext:
    candidate = candidate or make_candidate()
    ledger = CollisionLedger()
    ledger.mint(candidate.payload)
    base = {
        "candidate": candidate,
        "privacy": {"status": "clean"},
        "human": make_decision(candidate),
        "machine": None,
        "ledger": ledger,
        "corpus": CorpusIndex(entries=[]),
        "eval_corpus": CorpusIndex(entries=[]),
        "vault": EntityVault(),
        "allowed_sources": frozenset({"synthetic"}),
    }
    base.update(overrides)
    return PromotionContext(**base)


class TestGateTable:
    def test_there_are_fourteen_gates(self) -> None:
        assert len(GATES) == 14

    def test_gate_ids_are_unique_and_ordered(self) -> None:
        ids = [spec.id for spec in GATES]
        assert len(set(ids)) == len(ids)
        assert ids == sorted(ids), "execution order should read in id order"

    def test_mandatory_is_derived_from_the_table(self) -> None:
        """Not hand-maintained. A list maintained beside the table drifts the
        first time somebody adds a gate, and drifts permissively."""
        assert frozenset(s.id for s in GATES if not s.bypassable) == MANDATORY_GATE_IDS
        assert frozenset(s.id for s in GATES if s.bypassable) == BYPASSABLE_GATE_IDS
        assert not (MANDATORY_GATE_IDS & BYPASSABLE_GATE_IDS)

    def test_bypassable_defaults_to_false(self) -> None:
        """A gate added without thought is mandatory."""
        from kleos_training_data.promotion.gates import GateSpec

        spec = GateSpec("G99_NEW", "A newly added gate.", lambda ctx: None)  # type: ignore[arg-type]
        assert not spec.bypassable

    @pytest.mark.parametrize(
        "gate_id",
        [
            "G02_SCHEMA_VALID",
            "G04_SECRET_SCAN",
            "G05_PII_SCAN",
            "G06_SURROGATE_INTEGRITY",
            "G07_PRIVATE_FACT",
            "G08_REVIEW_PRESENT",
            "G09_REVIEW_APPROVED",
            "G10_PROVENANCE",
            "G12_CORPUS_DEDUP",
            "G13_EVAL_LEAKAGE",
            "G14_CONTRACT_RENDER",
        ],
    )
    def test_the_gates_that_matter_are_mandatory(self, gate_id: str) -> None:
        assert gate_id in MANDATORY_GATE_IDS

    def test_every_privacy_gate_is_mandatory(self) -> None:
        assert PRIVACY_GATE_IDS <= MANDATORY_GATE_IDS

    def test_dedup_is_mandatory_because_a_bypass_would_skip_exact_duplicates(self) -> None:
        """A bypassed gate does not run at all, so marking G12 bypassable "for
        near-duplicates only" would have skipped exact ones too. It does not need
        a bypass: near-duplicates already WARN rather than FAIL."""
        assert "G12_CORPUS_DEDUP" in MANDATORY_GATE_IDS


class TestForceCannotBypass:
    """Four mechanisms, asserted independently."""

    @pytest.mark.parametrize("gate_id", sorted(MANDATORY_GATE_IDS))
    def test_a_policy_bypassing_any_mandatory_gate_cannot_be_constructed(
        self, gate_id: str
    ) -> None:
        """Mechanism 2. The illegal object never exists, so no function can be
        handed one."""
        with pytest.raises(ValidationError, match="mandatory"):
            PromotionPolicy(bypass_gate_ids=frozenset({gate_id}))

    @pytest.mark.parametrize(
        "subset",
        [
            subset
            for size in (1, 2, 3)
            for subset in itertools.combinations(sorted(MANDATORY_GATE_IDS), size)
        ][:40],
    )
    def test_no_combination_including_a_mandatory_gate_is_constructible(
        self, subset: tuple[str, ...]
    ) -> None:
        with pytest.raises(ValidationError):
            PromotionPolicy(bypass_gate_ids=frozenset(subset) | BYPASSABLE_GATE_IDS)

    def test_force_maps_to_the_bypassable_constant(self) -> None:
        """Mechanism 3. The caller cannot choose which gates to skip."""
        assert PromotionPolicy.forced().bypass_gate_ids == BYPASSABLE_GATE_IDS

    def test_force_never_covers_a_mandatory_gate(self) -> None:
        assert not (PromotionPolicy.forced().bypass_gate_ids & MANDATORY_GATE_IDS)

    def test_forced_takes_no_gate_argument(self) -> None:
        """The set of safely-bypassable gates is a property of the table, not of
        the person running the command."""
        import inspect

        params = inspect.signature(PromotionPolicy.forced).parameters
        assert "bypass_gate_ids" not in params
        assert "gates" not in params

    def test_a_typo_in_a_gate_id_is_rejected(self) -> None:
        """Otherwise it silently bypasses nothing while looking like it did."""
        with pytest.raises(ValidationError, match="do not exist"):
            PromotionPolicy(bypass_gate_ids=frozenset({"G11_COVERAGE_AXIS"}))

    def test_the_policy_is_frozen(self) -> None:
        policy = PromotionPolicy()
        with pytest.raises(ValidationError):
            policy.bypass_gate_ids = MANDATORY_GATE_IDS

    def test_no_source_file_writes_bypass_gate_ids_from_user_input(self) -> None:
        """Mechanism 3, checked at the source level.

        There must be no argparse flag or config key that reaches
        ``bypass_gate_ids``. The only writer is ``PromotionPolicy.forced``.
        """
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        writers = []
        for path in list((root / "src").rglob("*.py")) + list((root / "scripts").rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "bypass_gate_ids=" in text and path.name not in {"policy.py", "gates.py"}:
                writers.append(str(path.relative_to(root)))
        assert not writers, (
            f"these files set bypass_gate_ids directly: {writers}. Only "
            f"PromotionPolicy.forced() may, and it uses a constant."
        )


class TestRunnerIntegrity:
    def test_every_gate_runs(self) -> None:
        report = run_gates(make_context(), PromotionPolicy())
        assert [r.gate_id for r in report.results] == [s.id for s in GATES]

    def test_there_is_no_short_circuit_on_failure(self) -> None:
        """A candidate with three problems reports all three, so fixing them
        takes one cycle instead of three."""
        broken = make_candidate(make_payload("Email j.doe@somecollege.edu about Initech."))
        report = run_gates(make_context(broken, human=None), PromotionPolicy())
        assert len(report.results) == len(GATES)
        assert len(report.failures) >= 3

    def test_a_missing_mandatory_gate_raises(self, monkeypatch) -> None:
        """Mechanism 4. A gate that does not run is indistinguishable from one
        that passed, which is why absence is checked rather than assumed."""
        import kleos_training_data.promotion.runner as runner

        monkeypatch.setattr(runner, "GATES", GATES[:5])
        with pytest.raises(PromotionIntegrityError, match="did not execute"):
            run_gates(make_context(), PromotionPolicy())

    def test_a_crashing_gate_fails_rather_than_vanishing(self, monkeypatch) -> None:
        """The other thirteen still have something to say about the candidate."""
        import kleos_training_data.promotion.runner as runner
        from kleos_training_data.promotion.gates import GateSpec

        def boom(ctx):
            raise RuntimeError("gate exploded")

        patched = tuple(
            GateSpec(s.id, s.description, boom, s.bypassable, s.privacy)
            if s.id == "G02_SCHEMA_VALID"
            else s
            for s in GATES
        )
        monkeypatch.setattr(runner, "GATES", patched)

        report = run_gates(make_context(), PromotionPolicy())
        schema_result = next(r for r in report.results if r.gate_id == "G02_SCHEMA_VALID")
        assert schema_result.status is GateStatus.FAIL
        assert len(report.results) == len(GATES)

    def test_bypassed_gates_are_recorded_not_hidden(self) -> None:
        report = run_gates(make_context(), PromotionPolicy.forced())
        bypassed = {r.gate_id for r in report.bypassed}
        assert bypassed == BYPASSABLE_GATE_IDS


class TestIndividualGates:
    """Each gate fails on its own target and passes on a clean candidate."""

    def _status(self, ctx: PromotionContext, gate_id: str) -> GateStatus:
        return GATES_BY_ID[gate_id].check(ctx).status

    def test_a_clean_candidate_passes_everything(self) -> None:
        report = run_gates(make_context(), PromotionPolicy())
        assert report.ok(), report.render()

    def test_g01_catches_a_tampered_record(self) -> None:
        candidate = make_candidate()
        tampered = candidate.model_copy(update={"sanitization_status": "edited"})
        assert self._status(make_context(tampered), "G01_STAGING_INTEGRITY") is GateStatus.FAIL

    def test_g02_catches_a_contract_violation(self) -> None:
        payload = make_payload()
        payload["messages"] = [{"role": "user", "content": "only a user turn"}]
        assert self._status(make_context(make_candidate(payload)), "G02_SCHEMA_VALID") is (
            GateStatus.FAIL
        )

    def test_g03_catches_an_id_that_no_longer_matches(self) -> None:
        candidate = make_candidate()
        edited = candidate.model_copy(
            update={"payload": make_payload("A completely different answer.")}
        ).sealed()
        assert self._status(make_context(edited), "G03_ID_INTEGRITY") is GateStatus.FAIL

    def test_g04_catches_a_surviving_secret(self) -> None:
        candidate = make_candidate(make_payload("token ghp_" + "a" * 36))
        assert self._status(make_context(candidate), "G04_SECRET_SCAN") is GateStatus.FAIL

    def test_g05_catches_surviving_pii(self) -> None:
        candidate = make_candidate(make_payload("Email j.doe@somecollege.edu about it."))
        assert self._status(make_context(candidate), "G05_PII_SCAN") is GateStatus.FAIL

    def test_g06_catches_placeholder_residue(self) -> None:
        candidate = make_candidate(make_payload("Ask [[PERSON_1]] about it."))
        assert self._status(make_context(candidate), "G06_SURROGATE_INTEGRITY") is GateStatus.FAIL

    def test_g06_catches_a_surviving_vault_literal(self) -> None:
        """The one failure nothing downstream could catch."""
        vault = EntityVault()
        vault.add("Wolfram Dynamics", "ORG")
        candidate = make_candidate(make_payload("Still at Wolfram Dynamics."))
        ctx = make_context(candidate, vault=vault)
        assert self._status(ctx, "G06_SURROGATE_INTEGRITY") is GateStatus.FAIL

    def test_g07_rejects_a_fact_teaching_example(self) -> None:
        candidate = make_candidate(
            make_payload("Do it because you work at Initech and your advisor agreed.")
        )
        assert self._status(make_context(candidate), "G07_PRIVATE_FACT") is GateStatus.FAIL

    def test_g07_requires_a_human_when_risk_is_ambiguous(self) -> None:
        """A `needs_fact_review` verdict is not something a heuristic may clear."""
        candidate = make_candidate(make_payload("Prioritize Wolfram Dynamics first."))
        ctx = make_context(
            candidate,
            human=make_decision(
                candidate,
                decision="reject",
                gates=GateResults(
                    no_private_data="PASS",
                    policy_not_facts="FAIL",
                    no_unsupported_claims="PASS",
                    schema_and_contract_valid="PASS",
                ),
                reason_codes=["PRIVATE_FACT"],
            ),
        )
        assert self._status(ctx, "G07_PRIVATE_FACT") is GateStatus.FAIL

    def test_g08_requires_a_decision(self) -> None:
        assert self._status(make_context(human=None), "G08_REVIEW_PRESENT") is GateStatus.FAIL

    def test_g08_rejects_a_decision_about_different_content(self) -> None:
        """An approval must not carry over to text nobody read."""
        candidate = make_candidate()
        stale = make_decision(candidate).model_copy(update={"content_hash": "b" * 64})
        assert self._status(make_context(candidate, human=stale), "G08_REVIEW_PRESENT") is (
            GateStatus.FAIL
        )

    def test_g08_rejects_a_broken_signature(self) -> None:
        candidate = make_candidate()
        tampered = make_decision(candidate).model_copy(update={"reviewer_role": "someone_else"})
        assert self._status(make_context(candidate, human=tampered), "G08_REVIEW_PRESENT") is (
            GateStatus.FAIL
        )

    def test_g09_rejects_a_non_approval(self) -> None:
        candidate = make_candidate()
        revise = make_decision(candidate, decision="revise")
        assert self._status(make_context(candidate, human=revise), "G09_REVIEW_APPROVED") is (
            GateStatus.FAIL
        )

    def test_g10_rejects_a_production_capture(self) -> None:
        """The single most consequential gate. A capture from a real deployment
        is one person's private data, whatever the prompt was."""
        candidate = make_candidate(lane=CaptureLane.PRODUCTION_OBSERVATION)
        ctx = make_context(candidate, allowed_sources=frozenset({"real_sanitized"}))
        outcome = GATES_BY_ID["G10_PROVENANCE"].check(ctx)
        assert outcome.status is GateStatus.FAIL
        assert "never be promoted" in outcome.message

    def test_g11_requires_a_domain_axis(self) -> None:
        payload = make_payload()
        payload["variation_axes"] = {"urgency": "high"}
        assert self._status(make_context(make_candidate(payload)), "G11_COVERAGE_AXES") is (
            GateStatus.FAIL
        )

    def test_g11_warns_on_a_pilot_axis(self) -> None:
        payload = make_payload()
        payload["variation_axes"] = {"domain": "career", "novel_axis": "v1"}
        assert self._status(make_context(make_candidate(payload)), "G11_COVERAGE_AXES") is (
            GateStatus.WARN
        )

    def test_g12_fails_on_an_exact_duplicate(self) -> None:
        candidate = make_candidate()
        corpus = CorpusIndex(
            entries=[
                CorpusEntry(
                    example_id="kx-npr-existing00000",
                    text=conversation_text(candidate.payload),
                )
            ]
        )
        assert self._status(make_context(candidate, corpus=corpus), "G12_CORPUS_DEDUP") is (
            GateStatus.FAIL
        )

    def test_g12_only_warns_on_a_near_duplicate(self) -> None:
        """A human weighs whether it is redundancy or a deliberate perturbation.

        One word changed lands at ~0.89 similarity, above the 0.85 threshold.
        Adding a whole sentence on top drops it to ~0.82 and stops being a
        near-duplicate at all — which is the threshold behaving as intended.
        """
        candidate = make_candidate()
        near = conversation_text(candidate.payload).replace("Friday", "Monday")
        corpus = CorpusIndex(entries=[CorpusEntry(example_id="kx-npr-near0000000", text=near)])
        assert self._status(make_context(candidate, corpus=corpus), "G12_CORPUS_DEDUP") is (
            GateStatus.WARN
        )

    def test_g12_passes_when_similarity_falls_below_the_threshold(self) -> None:
        candidate = make_candidate()
        distant = conversation_text(candidate.payload).replace("Friday", "Monday") + (
            " Also noted."
        )
        corpus = CorpusIndex(entries=[CorpusEntry(example_id="kx-npr-far000000000", text=distant)])
        assert self._status(make_context(candidate, corpus=corpus), "G12_CORPUS_DEDUP") is (
            GateStatus.PASS
        )

    def test_g13_fails_on_evaluation_overlap(self) -> None:
        candidate = make_candidate()
        evals = CorpusIndex(
            entries=[CorpusEntry(example_id="eval-0001", text=conversation_text(candidate.payload))]
        )
        assert self._status(make_context(candidate, eval_corpus=evals), "G13_EVAL_LEAKAGE") is (
            GateStatus.FAIL
        )

    def test_g13_catches_prompt_only_overlap(self) -> None:
        """The asymmetry that matters: an eval example's text is prompt-only, so
        a single full-text comparison would miss a matching prompt."""
        candidate = make_candidate()
        prompt_only = conversation_text(candidate.payload, include_assistant=False)
        evals = CorpusIndex(entries=[CorpusEntry(example_id="eval-0001", text=prompt_only)])
        assert self._status(make_context(candidate, eval_corpus=evals), "G13_EVAL_LEAKAGE") is (
            GateStatus.FAIL
        )

    def test_g14_rejects_a_metadata_extra(self) -> None:
        """A stray key would ship inside train.jsonl as a pointer into staging."""
        payload = make_payload()
        payload["metadata"] = {"capture_id": "cap-123"}
        assert self._status(make_context(make_candidate(payload)), "G14_CONTRACT_RENDER") is (
            GateStatus.FAIL
        )

    def test_g14_permits_an_allowlisted_extra(self) -> None:
        payload = make_payload()
        payload["metadata"] = {"ruleset_version": "privacy-rules-v1"}
        assert self._status(make_context(make_candidate(payload)), "G14_CONTRACT_RENDER") is (
            GateStatus.PASS
        )


class TestDeduplication:
    def test_an_exact_duplicate_is_fatal(self) -> None:
        index = CorpusIndex(entries=[CorpusEntry("a", "user: hello\nassistant: hi")])
        findings = index.find_duplicates(CorpusEntry("b", "user: hello\nassistant: hi"))
        assert findings[0].kind is DuplicateKind.EXACT
        assert findings[0].fatal

    def test_changing_only_a_number_is_still_a_duplicate(self) -> None:
        """The digit-collapse rule, restated: a dataset that looks diverse
        because its dates differ is not diverse."""
        index = CorpusIndex(entries=[CorpusEntry("a", "user: due Mar 3\nassistant: ok")])
        findings = index.find_duplicates(CorpusEntry("b", "user: due Mar 7\nassistant: ok"))
        assert findings[0].kind is DuplicateKind.NORMALIZED
        assert findings[0].fatal

    def test_a_near_duplicate_is_not_fatal(self) -> None:
        index = CorpusIndex(entries=[CorpusEntry("a", "user: rank these items by deadline please")])
        findings = index.find_duplicates(
            CorpusEntry("b", "user: rank these items by deadline please now"), threshold=0.5
        )
        assert findings and not findings[0].fatal

    def test_unrelated_text_produces_nothing(self) -> None:
        index = CorpusIndex(entries=[CorpusEntry("a", "user: rank these items")])
        assert index.find_duplicates(CorpusEntry("b", "totally unrelated subject matter")) == []

    def test_an_id_collision_is_reported(self) -> None:
        index = CorpusIndex(entries=[CorpusEntry("same-id", "user: one thing")])
        findings = index.find_duplicates(CorpusEntry("same-id", "user: another thing"))
        assert any(f.kind is DuplicateKind.ID_COLLISION for f in findings)

    def test_adding_to_the_index_takes_effect_immediately(self) -> None:
        """Two identical candidates in one batch must not both promote."""
        index = CorpusIndex(entries=[])
        entry = CorpusEntry("a", "user: hello\nassistant: hi")
        assert index.find_duplicates(entry) == []
        index.add(entry)
        assert index.find_duplicates(CorpusEntry("b", entry.text))


class TestPromotionReport:
    def test_a_clean_report_is_ok(self) -> None:
        assert run_gates(make_context(), PromotionPolicy()).ok()

    def test_warnings_do_not_block_by_default(self) -> None:
        payload = make_payload()
        payload["variation_axes"] = {"domain": "career", "novel_axis": "v1"}
        report = run_gates(make_context(make_candidate(payload)), PromotionPolicy())
        assert report.warnings
        assert report.ok()

    def test_strict_warnings_blocks(self) -> None:
        payload = make_payload()
        payload["variation_axes"] = {"domain": "career", "novel_axis": "v1"}
        report = run_gates(make_context(make_candidate(payload)), PromotionPolicy())
        assert not report.ok(strict_warnings=True)

    def test_privacy_failures_are_distinguished(self) -> None:
        """A privacy failure is an incident, not a quality problem."""
        candidate = make_candidate(make_payload("token ghp_" + "a" * 36))
        report = run_gates(make_context(candidate), PromotionPolicy())
        assert report.privacy_failures
        assert {r.gate_id for r in report.privacy_failures} <= PRIVACY_GATE_IDS

    def test_the_report_serializes(self) -> None:
        import json

        assert json.loads(json.dumps(run_gates(make_context(), PromotionPolicy()).to_dict()))

    def test_the_rendered_report_names_every_gate(self) -> None:
        rendered = run_gates(make_context(), PromotionPolicy()).render()
        for spec in GATES:
            assert spec.id in rendered
