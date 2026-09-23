from __future__ import annotations

import random

import pytest
from tests.contract_cases import VALID_CASES, base

from kleos_training_data.contract.constants import SUPPORTED_TASKS, TASK_CODES
from kleos_training_data.contract.schemas import ID_PATTERN, TrainingExample
from kleos_training_data.ids import (
    DIGEST_ESCALATION,
    CollisionLedger,
    canonicalize,
    content_hash,
    example_id,
)


class TestIdShape:
    @pytest.mark.parametrize("name", sorted(VALID_CASES))
    def test_every_minted_id_satisfies_the_public_contract(self, name: str) -> None:
        minted = example_id(VALID_CASES[name])
        assert ID_PATTERN.match(minted), f"{minted!r} would be rejected by kleos-models"

    @pytest.mark.parametrize("task", SUPPORTED_TASKS)
    def test_the_task_is_readable_from_the_id(self, task: str) -> None:
        assert example_id(base(task=task)).startswith(f"kx-{TASK_CODES[task]}-")

    def test_a_minted_id_is_accepted_by_the_schema(self) -> None:
        payload = base()
        payload["id"] = example_id(payload)
        assert TrainingExample.model_validate(payload).id == payload["id"]

    def test_ids_are_short_enough_to_read(self) -> None:
        assert len(example_id(base())) == 23


class TestContentDerivation:
    def test_identical_content_yields_an_identical_id(self) -> None:
        assert example_id(base()) == example_id(base())

    def test_a_one_character_change_yields_a_different_id(self) -> None:
        original = base()
        edited = base(
            messages=[
                {"role": "user", "content": "Two items are open. Which first?"},
                {"role": "assistant", "content": "Item A: nearest deadline, confirmed evidence!"},
            ]
        )
        assert example_id(original) != example_id(edited)

    @pytest.mark.parametrize(
        "metadata",
        [
            {"created_at": "2026-08-22T00:00:00Z"},
            {"author": "pipeline"},
            {"notes": "reviewed twice"},
            {"quality_status": "reviewed"},
            {"source": "expert_authored"},
            {"scenario_family": "fam-01"},
            {"group_id": "fam-01:0003"},
        ],
    )
    def test_provenance_does_not_change_identity(self, metadata: dict) -> None:
        assert example_id(base()) == example_id(base(metadata=metadata))

    def test_the_supplied_id_does_not_influence_the_derived_one(self) -> None:
        a = example_id(base(id="kx-npr-1111111111111111"))
        b = example_id(base(id="kx-npr-2222222222222222"))
        assert a == b

    def test_the_domain_mirror_does_not_influence_identity(self) -> None:
        assert example_id(base()) == example_id(base(domain="career"))

    def test_variation_axes_are_part_of_identity(self) -> None:
        a = base(variation_axes={"domain": "career", "urgency": "high"})
        b = base(variation_axes={"domain": "career", "urgency": "low"})
        assert example_id(a) != example_id(b)

    def test_the_task_is_part_of_identity(self) -> None:
        assert example_id(base(task="tool_routing")) != example_id(base())


class TestNormalization:
    def test_line_endings_are_normalized(self) -> None:
        crlf = base(
            messages=[
                {"role": "user", "content": "line one\r\nline two"},
                {"role": "assistant", "content": "ok"},
            ]
        )
        lf = base(
            messages=[
                {"role": "user", "content": "line one\nline two"},
                {"role": "assistant", "content": "ok"},
            ]
        )
        assert example_id(crlf) == example_id(lf), (
            "A file saved on Windows would otherwise fork every example in it"
        )

    def test_trailing_whitespace_is_normalized(self) -> None:
        a = base(
            messages=[
                {"role": "user", "content": "line one   \nline two"},
                {"role": "assistant", "content": "ok"},
            ]
        )
        b = base(
            messages=[
                {"role": "user", "content": "line one\nline two"},
                {"role": "assistant", "content": "ok"},
            ]
        )
        assert example_id(a) == example_id(b)

    def test_unicode_composition_is_normalized(self) -> None:
        nfc = "caf\u00e9"
        nfd = "cafe\u0301"
        assert nfc != nfd, "the two forms must genuinely differ or this proves nothing"

        composed = base(
            messages=[
                {"role": "user", "content": nfc},
                {"role": "assistant", "content": "ok"},
            ]
        )
        decomposed = base(
            messages=[
                {"role": "user", "content": nfd},
                {"role": "assistant", "content": "ok"},
            ]
        )
        assert example_id(composed) == example_id(decomposed)

    def test_an_absent_name_and_a_null_name_agree(self) -> None:
        a = base(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}])
        b = base(
            messages=[
                {"role": "user", "content": "x", "name": None},
                {"role": "assistant", "content": "y", "name": None},
            ]
        )
        assert example_id(a) == example_id(b)

    def test_axis_declaration_order_does_not_matter(self) -> None:
        a = base(variation_axes={"domain": "career", "urgency": "high"})
        b = base(variation_axes={"urgency": "high", "domain": "career"})
        assert example_id(a) == example_id(b)


class TestCanonicalization:
    def test_only_identity_bearing_fields_survive(self) -> None:
        projected = canonicalize(base(metadata={"notes": "x"}, domain="career"))
        assert set(projected) == {"task", "messages", "variation_axes"}

    def test_null_axes_are_dropped(self) -> None:
        projected = canonicalize(base(variation_axes={"domain": "career", "urgency": None}))
        assert projected["variation_axes"] == {"domain": "career"}


class TestPositionIndependence:
    def test_shuffling_the_corpus_leaves_the_id_set_identical(self) -> None:
        payloads = [
            base(
                messages=[
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": "a"},
                ]
            )
            for i in range(50)
        ]
        forward = [example_id(p) for p in payloads]

        shuffled = payloads[:]
        random.Random(1712).shuffle(shuffled)
        backward = [example_id(p) for p in shuffled]

        assert sorted(forward) == sorted(backward)
        assert len(set(forward)) == 50, "distinct content must produce distinct ids"

    def test_inserting_an_example_does_not_renumber_the_others(self) -> None:
        first = [
            example_id(
                base(
                    messages=[
                        {"role": "user", "content": f"q{i}"},
                        {"role": "assistant", "content": "a"},
                    ]
                )
            )
            for i in range(5)
        ]
        inserted = example_id(
            base(
                messages=[
                    {"role": "user", "content": "new"},
                    {"role": "assistant", "content": "a"},
                ]
            )
        )
        with_insert = [inserted, *first]
        assert first == with_insert[1:]


class TestCollisionLedger:
    def test_a_new_payload_is_assigned_and_remembered(self) -> None:
        ledger = CollisionLedger()
        minted = ledger.mint(base())
        assert ledger.mint(base()) == minted
        assert ledger.assignments == {content_hash(base()): minted}

    def test_assignment_is_independent_of_insertion_order(self) -> None:
        payloads = [
            base(
                messages=[
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": "a"},
                ]
            )
            for i in range(30)
        ]

        forward = CollisionLedger()
        for payload in payloads:
            forward.mint(payload)

        reverse = CollisionLedger()
        for payload in reversed(payloads):
            reverse.mint(payload)

        assert forward.assignments == reverse.assignments

    def test_a_collision_escalates_to_a_longer_digest(self) -> None:
        ledger = CollisionLedger()
        first = base()
        second = base(
            messages=[
                {"role": "user", "content": "different"},
                {"role": "assistant", "content": "a"},
            ]
        )

        taken = ledger.mint(first)
        ledger.assignments["deliberate-fake-hash"] = example_id(second)

        minted = ledger.mint(second)
        assert minted != taken
        assert minted != example_id(second), "should have escalated past the 16-char form"
        assert len(minted.rsplit("-", 1)[1]) in DIGEST_ESCALATION

    def test_verify_accepts_a_matching_id(self) -> None:
        ledger = CollisionLedger()
        payload = base()
        assert ledger.verify(payload, ledger.mint(payload))

    def test_verify_rejects_an_id_for_different_content(self) -> None:
        ledger = CollisionLedger()
        minted = ledger.mint(base())
        edited = base(
            messages=[
                {"role": "user", "content": "edited after approval"},
                {"role": "assistant", "content": "a"},
            ]
        )
        assert not ledger.verify(edited, minted)

    def test_a_ledger_round_trips_through_disk(self, tmp_path) -> None:
        ledger = CollisionLedger(path=tmp_path / "ledger.json")
        minted = ledger.mint(base())
        ledger.save()

        reloaded = CollisionLedger.load(tmp_path / "ledger.json")
        assert reloaded.mint(base()) == minted

    def test_loading_a_missing_ledger_yields_an_empty_one(self, tmp_path) -> None:
        assert CollisionLedger.load(tmp_path / "absent.json").assignments == {}

    def test_the_ledger_stores_no_content(self, tmp_path) -> None:
        ledger = CollisionLedger(path=tmp_path / "ledger.json")
        ledger.mint(
            base(
                messages=[
                    {"role": "user", "content": "SECRETPHRASE"},
                    {"role": "assistant", "content": "a"},
                ]
            )
        )
        assert "SECRETPHRASE" not in ledger.save().read_text(encoding="utf-8")
