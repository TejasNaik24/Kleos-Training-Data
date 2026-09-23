from __future__ import annotations

import json
from typing import ClassVar

import pytest

from kleos_training_data.errors import VaultError
from kleos_training_data.privacy.detect import (
    Detection,
    redacted_excerpt,
    scan_text,
)
from kleos_training_data.privacy.entities import VAULT_SLOTS, EntityVault
from kleos_training_data.privacy.facts import assess
from kleos_training_data.privacy.redaction import (
    PlaceholderMap,
    has_placeholder_residue,
    redact,
    rehydrate,
)
from kleos_training_data.privacy.rules import (
    ALL_RULES,
    PII_RULES,
    SECRET_RULES,
    is_allowlisted,
)
from kleos_training_data.privacy.sanitize import (
    STATUS_BLOCKED,
    STATUS_CLEAN,
    known_surrogate_names,
    sanitize,
    verify_sanitized,
)
from kleos_training_data.scenarios.generator import generate
from kleos_training_data.scenarios.loader import load_catalog
from kleos_training_data.scenarios.surrogates import SurrogatePool, load_pools


def payload(user: str, assistant: str = "Start with the nearest deadline.", **axes) -> dict:
    return {
        "task": "notification_prioritization",
        "messages": [
            {"role": "system", "content": "Rank the items."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "variation_axes": {"domain": "career", **axes},
    }


class TestSecretDetection:
    SAMPLES: ClassVar[list[tuple[str, str]]] = [
        ("aws_access_key", "AKIAIOSFODNN7EXAMPLE"),
        ("openai_key", "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD"),
        ("anthropic_key", "sk-ant-api03-notarealkey_abcdefghijklmnop"),
        ("hf_token", "hf_" + "a" * 34),
        ("github_token", "ghp_" + "a" * 36),
        ("slack_token", "xoxb-1111111111-notarealtoken"),
        ("google_api_key", "AIzaSy" + "A" * 33),
        ("jwt", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijk"),
        ("private_key_block", "-----BEGIN RSA PRIVATE KEY-----"),
        ("supabase_url", "https://abcdefghijklmnopqrst.supabase.co"),
        ("bearer_token", "Bearer abcdefghijklmnopqrstuvwxyz012345"),
        ("session_cookie", "sb-abcdefghij-auth-token=abcdefghijklmnopqrst"),
        ("database_url_with_password", "postgresql+asyncpg://postgres:hunter2xyz@db.host:5432/x"),
    ]

    @pytest.mark.parametrize(("kind", "sample"), SAMPLES, ids=[k for k, _ in SAMPLES])
    def test_each_secret_fires(self, kind: str, sample: str) -> None:
        kinds = {d.kind for d in scan_text(sample, field_path="t")}
        assert kind in kinds, f"{kind} did not fire on its own sample"

    @pytest.mark.parametrize(("kind", "sample"), SAMPLES, ids=[k for k, _ in SAMPLES])
    def test_each_secret_blocks(self, kind: str, sample: str) -> None:
        result = sanitize(payload(f"here it is: {sample}"), scenario_family="f")
        assert result.status == STATUS_BLOCKED

    def test_a_secret_is_never_redacted(self) -> None:
        original = payload("token ghp_" + "a" * 36)
        result = sanitize(original, scenario_family="f")
        assert result.replacements == 0
        assert result.payload == original

    def test_the_library_rules_cover_the_scanner_rules(self) -> None:
        import importlib.util
        import sys
        from pathlib import Path

        scanner_path = (
            Path(__file__).resolve().parent.parent / "scripts" / "check_no_private_data.py"
        )
        spec = importlib.util.spec_from_file_location("_scanner_for_test", scanner_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        scanner_error_kinds = {
            name for name, _pattern, severity in module.PATTERNS if severity == "error"
        }
        library_kinds = {rule.kind for rule in SECRET_RULES}
        missing = scanner_error_kinds - library_kinds
        assert not missing, (
            f"The pre-commit scanner blocks on {sorted(missing)} but the library "
            f"does not detect them. The library must never be the looser one."
        )


class TestPIIDetection:
    CASES: ClassVar[list[tuple[str, str]]] = [
        ("email", "reach me at j.doe@somecollege.edu"),
        ("phone", "call 614-555-9876 today"),
        ("ssn_like", "ssn 123-45-6789 on file"),
        ("home_path", "the log is at /Users/jdoe/logs/app.log"),
        ("url_with_token", "open https://app.example.org/x?access_token=abc123def456"),
        ("street_address", "mail to 1234 Maple Grove Avenue please"),
        ("student_id", "student id 8812345 was flagged"),
        ("social_handle", "ping @jdoe_real about it"),
        ("uuid", "user 9d0e6a2c-1f4b-4a55-9a8e-0b3f1c77e2d1 exists"),
        ("absolute_datetime", "it is due 2026-03-14"),
    ]

    @pytest.mark.parametrize(("kind", "text"), CASES, ids=[k for k, _ in CASES])
    def test_each_pii_rule_fires(self, kind: str, text: str) -> None:
        assert kind in {d.kind for d in scan_text(text, field_path="t")}

    @pytest.mark.parametrize(("kind", "text"), CASES, ids=[k for k, _ in CASES])
    def test_each_pii_rule_is_redacted_not_blocked(self, kind: str, text: str) -> None:
        result = sanitize(payload(text), scenario_family="f")
        assert result.status != STATUS_BLOCKED
        assert result.replacements >= 1

    def test_every_pii_rule_has_a_slot(self) -> None:
        for rule in PII_RULES:
            assert rule.slot, f"{rule.rule_id} is redact-severity but has no slot"


class TestAllowlists:
    @pytest.mark.parametrize(
        "address",
        ["noreply@example.com", "alice@example.invalid", "your-email@example.org"],
    )
    def test_placeholder_emails_are_allowed(self, address: str) -> None:
        assert is_allowlisted("email", address)

    def test_a_plausible_address_is_not_allowed(self) -> None:
        assert not is_allowlisted("email", "j.doe@somecollege.edu")

    def test_the_nil_uuid_is_allowed(self) -> None:
        assert is_allowlisted("uuid", "00000000-0000-0000-0000-000000000000")


class TestDetectionsCarryNoValue:
    @pytest.mark.parametrize(
        "secret",
        [
            "AKIAIOSFODNN7EXAMPLE",
            "ghp_" + "b" * 36,
            "j.doe@somecollege.edu",
            "614-555-9876",
        ],
    )
    def test_the_serialized_detection_omits_the_match(self, secret: str) -> None:
        detections = scan_text(f"value is {secret} here", field_path="t")
        assert detections
        serialized = json.dumps([d.to_dict() for d in detections])
        assert secret not in serialized

    def test_no_long_substring_of_the_match_survives(self) -> None:
        secret = "ghp_" + "cdefghij" * 5
        detections = scan_text(f"token {secret}", field_path="t")
        serialized = json.dumps([d.to_dict() for d in detections])
        for size in range(8, len(secret)):
            for start in range(len(secret) - size + 1):
                assert secret[start : start + size] not in serialized

    def test_an_excerpt_masks_its_neighbours(self) -> None:
        text = "email j.doe@somecollege.edu then call 614-555-9876 about it"
        detections = scan_text(text, field_path="t")
        assert len(detections) >= 2
        for detection in detections:
            assert "j.doe@somecollege.edu" not in detection.excerpt
            assert "614-555-9876" not in detection.excerpt

    def test_an_excerpt_still_locates_the_hit(self) -> None:
        text = "the deadline is 2026-03-14 for the review"
        excerpt = scan_text(text, field_path="t")[0].excerpt
        assert "deadline" in excerpt or "review" in excerpt

    def test_redacted_excerpt_masks_arbitrary_spans(self) -> None:
        text = "alpha SECRET1 beta SECRET2 gamma"
        excerpt = redacted_excerpt(text, 6, 13, mask_spans=((19, 26),))
        assert "SECRET1" not in excerpt
        assert "SECRET2" not in excerpt


class TestOverlapResolution:
    def test_the_most_severe_detection_wins(self) -> None:
        detections = scan_text('token = "Bearer ' + "a" * 30 + '"', field_path="t")
        spans = [(d.start, d.end) for d in detections]
        for i, (s1, e1) in enumerate(spans):
            for s2, e2 in spans[i + 1 :]:
                assert not (s1 < e2 and s2 < e1), "overlapping detections survived"

    def test_a_vault_entry_inside_an_email_does_not_double_replace(self) -> None:
        vault = EntityVault()
        vault.add("realcompany", "ORG")
        result = sanitize(
            payload("write to dana@realcompany.com about it"),
            scenario_family="f",
            vault=vault,
        )
        content = result.payload["messages"][1]["content"]
        assert "realcompany" not in content
        assert content.endswith("about it"), f"text was corrupted: {content!r}"


class TestVault:
    def test_a_literal_is_found_and_replaced(self) -> None:
        vault = EntityVault()
        vault.add("Wolfram Dynamics", "ORG")
        result = sanitize(payload("I work at Wolfram Dynamics"), scenario_family="f", vault=vault)
        assert "Wolfram Dynamics" not in result.payload["messages"][1]["content"]

    def test_the_longest_literal_matches_first(self) -> None:
        vault = EntityVault()
        vault.add("Northgate", "ORG")
        vault.add("Northgate Research Group", "ORG")
        detections = vault.scan("at Northgate Research Group today", field_path="t")
        assert detections[0].matched_len == len("Northgate Research Group")

    def test_an_unknown_slot_is_rejected(self) -> None:
        with pytest.raises(VaultError, match="Unknown vault slot"):
            EntityVault().add("x", "NICKNAME")

    def test_an_empty_entry_is_rejected(self) -> None:
        with pytest.raises(VaultError, match="cannot be empty"):
            EntityVault().add("   ", "ORG")

    def test_the_vault_round_trips_with_owner_only_permissions(self, tmp_path) -> None:
        import stat

        vault = EntityVault()
        vault.add("Wolfram Dynamics", "ORG")
        path = vault.save(tmp_path / "entity_vault.json")

        assert not stat.S_IMODE(path.stat().st_mode) & 0o077
        assert EntityVault.load(path).entries == vault.entries

    def test_an_absent_vault_loads_empty(self, tmp_path) -> None:
        assert len(EntityVault.load(tmp_path / "absent.json")) == 0

    def test_a_corrupt_vault_refuses_to_guess(self, tmp_path) -> None:
        path = tmp_path / "entity_vault.json"
        path.write_text("{ broken", encoding="utf-8")
        with pytest.raises(VaultError, match="not valid JSON"):
            EntityVault.load(path)

    def test_contains_any_detects_a_survivor(self) -> None:
        vault = EntityVault()
        vault.add("Wolfram Dynamics", "ORG")
        assert vault.contains_any("still at Wolfram Dynamics")
        assert not vault.contains_any("still at Northwind")

    def test_every_slot_is_registered(self) -> None:
        for slot in VAULT_SLOTS:
            EntityVault().add("x", slot)


class TestSurrogates:
    def test_the_same_value_maps_to_one_placeholder(self) -> None:
        mapping = PlaceholderMap()
        detections = scan_text(
            "email j.doe@somecollege.edu and j.doe@somecollege.edu", field_path="t"
        )
        redacted = redact(
            "email j.doe@somecollege.edu and j.doe@somecollege.edu", detections, mapping
        )
        assert redacted.count("[[EMAIL_1]]") == 2

    def test_surrogates_are_stable_within_a_family(self) -> None:
        pools = load_pools()
        first = sanitize(payload("ask j.doe@somecollege.edu"), scenario_family="fam.a", pools=pools)
        second = sanitize(
            payload("ask j.doe@somecollege.edu"), scenario_family="fam.a", pools=pools
        )
        assert first.payload == second.payload

    def test_surrogates_differ_across_families(self) -> None:
        pools = load_pools()
        a = sanitize(payload("ask j.doe@somecollege.edu"), scenario_family="fam.a", pools=pools)
        b = sanitize(payload("ask j.doe@somecollege.edu"), scenario_family="fam.b", pools=pools)
        assert a.payload["messages"][1]["content"] != b.payload["messages"][1]["content"]

    def test_a_structured_slot_keeps_its_shape(self) -> None:
        result = sanitize(payload("write to j.doe@somecollege.edu"), scenario_family="f")
        content = result.payload["messages"][1]["content"]
        assert "@example.invalid" in content

    def test_no_placeholder_survives_rehydration(self) -> None:
        result = sanitize(
            payload("email j.doe@somecollege.edu, call 614-555-9876, due 2026-03-14"),
            scenario_family="f",
        )
        for message in result.payload["messages"]:
            assert not has_placeholder_residue(message["content"])

    def test_placeholders_avoid_the_public_validators_markers(self) -> None:
        mapping = PlaceholderMap()
        placeholder = mapping.placeholder_for("PERSON", "abc12345")
        for marker in ("TODO", "FIXME", "lorem ipsum", "..."):
            assert marker.lower() not in placeholder.lower()

    def test_rehydration_handles_double_digit_placeholders(self) -> None:
        mapping = PlaceholderMap()
        for index in range(12):
            mapping.placeholder_for("PERSON", f"digest{index:02d}")
        text = " ".join(mapping.placeholders)
        result = rehydrate(text, mapping, scenario_family="f", pools=load_pools())
        assert not has_placeholder_residue(result)


class TestPrivateFacts:
    @pytest.mark.parametrize(
        ("rule_id", "text"),
        [
            ("fact.attributive_justification", "Do it because you work at Initech."),
            ("fact.document_recall", "Your resume says you led that team."),
            ("fact.session_recall", "As we discussed last Tuesday, ship it."),
            ("fact.biographical_possessive", "Your advisor already approved this."),
        ],
    )
    def test_each_fact_rule_fires(self, rule_id: str, text: str) -> None:
        assessment = assess(payload("What next?", assistant=text))
        assert rule_id in {s.rule_id for s in assessment.signals}

    def test_an_unsupported_entity_is_the_strongest_signal(self) -> None:
        assessment = assess(
            payload("Rank these two items.", assistant="Prioritize Wolfram Dynamics first.")
        )
        assert "fact.unsupported_entity" in {s.rule_id for s in assessment.signals}
        assert assessment.requires_human

    def test_a_supported_entity_is_not_flagged(self) -> None:
        assessment = assess(payload("Northwind is due Friday.", assistant="Start with Northwind."))
        assert "fact.unsupported_entity" not in {s.rule_id for s in assessment.signals}

    def test_a_sentence_initial_entity_in_the_prompt_still_counts(self) -> None:
        assessment = assess(
            payload("Silverbrook is due in 5 days.", assistant="Then Silverbrook, then wait.")
        )
        assert "fact.unsupported_entity" not in {s.rule_id for s in assessment.signals}

    def test_a_policy_style_answer_is_clean(self) -> None:
        assessment = assess(
            payload(
                "Two items, one due sooner but unverified.",
                assistant="Take the confirmed one: acting on unverified work risks "
                "doing the wrong thing entirely.",
            )
        )
        assert assessment.verdict == "policy_like"
        assert not assessment.requires_human

    def test_a_clear_fact_teacher_is_graded_as_such(self) -> None:
        assessment = assess(
            payload(
                "What first?",
                assistant="Prioritize it because you work at Initech and your "
                "advisor confirmed the date.",
            )
        )
        assert assessment.verdict == "fact_teaching"

    def test_the_assessment_carries_no_matched_text(self) -> None:
        assessment = assess(payload("What next?", assistant="Your advisor at Initech agreed."))
        assert "Initech" not in json.dumps(assessment.to_dict())


class TestNoFalsePositivesOnOurOwnCorpus:
    def test_the_generated_corpus_sanitizes_clean(self) -> None:
        pools = load_pools()
        for scenario in load_catalog():
            pool = SurrogatePool.load(scenario.entities.pool)
            for candidate in generate(scenario, pool):
                result = sanitize(
                    candidate.to_payload(),
                    scenario_family=scenario.family,
                    pools=pools,
                )
                assert result.status == STATUS_CLEAN, (
                    f"{scenario.family} produced {result.status} with "
                    f"{[d.kind for d in result.detections]}"
                )

    def test_the_generated_corpus_has_no_fact_risk(self) -> None:
        for scenario in load_catalog():
            pool = SurrogatePool.load(scenario.entities.pool)
            for candidate in generate(scenario, pool):
                assessment = assess(candidate.to_payload())
                assert assessment.verdict == "policy_like", (
                    f"{scenario.family}: {[s.rule_id for s in assessment.signals]}"
                )

    def test_our_own_surrogate_names_are_not_flagged(self) -> None:
        known = known_surrogate_names(load_pools())
        detections = scan_text("Blue Harbor is due Friday", field_path="t", known_names=known)
        assert not [d for d in detections if d.kind == "person_name"]

    def test_a_real_looking_name_is_still_flagged(self) -> None:
        known = known_surrogate_names(load_pools())
        detections = scan_text("Marcus Holloway is due Friday", field_path="t", known_names=known)
        assert [d for d in detections if d.kind == "person_name"]

    def test_a_secret_inside_a_known_name_still_fires(self) -> None:
        known = known_surrogate_names(load_pools())
        detections = scan_text(
            "Northwind uses AKIAIOSFODNN7EXAMPLE", field_path="t", known_names=known
        )
        assert "aws_access_key" in {d.kind for d in detections}

    def test_a_bigram_does_not_span_a_sentence_boundary(self) -> None:
        detections = scan_text("Do that first. Then Silverbrook follows.", field_path="t")
        assert not [d for d in detections if d.kind == "person_name"]


class TestVerifySanitized:
    def test_a_clean_payload_has_no_problems(self) -> None:
        assert verify_sanitized(payload("Northwind is due Friday.")) == []

    def test_a_surviving_secret_is_reported(self) -> None:
        assert verify_sanitized(payload("token ghp_" + "a" * 36))

    def test_placeholder_residue_is_reported(self) -> None:
        assert verify_sanitized(payload("ask [[PERSON_1]] about it"))

    def test_a_surviving_vault_literal_is_reported(self) -> None:
        vault = EntityVault()
        vault.add("Wolfram Dynamics", "ORG")
        assert verify_sanitized(payload("at Wolfram Dynamics"), vault=vault)


class TestSanitizationResult:
    def test_hashes_change_when_content_does(self) -> None:
        result = sanitize(payload("email j.doe@somecollege.edu"), scenario_family="f")
        assert result.input_hash != result.output_hash

    def test_hashes_match_when_nothing_changes(self) -> None:
        result = sanitize(payload("Northwind is due Friday."), scenario_family="f")
        assert result.input_hash == result.output_hash

    def test_sanitization_is_idempotent(self) -> None:
        pools = load_pools()
        once = sanitize(payload("email j.doe@somecollege.edu"), scenario_family="f", pools=pools)
        twice = sanitize(once.payload, scenario_family="f", pools=pools)
        assert twice.payload == once.payload

    def test_the_sidecar_serializes(self) -> None:
        result = sanitize(payload("call 614-555-9876"), scenario_family="f")
        assert json.loads(json.dumps(result.to_dict()))["status"]

    def test_axes_are_scanned_too(self) -> None:
        result = sanitize(
            payload("nothing here", workspace="j.doe@somecollege.edu"), scenario_family="f"
        )
        assert any(d.field_path.startswith("variation_axes") for d in result.detections)


class TestRuleRegistry:
    def test_every_rule_id_is_unique(self) -> None:
        ids = [rule.rule_id for rule in ALL_RULES]
        assert len(set(ids)) == len(ids)

    def test_every_rule_id_is_versioned(self) -> None:
        for rule in ALL_RULES:
            assert rule.rule_id.endswith((".v1", ".v2", ".v3")), rule.rule_id

    def test_secret_rules_are_never_redactable(self) -> None:
        for rule in SECRET_RULES:
            detection = Detection(
                rule_id=rule.rule_id,
                kind=rule.kind,
                layer=rule.layer,
                severity=rule.severity,
                field_path="t",
                start=0,
                end=1,
                matched_len=1,
                matched_sha256_8="0" * 8,
                excerpt="",
                slot=rule.slot,
            )
            assert not detection.redactable, f"{rule.rule_id} would be auto-redacted"
