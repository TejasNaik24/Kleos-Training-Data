from __future__ import annotations

import json
import stat

import pytest

from kleos_training_data.contract.schemas import DatasetManifest, TrainingExample
from kleos_training_data.contract.splitting import SplitConfig, split_examples, verify_split
from kleos_training_data.datasets.holdouts import HoldoutPlan, resolve_holdouts
from kleos_training_data.datasets.manifest import build_manifest, build_provenance
from kleos_training_data.datasets.release import ReleaseWriter, open_for_rewrite
from kleos_training_data.datasets.verify import verify_release
from kleos_training_data.errors import (
    ConfigError,
    ContractViolationError,
    ReleaseImmutabilityError,
)
from kleos_training_data.paths import Workspace
from kleos_training_data.scenarios.models import Scenario


def make_example(index: int, **axes) -> TrainingExample:
    payload = {
        "id": f"kx-npr-{index:016d}",
        "task": "notification_prioritization",
        "messages": [
            {"role": "user", "content": f"Rank the items for situation {index}."},
            {"role": "assistant", "content": f"Start with the nearest deadline ({index})."},
        ],
        "variation_axes": {"domain": "career", **axes},
        "metadata": {
            "source": "synthetic",
            "quality_status": "reviewed",
            "scenario_family": f"fam-{index % 3}",
            "group_id": f"grp-{index // 4:04d}",
        },
    }
    return TrainingExample.model_validate(payload)


def corpus(count: int = 24, **axes) -> list[TrainingExample]:
    formats = ["bullets", "prose", "json"]
    domains = ["career", "research", "coursework"]
    return [
        make_example(
            i,
            format=formats[i % 3],
            domain=domains[i % 3],
            entities=f"set_{'abc'[i % 3]}",
            **axes,
        )
        for i in range(count)
    ]


def scenario_with_holdout(**holdout) -> Scenario:
    return Scenario.model_validate(
        {
            "family": "notif.test",
            "task": "notification_prioritization",
            "policy_claim": "Rank by expected cost of delay, always.",
            "anti_claim": "Do not rank by position in the list.",
            "axes": {"domain": ["career", "research", "coursework"]},
            "entities": {"pool": "generic_pool_a"},
            "expected": {"policy": "rank_by_deadline_then_evidence"},
            "holdout": holdout,
        }
    )


class TestSplitDeterminism:
    def test_the_same_input_produces_the_same_split(self) -> None:
        examples = corpus()
        config = SplitConfig(strategy="group", seed=42)
        first = split_examples(examples, config)
        second = split_examples(examples, config)
        assert [e.id for e in first.train] == [e.id for e in second.train]
        assert [e.id for e in first.test] == [e.id for e in second.test]

    def test_adding_examples_does_not_reshuffle_the_existing_ones(self) -> None:
        config = SplitConfig(strategy="group", seed=42)
        small = split_examples(corpus(12), config)
        large = split_examples(corpus(24), config)

        def placement(result):
            return {
                e.id: name for name in ("train", "validation", "test") for e in result.split(name)
            }

        before, after = placement(small), placement(large)
        moved = {i for i in before if after.get(i) != before[i]}
        assert not moved, f"{len(moved)} example(s) changed split when the corpus grew"

    def test_a_different_seed_produces_a_different_split(self) -> None:
        examples = corpus()
        a = split_examples(examples, SplitConfig(strategy="group", seed=1))
        b = split_examples(examples, SplitConfig(strategy="group", seed=2))
        assert [e.id for e in a.train] != [e.id for e in b.train]

    def test_input_order_does_not_matter(self) -> None:
        examples = corpus()
        config = SplitConfig(strategy="group", seed=42)
        forward = split_examples(examples, config)
        backward = split_examples(list(reversed(examples)), config)
        assert [e.id for e in forward.train] == [e.id for e in backward.train]


class TestGroupIsolation:
    def test_the_fixture_actually_has_multi_member_groups(self) -> None:
        from collections import Counter

        sizes = Counter(e.group_key() for e in corpus(30))
        assert max(sizes.values()) >= 4
        assert len(sizes) < 30

    def test_a_group_never_straddles_a_boundary(self) -> None:
        result = split_examples(corpus(30), SplitConfig(strategy="group", seed=7))
        placement: dict[str, str] = {}
        for name in ("train", "validation", "test"):
            for example in result.split(name):
                group = example.group_key()
                assert placement.setdefault(group, name) == name

    def test_verify_split_catches_a_straddled_group(self) -> None:
        result = split_examples(corpus(30), SplitConfig(strategy="group", seed=7))
        moved = result.train.pop()
        result.test.append(moved)
        with pytest.raises(ContractViolationError, match="split across"):
            verify_split(result)

    def test_verify_split_catches_overlap(self) -> None:
        result = split_examples(corpus(), SplitConfig(strategy="group", seed=7))
        result.test.append(result.train[0])
        with pytest.raises(ContractViolationError, match="overlap"):
            verify_split(result)

    def test_verify_split_catches_loss(self) -> None:
        result = split_examples(corpus(), SplitConfig(strategy="group", seed=7))
        result.train.pop()
        with pytest.raises(ContractViolationError, match="lost or duplicated"):
            verify_split(result, expected_total=24)

    def test_an_empty_training_set_is_an_error(self) -> None:
        result = split_examples(corpus(), SplitConfig(strategy="group", seed=7))
        result.test.extend(result.train)
        result.train.clear()
        with pytest.raises(ContractViolationError, match="empty training set"):
            verify_split(result)


class TestHoldoutSplits:
    def test_held_out_values_land_only_in_test(self) -> None:
        result = split_examples(
            corpus(),
            SplitConfig(strategy="format_holdout", seed=42, holdout_values=["json"]),
        )
        assert {e.variation_axes.format for e in result.test} == {"json"}
        assert "json" not in {e.variation_axes.format for e in result.train}

    def test_validation_is_drawn_from_seen_values(self) -> None:
        result = split_examples(
            corpus(60),
            SplitConfig(
                strategy="format_holdout",
                seed=42,
                holdout_values=["json"],
                train_fraction=0.7,
                validation_fraction=0.15,
                test_fraction=0.15,
            ),
        )
        assert "json" not in {e.variation_axes.format for e in result.validation}

    def test_an_unknown_holdout_value_is_rejected(self) -> None:
        with pytest.raises(ContractViolationError, match="do not occur"):
            split_examples(
                corpus(),
                SplitConfig(strategy="format_holdout", holdout_values=["markdown"]),
            )

    def test_holding_out_everything_is_rejected(self) -> None:
        with pytest.raises(ContractViolationError, match="no training data"):
            split_examples(
                corpus(),
                SplitConfig(
                    strategy="format_holdout",
                    holdout_values=["bullets", "prose", "json"],
                ),
            )

    def test_a_single_valued_attribute_cannot_be_held_out(self) -> None:
        examples = [make_example(i, format="bullets") for i in range(9)]
        with pytest.raises(ContractViolationError, match="only 1 distinct"):
            split_examples(examples, SplitConfig(strategy="format_holdout"))


class TestHoldoutResolution:
    def test_a_declared_reservation_becomes_an_explicit_plan(self) -> None:
        plan = resolve_holdouts(corpus(), [scenario_with_holdout(reserve_formats=["json"])])
        assert plan.attribute == "format"
        assert plan.values == ["json"]
        assert plan.strategy == "format_holdout"

    def test_an_uncovered_reservation_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="no examples"):
            resolve_holdouts(corpus(), [scenario_with_holdout(reserve_formats=["markdown"])])

    def test_an_uncovered_reservation_can_be_allowed_explicitly(self) -> None:
        plan = resolve_holdouts(
            corpus(),
            [scenario_with_holdout(reserve_formats=["markdown"])],
            require_coverage=False,
        )
        assert plan.values == []

    def test_two_declared_attributes_are_ambiguous(self) -> None:
        scenario = scenario_with_holdout(reserve_formats=["json"], reserve_domains=["research"])
        with pytest.raises(ConfigError, match="declares holdouts on"):
            resolve_holdouts(corpus(), [scenario])

    def test_an_ambiguous_catalog_can_be_disambiguated(self) -> None:
        scenario = scenario_with_holdout(reserve_formats=["json"], reserve_domains=["research"])
        plan = resolve_holdouts(corpus(), [scenario], attribute="format")
        assert plan.values == ["json"]

    def test_reserving_everything_is_rejected(self) -> None:
        scenario = scenario_with_holdout(reserve_formats=["bullets", "prose", "json"])
        with pytest.raises(ConfigError, match="every format value"):
            resolve_holdouts(corpus(), [scenario])

    def test_no_declaration_yields_an_empty_plan(self) -> None:
        plan = resolve_holdouts(corpus(), [scenario_with_holdout()])
        assert not plan.declared
        assert plan.strategy is None

    def test_an_unregistered_shift_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown ood_shift"):
            scenario_with_holdout(reserve_formats=["json"], ood_shift="vibe_shift")

    def test_the_plan_records_who_declared_what(self) -> None:
        plan = resolve_holdouts(corpus(), [scenario_with_holdout(reserve_formats=["json"])])
        assert plan.declared_by["json"] == ["notif.test"]


class TestManifest:
    def _sealed(self, tmp_path, **kwargs):
        workspace = Workspace.from_env(tmp_path)
        for zone in workspace.all_zones():
            zone.mkdir(parents=True, exist_ok=True)
        split = split_examples(corpus(), SplitConfig(strategy="group", seed=42))
        writer = ReleaseWriter(workspace)
        return workspace, writer.seal(
            version=kwargs.pop("version", "kleos-policy-v0.1.0"),
            split=split,
            provenance_builder=lambda m: build_provenance(
                version="kleos-policy-v0.1.0",
                split=split,
                holdout=HoldoutPlan(),
                manifest=m,
                examples=corpus(),
                scenario_fingerprints={},
            ),
            **kwargs,
        )

    def test_the_manifest_holds_only_public_fields(self, tmp_path) -> None:
        _, sealed = self._sealed(tmp_path)
        raw = json.loads((sealed.path / "manifest.json").read_text(encoding="utf-8"))
        assert set(raw) == set(DatasetManifest.model_fields)

    def test_distributions_are_computed_from_content(self, tmp_path) -> None:
        _, sealed = self._sealed(tmp_path)
        assert sealed.manifest.task_distribution == {"notification_prioritization": 24}
        assert sealed.manifest.quality_distribution == {"reviewed": 24}

    def test_file_hashes_cover_only_the_shipped_splits(self, tmp_path) -> None:
        _, sealed = self._sealed(tmp_path)
        assert set(sealed.manifest.file_hashes) <= {
            "train.jsonl",
            "validation.jsonl",
            "test.jsonl",
        }
        assert "provenance.json" not in sealed.manifest.file_hashes

    def test_the_content_hash_is_finalized(self, tmp_path) -> None:
        _, sealed = self._sealed(tmp_path)
        assert sealed.manifest.content_hash == sealed.manifest.compute_content_hash()

    def test_contains_private_data_defaults_to_false(self, tmp_path) -> None:
        _, sealed = self._sealed(tmp_path)
        assert sealed.manifest.contains_private_data is False

    def test_declare_private_sets_the_flag(self, tmp_path) -> None:
        _, sealed = self._sealed(tmp_path, declare_private=True)
        assert sealed.manifest.contains_private_data is True

    def test_provenance_is_a_sibling_not_a_manifest_field(self, tmp_path) -> None:
        _, sealed = self._sealed(tmp_path)
        assert (sealed.path / "provenance.json").is_file()
        assert "provenance" not in json.loads(
            (sealed.path / "manifest.json").read_text(encoding="utf-8")
        )

    def test_provenance_records_the_pinned_contract(self, tmp_path) -> None:
        _, sealed = self._sealed(tmp_path)
        provenance = json.loads((sealed.path / "provenance.json").read_text(encoding="utf-8"))
        assert provenance["contract"]["pinned_commit"]
        assert provenance["versions"]["privacy_ruleset"]


class TestReleaseImmutability:
    def _workspace(self, tmp_path) -> Workspace:
        workspace = Workspace.from_env(tmp_path)
        for zone in workspace.all_zones():
            zone.mkdir(parents=True, exist_ok=True)
        return workspace

    def _seal(self, workspace, version="kleos-policy-v0.1.0", examples=None):
        split = split_examples(examples or corpus(), SplitConfig(strategy="group", seed=42))
        return ReleaseWriter(workspace).seal(
            version=version,
            split=split,
            provenance_builder=lambda m: build_provenance(
                version=version,
                split=split,
                holdout=HoldoutPlan(),
                manifest=m,
                examples=examples or corpus(),
                scenario_fingerprints={},
            ),
        )

    def test_sealing_an_existing_version_is_refused(self, tmp_path) -> None:
        workspace = self._workspace(tmp_path)
        self._seal(workspace)
        with pytest.raises(ReleaseImmutabilityError, match="already exists"):
            self._seal(workspace)

    def test_there_is_no_force_parameter(self) -> None:
        import inspect

        params = inspect.signature(ReleaseWriter.seal).parameters
        assert "force" not in params
        assert "overwrite" not in params

    def test_a_sealed_release_is_read_only(self, tmp_path) -> None:
        sealed = self._seal(self._workspace(tmp_path))
        for path in sealed.path.iterdir():
            assert not stat.S_IMODE(path.stat().st_mode) & 0o222

    def test_a_release_lock_is_written(self, tmp_path) -> None:
        sealed = self._seal(self._workspace(tmp_path))
        lock = json.loads((sealed.path / "RELEASE.lock").read_text(encoding="utf-8"))
        assert lock["content_hash"] == sealed.content_hash
        assert "provenance.json" in lock["all_file_hashes"]

    def test_a_failed_seal_leaves_nothing_behind(self, tmp_path, monkeypatch) -> None:
        workspace = self._workspace(tmp_path)

        import kleos_training_data.datasets.release as release_module

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(release_module.os, "replace", boom)
        with pytest.raises(OSError):
            self._seal(workspace)

        assert not workspace.release("kleos-policy-v0.1.0").exists()
        assert not list(workspace.releases.glob(".staging-*"))

    def test_only_canonical_filenames_are_written(self, tmp_path) -> None:
        sealed = self._seal(self._workspace(tmp_path))
        jsonl = {p.name for p in sealed.path.glob("*.jsonl")}
        assert jsonl <= {"train.jsonl", "validation.jsonl", "test.jsonl"}

    def test_an_empty_split_is_not_written(self, tmp_path) -> None:
        sealed = self._seal(self._workspace(tmp_path))
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
            path = sealed.path / name
            if path.is_file():
                assert path.read_text(encoding="utf-8").strip()


class TestVerification:
    @pytest.fixture
    def sealed(self, tmp_path):
        workspace = Workspace.from_env(tmp_path)
        for zone in workspace.all_zones():
            zone.mkdir(parents=True, exist_ok=True)
        examples = corpus()
        split = split_examples(examples, SplitConfig(strategy="group", seed=42))
        result = ReleaseWriter(workspace).seal(
            version="kleos-policy-v0.1.0",
            split=split,
            provenance_builder=lambda m: build_provenance(
                version="kleos-policy-v0.1.0",
                split=split,
                holdout=HoldoutPlan(),
                manifest=m,
                examples=examples,
                scenario_fingerprints={},
            ),
        )
        open_for_rewrite(result.path)
        return result

    def test_an_untouched_release_verifies(self, sealed) -> None:
        report = verify_release(sealed.path, strict=True)
        assert report.ok, report.render()
        assert report.checks_run > 20

    def test_a_flipped_byte_is_caught(self, sealed) -> None:
        path = sealed.path / "train.jsonl"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("nearest", "NEAREST", 1), encoding="utf-8")

        report = verify_release(sealed.path)
        assert not report.ok
        assert any("changed since it was sealed" in p for p in report.problems)

    def test_a_tampered_count_is_caught(self, sealed) -> None:
        path = sealed.path / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["example_count"] = 999
        path.write_text(json.dumps(manifest), encoding="utf-8")

        report = verify_release(sealed.path)
        assert any("example_count" in p for p in report.problems)

    def test_a_tampered_distribution_is_caught(self, sealed) -> None:
        path = sealed.path / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["task_distribution"] = {"tool_routing": 24}
        path.write_text(json.dumps(manifest), encoding="utf-8")

        report = verify_release(sealed.path)
        assert any("task_distribution" in p for p in report.problems)

    def test_an_id_in_two_splits_is_caught(self, sealed) -> None:
        train = (sealed.path / "train.jsonl").read_text(encoding="utf-8")
        first = train.splitlines()[0]
        test_path = sealed.path / "test.jsonl"
        test_path.write_text(test_path.read_text(encoding="utf-8") + first + "\n", encoding="utf-8")

        report = verify_release(sealed.path)
        assert any("appear in both" in p or "more than once" in p for p in report.problems)

    def test_an_unreviewed_example_is_caught(self, sealed) -> None:
        path = sealed.path / "train.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        first["metadata"]["quality_status"] = "draft"
        lines[0] = json.dumps(first, sort_keys=True, ensure_ascii=False)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        report = verify_release(sealed.path)
        assert any("reviewed" in p for p in report.problems)

    def test_a_stray_jsonl_is_caught(self, sealed) -> None:
        (sealed.path / "synthetic_train.jsonl").write_text("{}\n", encoding="utf-8")
        report = verify_release(sealed.path)
        assert any("unexpected JSONL" in p for p in report.problems)

    def test_a_missing_manifest_is_caught(self, sealed) -> None:
        (sealed.path / "manifest.json").unlink()
        report = verify_release(sealed.path)
        assert any("manifest.json is missing" in p for p in report.problems)

    def test_a_missing_train_split_is_caught(self, sealed) -> None:
        (sealed.path / "train.jsonl").unlink()
        report = verify_release(sealed.path)
        assert not report.ok

    def test_privacy_content_is_caught(self, sealed) -> None:
        path = sealed.path / "train.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        first["messages"][0]["content"] += " Contact j.doe@somecollege.edu."
        lines[0] = json.dumps(first, sort_keys=True, ensure_ascii=False)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        report = verify_release(sealed.path)
        assert any("privacy" in p or "contains_private_data" in p for p in report.problems)

    def test_a_missing_lock_is_a_warning_not_a_failure(self, sealed) -> None:
        (sealed.path / "RELEASE.lock").unlink()
        report = verify_release(sealed.path)
        assert report.ok
        assert any("RELEASE.lock" in w for w in report.warnings)

    def test_strict_promotes_warnings_to_failures(self, sealed) -> None:
        (sealed.path / "RELEASE.lock").unlink()
        assert not verify_release(sealed.path, strict=True).ok

    def test_a_missing_directory_is_reported(self, tmp_path) -> None:
        report = verify_release(tmp_path / "absent")
        assert not report.ok


@pytest.mark.requires_kleos_models
class TestDifferentialSplitting:
    def _payloads(self, count: int = 60) -> list[dict]:
        domains = ["career", "research", "coursework", "projects"]
        formats = ["bullets", "prose", "json"]
        entities = ["set_a", "set_b", "set_c", "pool_z"]
        return [
            {
                "id": f"kx-npr-{i:016d}",
                "task": "notification_prioritization",
                "messages": [
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"a{i}"},
                ],
                "variation_axes": {
                    "domain": domains[i % 4],
                    "format": formats[i % 3],
                    "entities": entities[i % 4],
                },
                "metadata": {
                    "scenario_family": f"fam-{i % 7}",
                    "group_id": f"fam-{i % 7}:{i % 3}",
                },
            }
            for i in range(count)
        ]

    @pytest.mark.parametrize(
        "strategy",
        [
            "random",
            "group",
            "scenario_family_holdout",
            "entity_holdout",
            "domain_holdout",
            "format_holdout",
        ],
    )
    @pytest.mark.parametrize("seed", [0, 1, 42, 1712, 99991])
    def test_the_assignment_matches_the_public_splitter(self, strategy: str, seed: int) -> None:
        from kleos_models.config import SplitConfig as PublicConfig
        from kleos_models.data.schemas import TrainingExample as PublicExample
        from kleos_models.data.splitting import split_examples as public_split

        payloads = self._payloads()
        mine = [TrainingExample.model_validate(p) for p in payloads]
        theirs = [PublicExample.model_validate(p) for p in payloads]

        fractions = {"train_fraction": 0.7, "validation_fraction": 0.15, "test_fraction": 0.15}
        ours = split_examples(mine, SplitConfig(strategy=strategy, seed=seed, **fractions))
        public = public_split(theirs, PublicConfig(strategy=strategy, seed=seed, **fractions))

        def placement(result):
            return {
                e.id: name for name in ("train", "validation", "test") for e in result.split(name)
            }

        assert placement(ours) == placement(public), (
            f"{strategy}@{seed}: our split disagrees with the public one on the "
            f"same data. Nothing would crash — the two repositories would simply "
            f"report different results."
        )

    @pytest.mark.parametrize("strategy", ["entity_holdout", "domain_holdout", "format_holdout"])
    def test_the_chosen_holdout_values_match(self, strategy: str) -> None:
        from kleos_models.config import SplitConfig as PublicConfig
        from kleos_models.data.schemas import TrainingExample as PublicExample
        from kleos_models.data.splitting import split_examples as public_split

        payloads = self._payloads()
        ours = split_examples(
            [TrainingExample.model_validate(p) for p in payloads],
            SplitConfig(strategy=strategy, seed=42),
        )
        public = public_split(
            [PublicExample.model_validate(p) for p in payloads],
            PublicConfig(strategy=strategy, seed=42),
        )
        assert sorted(ours.holdout_values) == sorted(public.holdout_values)

    def test_the_public_loader_reads_our_release(self, tmp_path) -> None:
        from kleos_models.config import DatasetConfig
        from kleos_models.data.loaders import load_dataset_bundle

        workspace = Workspace.from_env(tmp_path)
        for zone in workspace.all_zones():
            zone.mkdir(parents=True, exist_ok=True)

        examples = corpus()
        split = split_examples(examples, SplitConfig(strategy="group", seed=42))
        sealed = ReleaseWriter(workspace).seal(
            version="kleos-policy-v0.1.0",
            split=split,
            provenance_builder=lambda m: build_provenance(
                version="kleos-policy-v0.1.0",
                split=split,
                holdout=HoldoutPlan(),
                manifest=m,
                examples=examples,
                scenario_fingerprints={},
            ),
        )

        bundle = load_dataset_bundle(DatasetConfig(path=sealed.path), strict=True)
        assert len(bundle.train) == len(split.train)

    def test_our_manifest_content_hash_matches_the_public_computation(self, tmp_path) -> None:
        from kleos_models.data.schemas import DatasetManifest as PublicManifest

        workspace = Workspace.from_env(tmp_path)
        for zone in workspace.all_zones():
            zone.mkdir(parents=True, exist_ok=True)
        split = split_examples(corpus(), SplitConfig(strategy="group", seed=42))
        (workspace.releases / "scratch").mkdir(parents=True, exist_ok=True)

        from kleos_training_data.contract.writer import write_jsonl

        write_jsonl(split.train, workspace.releases / "scratch" / "train.jsonl")
        manifest = build_manifest(
            version="v1", split=split, directory=workspace.releases / "scratch"
        )
        public = PublicManifest(version="v1", file_hashes=manifest.file_hashes)
        assert manifest.content_hash == public.compute_content_hash()
