"""The compatibility handshake.

The property that matters most is negative: **a skip must not read as a pass.**
A check that silently succeeds when it could not run produces a green build that
means nothing, and that is worse than having no check at all.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest
from tests.conftest import REPO_ROOT

from kleos_training_data.contract.compat import (
    CompatCheck,
    CompatReport,
    kleos_models_available,
    run_handshake,
)
from kleos_training_data.contract.pin import (
    CONTRACT_SOURCE_COMMIT,
    DELIBERATE_DELTAS,
    KNOWN_UPSTREAM_ISSUES,
    MIRRORED_BEHAVIOURS,
    MIRRORED_CONSTANTS,
)
from kleos_training_data.errors import CompatibilityDriftError


def _load_cli():
    """Import the compat CLI by path — scripts/ is not a package."""
    path = REPO_ROOT / "scripts" / "check_contract_compat.py"
    spec = importlib.util.spec_from_file_location("_compat_cli", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestPin:
    def test_the_pin_names_a_commit_not_a_branch(self) -> None:
        """A moving reference would let the public contract change underneath a
        release without any test failing — the one failure mode this exists to
        prevent."""
        assert len(CONTRACT_SOURCE_COMMIT) == 40
        assert all(c in "0123456789abcdef" for c in CONTRACT_SOURCE_COMMIT)

    def test_every_mirrored_constant_exists_in_the_mirror(self) -> None:
        from kleos_training_data.contract import constants

        missing = [name for name in MIRRORED_CONSTANTS if not hasattr(constants, name)]
        assert not missing, f"pin.py claims to mirror {missing}, which do not exist"

    def test_the_deltas_are_documented(self) -> None:
        """A strictness difference nobody wrote down becomes an unexplained
        disagreement between two repositories."""
        assert DELIBERATE_DELTAS
        for name, why in DELIBERATE_DELTAS.items():
            assert len(why) > 40, f"{name} needs a real explanation"

    def test_upstream_issues_are_recorded_not_reproduced(self) -> None:
        assert KNOWN_UPSTREAM_ISSUES
        for name, why in KNOWN_UPSTREAM_ISSUES.items():
            assert len(why) > 40, f"{name} needs a real explanation"

    def test_mirrored_behaviours_name_their_source(self) -> None:
        """So a reviewer can find the original without searching."""
        for name, source in MIRRORED_BEHAVIOURS.items():
            assert "kleos_models" in source, f"{name} does not name its public source"


class TestReportSemantics:
    def test_an_unavailable_package_is_not_ok(self) -> None:
        """The core property. Absence is not agreement."""
        report = CompatReport(available=False)
        assert not report.ok

    def test_an_available_package_with_no_failures_is_ok(self) -> None:
        report = CompatReport(available=True)
        report.add("something", True)
        assert report.ok

    def test_any_failure_makes_it_not_ok(self) -> None:
        report = CompatReport(available=True)
        report.add("a", True)
        report.add("b", False, "drifted")
        assert not report.ok
        assert [c.name for c in report.failures] == ["b"]

    def test_the_report_serializes(self) -> None:
        import json

        report = CompatReport(available=True)
        report.add("a", True)
        assert json.loads(json.dumps(report.to_dict()))["pinned_commit"]

    def test_the_rendered_report_names_the_pin(self) -> None:
        assert CONTRACT_SOURCE_COMMIT[:12] in CompatReport(available=True).render()

    def test_a_check_renders_its_verdict(self) -> None:
        assert "✗" in CompatCheck("x", False, "why").render()
        assert "✓" in CompatCheck("x", True, "").render()


class TestCLIExitCodes:
    """A skip and a pass must be distinguishable from the exit code alone."""

    def test_strict_fails_when_the_package_is_absent(self, monkeypatch, capsys) -> None:
        cli = _load_cli()
        monkeypatch.setattr(
            cli, "run_handshake", lambda release=None: CompatReport(available=False)
        )
        assert cli.main(["--strict"]) == 3

    def test_without_strict_an_absent_package_skips(self, monkeypatch, capsys) -> None:
        """The normal local state: the offline pipeline does not need the public
        repo checked out."""
        cli = _load_cli()
        monkeypatch.setattr(
            cli, "run_handshake", lambda release=None: CompatReport(available=False)
        )
        assert cli.main([]) == 0
        assert "SKIPPED" in capsys.readouterr().out

    def test_drift_fails_regardless_of_strict(self, monkeypatch, capsys) -> None:
        cli = _load_cli()
        drifted = CompatReport(available=True)
        drifted.add("SUPPORTED_TASKS", False, "a new task appeared")
        monkeypatch.setattr(cli, "run_handshake", lambda release=None: drifted)

        assert cli.main([]) == 3
        assert "INCOMPATIBLE" in capsys.readouterr().err

    def test_agreement_passes(self, monkeypatch, capsys) -> None:
        cli = _load_cli()
        agreed = CompatReport(available=True)
        agreed.add("SUPPORTED_TASKS", True, "identical")
        monkeypatch.setattr(cli, "run_handshake", lambda release=None: agreed)

        assert cli.main([]) == 0
        assert "COMPATIBLE" in capsys.readouterr().out

    def test_the_verdict_is_never_ambiguous(self, monkeypatch, capsys) -> None:
        """One word, always. It never silently continues."""
        cli = _load_cli()
        for report, expected in (
            (CompatReport(available=True), "COMPATIBLE"),
            (CompatReport(available=False), "SKIPPED"),
        ):
            monkeypatch.setattr(cli, "run_handshake", lambda release=None, r=report: r)
            cli.main([])
            assert expected in capsys.readouterr().out


@pytest.mark.requires_kleos_models
class TestHandshakeAgainstThePublicRepo:
    def test_the_handshake_reports_compatible(self) -> None:
        report = run_handshake()
        assert report.available
        assert report.ok, "\n".join(c.render() for c in report.failures)

    def test_every_mirrored_constant_is_checked(self) -> None:
        """A constant listed in the pin but never compared is a false assurance."""
        checked = {c.name for c in run_handshake().checks}
        assert set(MIRRORED_CONSTANTS) <= checked

    def test_the_ported_functions_are_checked(self) -> None:
        checked = {c.name for c in run_handshake().checks}
        assert {"stable_rank", "normalize_text", "jsonl_writer_bytes"} <= checked

    def test_assert_compatible_raises_on_drift(self, monkeypatch) -> None:
        import kleos_training_data.contract.compat as compat

        drifted = CompatReport(available=True)
        drifted.add("SUPPORTED_TASKS", False, "drifted")
        monkeypatch.setattr(compat, "run_handshake", lambda release=None: drifted)

        with pytest.raises(CompatibilityDriftError, match="compatibility check"):
            compat.assert_compatible()

    def test_assert_compatible_raises_when_unavailable(self, monkeypatch) -> None:
        import kleos_training_data.contract.compat as compat

        monkeypatch.setattr(
            compat, "run_handshake", lambda release=None: CompatReport(available=False)
        )
        with pytest.raises(CompatibilityDriftError, match="not installed"):
            compat.assert_compatible()

    def test_a_real_release_round_trips_through_the_public_loader(self, tmp_path) -> None:
        """The end the whole arrangement serves."""
        from tests.test_datasets import corpus

        from kleos_training_data.contract.splitting import SplitConfig, split_examples
        from kleos_training_data.datasets.holdouts import HoldoutPlan
        from kleos_training_data.datasets.manifest import build_provenance
        from kleos_training_data.datasets.release import ReleaseWriter
        from kleos_training_data.paths import Workspace

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

        report = run_handshake(sealed.path)
        assert report.ok, "\n".join(c.render() for c in report.failures)
        names = {c.name for c in report.checks}
        assert {"public_loader", "public_validator", "manifest_content_hash"} <= names


class TestEnvironment:
    def test_availability_is_a_boolean_not_an_exception(self) -> None:
        """Callers branch on this; it must never raise."""
        assert isinstance(kleos_models_available(), bool)

    def test_a_bogus_local_checkout_is_ignored(self, monkeypatch, tmp_path) -> None:
        from kleos_training_data.contract.compat import ENV_MODELS_PATH

        monkeypatch.setenv(ENV_MODELS_PATH, str(tmp_path / "nowhere"))
        assert isinstance(kleos_models_available(), bool)


class TestSliceTarget:
    def test_the_makefile_defines_the_slice(self) -> None:
        """The checkpoint is a command somebody can run, not a description."""
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        assert "\nslice:" in makefile
        assert "slice-clean:" in makefile

    def test_the_slice_covers_the_whole_chain(self) -> None:
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        slice_body = makefile.split("\nslice:")[1].split("\n.PHONY")[0]
        for stage in (
            "validate_scenarios",
            "capture_backend",
            "normalize_captures",
            "sanitize_candidates",
            "build_review_packet",
            "run_llm_review",
            "record_decision",
            "promote_examples",
            "build_release",
            "verify_release",
            "check_contract_compat",
        ):
            assert stage in slice_body, f"the slice does not run {stage}"

    def test_the_slice_uses_the_mock_adapter(self) -> None:
        """It must be provable offline, with no credentials and no private data."""
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        slice_body = makefile.split("\nslice:")[1].split("\n.PHONY")[0]
        assert "--adapter mock" in slice_body
        assert "--reviewer mock" in slice_body


class TestCIConfiguration:
    def test_ci_needs_no_credentials(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert "secrets." not in workflow

    def test_the_privacy_scan_runs_first(self) -> None:
        """A leaked secret should fail the build in seconds, not after an install."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        jobs = workflow.split("\njobs:")[1]
        assert jobs.index("privacy:") < jobs.index("lint:")

    def test_the_compat_job_is_not_allowed_to_fail(self) -> None:
        """A SKIPPED differential suite is a degraded run, not a passing one."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert "continue-on-error" not in workflow

    def test_the_compat_job_runs_strict(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert "check_contract_compat.py --strict" in workflow

    def test_the_slice_runs_in_ci(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert "make slice-clean slice" in workflow
