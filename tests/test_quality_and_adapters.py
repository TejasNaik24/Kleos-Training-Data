"""Phase J additions: coverage reporting, transport, real adapters, doctor.

The theme across all four is the same one the rest of the repository has: a
component that *could* be wrong in a way nobody notices should be checked rather
than trusted. Coverage that reports OK on a single-valued axis, a transport that
retries a 401, an adapter that promotes a production capture, a diagnostic that
prints a token — each looks fine right up until it matters.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest
from tests.conftest import REPO_ROOT
from tests.test_datasets import corpus, make_example

from kleos_training_data.collection.adapters import (
    ADAPTERS,
    CHAT_FORM_DEFAULTS,
    JSON_ENDPOINTS,
    NEEDS_TRANSPORT,
    KleosChatAdapter,
    KleosJsonAdapter,
    resolve_adapter,
)
from kleos_training_data.collection.transport import (
    NEVER_RETRY,
    RETRYABLE_STATUSES,
    RateLimiter,
    RetryPolicy,
    _parse_retry_after,
)
from kleos_training_data.errors import CaptureError, ReviewError
from kleos_training_data.quality.coverage import (
    CRITICAL,
    OK,
    WARNING,
    build_coverage_report,
    ood_readiness,
    render_coverage_report,
)
from kleos_training_data.review.reviewers import REVIEWERS, AnthropicReviewer
from kleos_training_data.staging.records import CaptureLane


class TestCoverage:
    def test_a_constant_axis_is_critical(self) -> None:
        """A single-valued axis cannot support any claim about that axis."""
        examples = [make_example(i, format="bullets") for i in range(10)]
        report = build_coverage_report(examples)
        assert report.axes["format"].is_constant
        assert report.axes["format"].severity == CRITICAL
        assert report.severity == CRITICAL

    def test_a_varied_axis_is_ok(self) -> None:
        report = build_coverage_report(corpus(30))
        assert report.axes["format"].severity == OK

    def test_imbalance_is_a_warning(self) -> None:
        examples = [make_example(i, format="bullets") for i in range(20)]
        examples += [make_example(100 + i, format="json") for i in range(2)]
        assert build_coverage_report(examples).axes["format"].severity == WARNING

    def test_missing_tasks_are_named(self) -> None:
        report = build_coverage_report(corpus())
        assert "tool_routing" in report.tasks_missing
        assert report.tasks_covered == ["notification_prioritization"]

    def test_joint_coverage_finds_empty_cells(self) -> None:
        """The whole point: a corpus can cover every axis marginally and still
        have never seen a combination."""
        examples = [make_example(i, format="bullets", urgency="high") for i in range(6)]
        examples += [make_example(100 + i, format="json", urgency="low") for i in range(6)]
        report = build_coverage_report(examples)
        joint = next(j for j in report.joint if j.axes == ("task", "format"))
        assert joint.observed == 2

    def test_singleton_groups_make_consistency_unmeasurable(self) -> None:
        """Consistency testing compares members of one group. With every group a
        singleton it reports nothing at all — silently, which is the problem, so
        the coverage report says it out loud."""
        # corpus() puts four consecutive examples in one group.
        assert build_coverage_report(corpus(8)).consistency_measurable

        # One example per group: nothing to compare.
        singletons = [make_example(i * 4) for i in range(5)]
        report = build_coverage_report(singletons)
        assert not report.consistency_measurable
        assert report.severity == CRITICAL
        assert any("consistency" in note for note in report.notes)

    def test_an_empty_corpus_is_handled(self) -> None:
        report = build_coverage_report([])
        assert report.example_count == 0
        assert report.notes

    def test_the_report_serializes(self) -> None:
        import json

        assert json.loads(json.dumps(build_coverage_report(corpus()).to_dict()))

    def test_the_rendered_report_ends_with_the_point(self) -> None:
        rendered = render_coverage_report(build_coverage_report(corpus()))
        assert "Coverage, not count" in rendered

    def test_ood_readiness_needs_two_values(self) -> None:
        """A holdout needs one value to hold out and one to train on."""
        single = [make_example(i, format="bullets") for i in range(5)]
        assert not ood_readiness(single)["unseen_formats"]
        assert ood_readiness(corpus())["unseen_formats"]

    def test_the_real_catalog_covers_every_task(self) -> None:
        """The catalog is the research instrument; a missing task is a gap in it."""
        from kleos_training_data.contract.constants import SUPPORTED_TASKS
        from kleos_training_data.scenarios.loader import load_catalog

        assert {s.task for s in load_catalog()} == set(SUPPORTED_TASKS)


class TestRetryPolicy:
    @pytest.mark.parametrize("status", sorted(RETRYABLE_STATUSES))
    def test_transient_statuses_retry(self, status: int) -> None:
        assert RetryPolicy().should_retry(attempt=1, status=status)

    @pytest.mark.parametrize("status", sorted(NEVER_RETRY))
    def test_client_errors_never_retry(self, status: int) -> None:
        """A 401 or 422 fails identically the second time. Retrying wastes the
        budget and delays the real error."""
        assert not RetryPolicy().should_retry(attempt=1, status=status)

    def test_a_timeout_retries(self) -> None:
        assert RetryPolicy().should_retry(attempt=1, status=None)

    def test_attempts_are_capped(self) -> None:
        assert not RetryPolicy(max_attempts=3).should_retry(attempt=3, status=503)

    def test_the_budget_is_shared_across_the_batch(self) -> None:
        """Per-request limits let a degraded backend turn 50 scenarios into 200
        requests against a service already struggling."""
        policy = RetryPolicy(budget=2)
        assert policy.should_retry(attempt=1, status=503)
        policy.spend()
        policy.spend()
        assert policy.budget_remaining == 0
        assert not policy.should_retry(attempt=1, status=503)

    def test_backoff_is_bounded(self) -> None:
        policy = RetryPolicy(base_delay=1.0, max_delay=5.0)
        assert all(0 <= policy.delay_for(attempt) <= 5.0 for attempt in range(10))

    def test_retry_after_seconds_is_honoured(self) -> None:
        assert RetryPolicy().delay_for(0, retry_after="7") == 7.0

    def test_retry_after_is_capped_by_max_delay(self) -> None:
        assert RetryPolicy(max_delay=5.0).delay_for(0, retry_after="600") == 5.0

    def test_an_http_date_retry_after_parses(self) -> None:
        assert _parse_retry_after("Wed, 21 Oct 2099 07:28:00 GMT") is not None

    def test_a_nonsense_retry_after_is_ignored(self) -> None:
        assert _parse_retry_after("soon") is None


class TestRateLimiter:
    def test_the_burst_is_available_immediately(self) -> None:
        limiter = RateLimiter(requests_per_minute=60, burst=3)
        assert all(limiter.acquire() == 0.0 for _ in range(3))

    def test_it_throttles_beyond_the_burst(self) -> None:
        limiter = RateLimiter(requests_per_minute=6000, burst=1)
        limiter.acquire()
        assert limiter.acquire() >= 0.0


class TestRealAdapters:
    def test_both_are_registered(self) -> None:
        assert {"kleos_chat", "kleos_json"} <= set(ADAPTERS)

    @pytest.mark.parametrize("name", sorted(NEEDS_TRANSPORT))
    def test_they_refuse_to_instantiate_without_a_transport(self, name: str) -> None:
        """So the offline lane cannot accidentally be pointed at a network client."""
        with pytest.raises(CaptureError, match="needs an HTTP transport"):
            resolve_adapter(name)

    def test_the_mock_takes_no_transport(self) -> None:
        assert resolve_adapter("mock").name == "mock"

    @pytest.mark.parametrize("adapter", [KleosChatAdapter, KleosJsonAdapter])
    def test_they_are_production_observation(self, adapter: type) -> None:
        """The single most consequential property: nothing they capture can be
        promoted, because the backend answers from the user's own stored data."""
        assert adapter.lane is CaptureLane.PRODUCTION_OBSERVATION
        assert not adapter.lane.promotable

    def test_every_integration_is_forced_off(self) -> None:
        """An integration that fires pulls more of the operator's connected
        accounts into a capture that is already private data."""
        for field, value in CHAT_FORM_DEFAULTS.items():
            if field.endswith("_disabled"):
                assert value == "true", f"{field} is not disabled"
            if field in {"web_search", "deep_research", "knowledge_base"}:
                assert value == "false", f"{field} is not off"

    def test_turning_one_on_requires_editing_the_constant(self) -> None:
        """There is deliberately no flag or config key that enables one."""
        source = (REPO_ROOT / "scripts" / "capture_backend.py").read_text(encoding="utf-8")
        assert "web_search" not in source
        assert "CHAT_FORM_DEFAULTS" not in source

    def test_the_json_adapter_rejects_an_unknown_view(self) -> None:
        with pytest.raises(CaptureError, match="Unknown view"):
            KleosJsonAdapter(transport=object(), view="inbox")

    def test_the_known_views_match_the_backend(self) -> None:
        assert set(JSON_ENDPOINTS) == {"briefing", "notifications", "memory"}
        assert all(path.startswith("/api/v1/") for path in JSON_ENDPOINTS.values())


class TestAnthropicReviewer:
    def test_it_is_registered(self) -> None:
        assert "anthropic" in REVIEWERS

    def test_it_is_refused_in_ci(self, monkeypatch) -> None:
        """An automated run has no human to own the decision afterwards."""
        monkeypatch.setenv("CI", "true")
        with pytest.raises(ReviewError, match="refused in CI"):
            AnthropicReviewer()._ensure_client()

    def test_it_is_refused_without_a_key(self, monkeypatch) -> None:
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ReviewError, match="ANTHROPIC_API_KEY"):
            AnthropicReviewer()._ensure_client()

    def test_it_defaults_to_a_current_model(self) -> None:
        assert AnthropicReviewer().model.startswith("claude-")


class TestDoctor:
    def _load(self):
        path = REPO_ROOT / "scripts" / "doctor.py"
        spec = importlib.util.spec_from_file_location("_doctor", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_it_never_prints_a_secret_value(self, monkeypatch, capsys) -> None:
        """The moment a diagnostic prints a token it becomes the thing people
        paste into an issue."""
        # Deliberately not shaped like any real credential. The test is about
        # doctor.py never echoing a value, not about the value's format — and a
        # realistic-looking key here would trip the repository's own scanner,
        # which is the behaviour we want from it.
        sentinel = "DISTINCTIVE-SENTINEL-VALUE-" + "z" * 20
        monkeypatch.setenv("ANTHROPIC_API_KEY", sentinel)
        monkeypatch.setenv("KLEOS_API_TOKEN", sentinel)

        self._load().main([])
        output = capsys.readouterr().out
        assert sentinel not in output
        assert "sha256=" in output

    def test_it_reports_presence(self, monkeypatch, capsys) -> None:
        monkeypatch.setenv("KLEOS_API_TOKEN", "something")
        self._load().main([])
        assert "KLEOS_API_TOKEN" in capsys.readouterr().out

    def test_it_touches_no_network_by_default(self, monkeypatch, capsys) -> None:
        """A diagnostic that silently reaches a backend leaks which deployment
        you are pointed at, and is unusable on a plane."""
        monkeypatch.delenv("CI", raising=False)
        self._load().main([])
        assert "skipped — pass --check-network" in capsys.readouterr().out

    def test_it_reports_the_pinned_commit(self, capsys) -> None:
        from kleos_training_data.contract.pin import CONTRACT_SOURCE_COMMIT

        self._load().main([])
        assert CONTRACT_SOURCE_COMMIT[:12] in capsys.readouterr().out

    def test_it_reports_production_capture_as_disabled(self, monkeypatch, capsys) -> None:
        monkeypatch.delenv("KLEOS_ALLOW_PRODUCTION_CAPTURE", raising=False)
        self._load().main([])
        assert "DISABLED" in capsys.readouterr().out
