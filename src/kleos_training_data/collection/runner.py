from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kleos_training_data.collection.adapters import BackendAdapter, ScenarioRequest
from kleos_training_data.contract.constants import PIPELINE_VERSION
from kleos_training_data.logging_utils import get_logger
from kleos_training_data.paths import Workspace
from kleos_training_data.scenarios.generator import Candidate, generate, generation_fingerprint
from kleos_training_data.scenarios.models import Scenario
from kleos_training_data.scenarios.surrogates import SurrogatePool
from kleos_training_data.staging.records import CaptureLane, RawCapture, ScenarioRef
from kleos_training_data.staging.store import write_record

logger = get_logger(__name__)


def scenario_ref(scenario: Scenario, point_index: int) -> ScenarioRef:
    return ScenarioRef(
        family=scenario.family,
        task=scenario.task,
        point_index=point_index,
        catalog_version=scenario.catalog_version,
        policy_claim=" ".join(scenario.policy_claim.split()),
        policy=scenario.expected.policy,
        scenario_fingerprint=generation_fingerprint(scenario),
    )


def to_request(scenario: Scenario, candidate: Candidate) -> ScenarioRequest:
    system, user, assistant = (m["content"] for m in candidate.messages)
    return ScenarioRequest(
        scenario=scenario_ref(scenario, candidate.situation.point_index),
        system_prompt=system,
        user_message=user,
        variation_axes=dict(candidate.variation_axes),
        group_id=candidate.group_id,
        perturbation_of=candidate.perturbation_of,
        perturbation_kind=candidate.perturbation_kind,
        expected_answer=assistant,
        expected_reasoning=candidate.messages[2].get("reasoning"),
    )


@dataclass
class BatchResult:
    batch_id: str
    lane: CaptureLane
    adapter: str
    requests: int
    captures: int
    written: list[Path]
    failures: list[tuple[str, str]]

    @property
    def ok(self) -> bool:
        return not self.failures


def run_batch(
    scenarios: list[Scenario],
    *,
    adapter: BackendAdapter,
    workspace: Workspace,
    batch_id: str,
    limit: int | None = None,
) -> BatchResult:
    destination = workspace.raw_batch(batch_id)
    written: list[Path] = []
    failures: list[tuple[str, str]] = []
    requests = 0

    seen_capture_ids: dict[str, str] = {}

    for scenario in scenarios:
        pool = SurrogatePool.load(scenario.entities.pool)
        candidates = generate(scenario, pool)

        for candidate in candidates:
            if limit is not None and requests >= limit:
                break
            requests += 1
            request = to_request(scenario, candidate)
            try:
                capture = adapter.run(request, batch_id=batch_id)
            except Exception as exc:
                failures.append((f"{scenario.family}:{candidate.situation.point_index}", str(exc)))
                logger.warning(
                    "capture failed family=%s point=%d adapter=%s",
                    scenario.family,
                    candidate.situation.point_index,
                    adapter.name,
                )
                continue

            if capture.capture_id in seen_capture_ids:
                failures.append(
                    (
                        f"{scenario.family}:{candidate.situation.point_index}",
                        f"capture_id {capture.capture_id} was already produced by "
                        f"{seen_capture_ids[capture.capture_id]}; writing it would "
                        f"overwrite that capture",
                    )
                )
                continue
            seen_capture_ids[capture.capture_id] = (
                f"{scenario.family}:{candidate.situation.point_index}"
                f"{'/' + candidate.perturbation_kind if candidate.perturbation_kind else ''}"
            )

            path = write_record(capture, destination / f"{capture.capture_id}.json")
            written.append(path)
            logger.info(
                "capture_id=%s family=%s point=%d status=%s latency_ms=%s frames=%d bytes=%s",
                capture.capture_id,
                scenario.family,
                candidate.situation.point_index,
                capture.transport.status,
                capture.transport.latency_ms,
                sum(capture.transport.frame_type_counts.values()),
                capture.transport.response_bytes,
            )

    manifest = {
        "batch_id": batch_id,
        "lane": adapter.lane.value,
        "adapter": adapter.name,
        "endpoint": adapter.endpoint(),
        "pipeline_version": PIPELINE_VERSION,
        "families": sorted(s.family for s in scenarios),
        "requests": requests,
        "captures": len(written),
        "failures": len(failures),
    }
    (destination / "_batch.json").write_text(
        __import__("json").dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return BatchResult(
        batch_id=batch_id,
        lane=adapter.lane,
        adapter=adapter.name,
        requests=requests,
        captures=len(written),
        written=written,
        failures=failures,
    )


def load_batch(workspace: Workspace, batch_id: str) -> list[RawCapture]:
    from kleos_training_data.staging.store import iter_records

    return list(iter_records(workspace.raw_batch(batch_id), RawCapture))
