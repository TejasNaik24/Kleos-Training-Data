from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from kleos_training_data.errors import CaptureError
from kleos_training_data.hashing import canonical_hash
from kleos_training_data.staging.records import (
    CaptureLane,
    RawCapture,
    ScenarioRef,
    TransportInfo,
)


@dataclass(frozen=True)
class ScenarioRequest:
    scenario: ScenarioRef
    system_prompt: str
    user_message: str
    variation_axes: dict[str, str]
    group_id: str
    perturbation_of: str | None = None
    perturbation_kind: str | None = None
    expected_answer: str | None = None
    expected_reasoning: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BackendAdapter(Protocol):
    name: str
    lane: CaptureLane

    def endpoint(self) -> str: ...

    def run(self, request: ScenarioRequest, *, batch_id: str) -> RawCapture: ...


@dataclass(frozen=True)
class SSEFrame:
    type: str
    data: dict[str, Any]


def parse_sse_stream(lines: Iterator[str]) -> Iterator[SSEFrame]:
    buffer: list[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if line.startswith(":"):
            continue
        if line == "":
            if buffer:
                yield _frame_from("\n".join(buffer))
                buffer = []
            continue
        if line.startswith("data:"):
            buffer.append(line[len("data:") :].lstrip())
    if buffer:
        yield _frame_from("\n".join(buffer))


def _frame_from(payload: str) -> SSEFrame:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CaptureError(
            "Malformed SSE frame.",
            details={"error": str(exc), "payload_length": len(payload)},
            suggestions=["Silently dropping frames would truncate the answer."],
        ) from exc
    if not isinstance(data, dict) or "type" not in data:
        raise CaptureError("SSE frame has no 'type'.", details={"keys": ", ".join(sorted(data))})
    return SSEFrame(type=str(data["type"]), data=data)


def collect_answer(frames: Iterator[SSEFrame]) -> tuple[str, dict[str, int]]:
    parts: list[str] = []
    counts: dict[str, int] = {}
    done = False

    for frame in frames:
        counts[frame.type] = counts.get(frame.type, 0) + 1
        if frame.type == "answer_delta":
            parts.append(str(frame.data.get("text", "")))
        elif frame.type == "error":
            raise CaptureError(
                "The backend returned an error frame.",
                details={"code": str(frame.data.get("code", "unknown"))},
            )
        elif frame.type == "done":
            done = True

    if not done:
        raise CaptureError(
            "The stream ended without a 'done' frame.",
            details={"frames_seen": ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))},
            suggestions=["A truncated answer must not be treated as a complete one."],
        )
    return "".join(parts), counts


class MockBackendAdapter:
    name = "mock"
    lane = CaptureLane.MOCK_BACKEND

    def __init__(self, *, chunk_size: int = 48, emit_reasoning: bool = True) -> None:
        self.chunk_size = chunk_size
        self.emit_reasoning = emit_reasoning

    def endpoint(self) -> str:
        return "mock://kleos/api/v1/career/projects/chat"

    def _stream(self, answer: str, request: ScenarioRequest) -> Iterator[str]:
        yield ": keepalive"
        yield ""
        yield 'data: {"type": "answer_start"}'
        yield ""

        body = answer
        if self.emit_reasoning:
            body = (
                "<think>Weighing deadline against evidence strength for "
                f"{len(request.variation_axes)} axes.</think>" + answer
            )
        body = body.replace("\n", "\r\n")

        for start in range(0, len(body), self.chunk_size):
            chunk = body[start : start + self.chunk_size]
            yield f"data: {json.dumps({'type': 'answer_delta', 'text': chunk})}"
            yield ""

        yield 'data: {"type": "citation", "n": 2}'
        yield ""
        yield 'data: {"type": "done"}'
        yield ""

    def run(self, request: ScenarioRequest, *, batch_id: str) -> RawCapture:
        if request.expected_answer is None:
            raise CaptureError(
                "The mock adapter needs the policy-derived answer.",
                details={"family": request.scenario.family},
                suggestions=[
                    "The mock lane exercises transport and normalization, not "
                    "answer generation — the target still comes from the policy.",
                ],
            )

        frames = parse_sse_stream(iter(self._stream(request.expected_answer, request)))
        answer, counts = collect_answer(frames)

        seed = canonical_hash(
            {
                "batch": batch_id,
                "family": request.scenario.family,
                "point": request.scenario.point_index,
                "kind": request.perturbation_kind,
                "of": request.perturbation_of,
                "axes": request.variation_axes,
                "system": request.system_prompt,
                "user": request.user_message,
            }
        )
        capture_id = str(uuid.UUID(seed[:32]))

        return RawCapture(
            capture_id=capture_id,
            batch_id=batch_id,
            lane=self.lane,
            adapter=self.name,
            scenario=request.scenario,
            endpoint=self.endpoint(),
            request_field_names=["question", "thinking", "model_mode", "knowledge_base"],
            request_hash=canonical_hash(
                {"system": request.system_prompt, "user": request.user_message}
            ),
            integrations_disabled=True,
            answer_text=answer,
            answer_sha256=hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            transport=TransportInfo(
                attempts=1,
                status=200,
                latency_ms=0,
                request_ids=[capture_id],
                frame_type_counts=counts,
                response_bytes=len(answer.encode("utf-8")),
            ),
            captured_at="1970-01-01T00:00:00+00:00",
        )


CHAT_FORM_DEFAULTS: dict[str, Any] = {
    "thinking": "false",
    "model_mode": "small",
    "knowledge_base": "false",
    "web_search": "false",
    "deep_research": "false",
    "ask_tool_permission": "false",
    "drive_disabled": "true",
    "github_disabled": "true",
    "gitlab_disabled": "true",
    "notion_disabled": "true",
    "slack_disabled": "true",
    "discord_disabled": "true",
    "dropbox_disabled": "true",
    "onedrive_disabled": "true",
}


class KleosChatAdapter:
    name = "kleos_chat"
    lane = CaptureLane.PRODUCTION_OBSERVATION

    def __init__(self, transport: Any) -> None:
        self.transport = transport

    def endpoint(self) -> str:
        return "/api/v1/career/projects/chat"

    def run(self, request: ScenarioRequest, *, batch_id: str) -> RawCapture:
        form = {**CHAT_FORM_DEFAULTS, "question": request.user_message}
        capture_id = str(uuid.uuid4())

        lines, response = self.transport.stream_lines(
            "POST", self.endpoint(), request_id=capture_id, data=form
        )
        answer, counts = collect_answer(parse_sse_stream(iter(lines)))

        return RawCapture(
            capture_id=capture_id,
            batch_id=batch_id,
            lane=self.lane,
            adapter=self.name,
            scenario=request.scenario,
            endpoint=self.endpoint(),
            request_field_names=sorted(form),
            request_hash=canonical_hash(
                {"system": request.system_prompt, "user": request.user_message}
            ),
            integrations_disabled=True,
            answer_text=answer,
            answer_sha256=hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            transport=TransportInfo(
                attempts=response.attempts,
                retried_on=response.retried_on,
                status=response.status,
                latency_ms=response.latency_ms,
                request_ids=[response.request_id],
                frame_type_counts=counts,
                response_bytes=len(answer.encode("utf-8")),
            ),
        )


JSON_ENDPOINTS: dict[str, str] = {
    "briefing": "/api/v1/mission/briefing",
    "notifications": "/api/v1/notifications",
    "memory": "/api/v1/memory",
}


class KleosJsonAdapter:
    name = "kleos_json"
    lane = CaptureLane.PRODUCTION_OBSERVATION

    def __init__(self, transport: Any, *, view: str = "briefing") -> None:
        if view not in JSON_ENDPOINTS:
            raise CaptureError(
                f"Unknown view {view!r}.",
                details={"available": ", ".join(sorted(JSON_ENDPOINTS))},
            )
        self.transport = transport
        self.view = view

    def endpoint(self) -> str:
        return JSON_ENDPOINTS[self.view]

    def run(self, request: ScenarioRequest, *, batch_id: str) -> RawCapture:
        capture_id = str(uuid.uuid4())
        response = self.transport.request("GET", self.endpoint(), request_id=capture_id)

        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise CaptureError(
                f"{self.endpoint()} did not return JSON.",
                details={"status": str(response.status), "bytes": str(len(response.text))},
            ) from exc

        answer = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)

        return RawCapture(
            capture_id=capture_id,
            batch_id=batch_id,
            lane=self.lane,
            adapter=self.name,
            scenario=request.scenario,
            endpoint=self.endpoint(),
            request_field_names=[],
            request_hash=canonical_hash({"view": self.view}),
            integrations_disabled=True,
            answer_text=answer,
            answer_sha256=hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            transport=TransportInfo(
                attempts=response.attempts,
                retried_on=response.retried_on,
                status=response.status,
                latency_ms=response.latency_ms,
                request_ids=[response.request_id],
                frame_type_counts={"json_view": 1},
                response_bytes=len(answer.encode("utf-8")),
            ),
        )


ADAPTERS: dict[str, type] = {
    "mock": MockBackendAdapter,
    "kleos_chat": KleosChatAdapter,
    "kleos_json": KleosJsonAdapter,
}

NEEDS_TRANSPORT: frozenset[str] = frozenset({"kleos_chat", "kleos_json"})


def resolve_adapter(name: str, *, transport: Any = None) -> BackendAdapter:
    factory = ADAPTERS.get(name)
    if factory is None:
        raise CaptureError(
            f"Unknown adapter {name!r}.",
            details={"available": ", ".join(sorted(ADAPTERS))},
        )
    if name in NEEDS_TRANSPORT:
        if transport is None:
            raise CaptureError(
                f"Adapter {name!r} needs an HTTP transport.",
                suggestions=[
                    'Install the extra: pip install -e ".[collect]"',
                    "Everything it captures is lane=production_observation and can "
                    "never be promoted. Use --adapter mock unless you specifically "
                    "want seed material.",
                ],
            )
        return factory(transport)
    return factory()
