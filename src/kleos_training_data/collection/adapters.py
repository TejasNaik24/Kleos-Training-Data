"""Backend adapters.

The adapter boundary is ``ScenarioRequest -> RawCapture``; everything downstream
sees only a ``RawCapture``. Three genuinely different response disciplines sit
behind it — the real chat endpoint is a ~40-field multipart form returning an SSE
stream, the mission and notification endpoints are plain JSON GETs, and the mock
is a pure function — so a single client with ``isinstance`` branches would be
unreadable.

Only the mock adapter is implemented here. The real ones arrive after the
vertical slice is proven, deliberately: the slice has to be demonstrable offline,
in CI, with no credentials. If a real adapter were load-bearing for it, every
test would depend on a live service holding one person's private records, and
the first end-to-end run would create private data before any gate existed to
catch it.
"""

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
    """One thing to ask a backend."""

    scenario: ScenarioRef
    system_prompt: str
    user_message: str
    #: Contract-shaped axes, carried through so normalization need not re-derive.
    variation_axes: dict[str, str]
    group_id: str
    perturbation_of: str | None = None
    perturbation_kind: str | None = None
    #: Set for the mock lane, which knows the answer the policy implies.
    expected_answer: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BackendAdapter(Protocol):
    """What every adapter must provide."""

    name: str
    lane: CaptureLane

    def endpoint(self) -> str:
        """The path this adapter targets, for the audit record."""
        ...

    def run(self, request: ScenarioRequest, *, batch_id: str) -> RawCapture:
        """Execute one request and return an untrusted capture."""
        ...


# ---------------------------------------------------------------------------
# SSE parsing — shared by the mock and, later, the real chat adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SSEFrame:
    """One server-sent event."""

    type: str
    data: dict[str, Any]


def parse_sse_stream(lines: Iterator[str]) -> Iterator[SSEFrame]:
    """Parse an SSE byte stream into frames.

    Handles the cases the real endpoint actually produces: multi-line ``data:``
    payloads, comment lines, and blank-line frame separators. A malformed frame
    raises rather than being skipped — silently dropping frames would truncate an
    answer without anything noticing.
    """
    buffer: list[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if line.startswith(":"):  # comment / keepalive
            continue
        if line == "":
            if buffer:
                yield _frame_from("\n".join(buffer))
                buffer = []
            continue
        if line.startswith("data:"):
            buffer.append(line[len("data:") :].lstrip())
        # `event:` and `id:` fields are ignored: the frame type travels inside
        # the JSON payload for this backend.
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
    """Assemble an answer from a frame stream.

    Returns:
        ``(answer_text, frame_type_counts)``. Counts only — frame *bodies* are
        never recorded, so the audit trail can show what the backend did without
        storing what it said.
    """
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


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------


class MockBackendAdapter:
    """A deterministic offline backend.

    It does not merely hand back the expected answer. It emits a realistic SSE
    frame stream, chunked across ``answer_delta`` frames, wrapped in a
    ``<think>`` span and with CRLF line endings — so the normalization stage is
    genuinely exercised rather than reduced to a pass-through. A mock that gives
    normalization nothing to do proves nothing about it.
    """

    name = "mock"
    lane = CaptureLane.MOCK_BACKEND

    def __init__(self, *, chunk_size: int = 48, emit_reasoning: bool = True) -> None:
        self.chunk_size = chunk_size
        self.emit_reasoning = emit_reasoning

    def endpoint(self) -> str:
        return "mock://kleos/api/v1/career/projects/chat"

    def _stream(self, answer: str, request: ScenarioRequest) -> Iterator[str]:
        """Render an answer as SSE lines, the way the real endpoint would."""
        yield ": keepalive"
        yield ""
        yield 'data: {"type": "answer_start"}'
        yield ""

        body = answer
        if self.emit_reasoning:
            # The real backend's thinking mode emits a reasoning span that the
            # public formatter strips before tokenization. Normalization has to
            # remove it here, or it becomes a training target.
            body = (
                "<think>Weighing deadline against evidence strength for "
                f"{len(request.variation_axes)} axes.</think>" + answer
            )
        # CRLF on purpose: a real HTTP stream carries them and they must not
        # reach a content hash.
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

        # Deterministic capture id: re-running a batch must not produce a new
        # file for identical content, or staging fills with near-duplicates.
        seed = canonical_hash(
            {
                "batch": batch_id,
                "family": request.scenario.family,
                "point": request.scenario.point_index,
                "kind": request.perturbation_kind,
                "of": request.perturbation_of,
                "axes": request.variation_axes,
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
            # Frozen, not `now()`: a batch regenerated from the same scenario
            # must be byte-identical, or reproducibility is a claim rather than
            # a property.
            captured_at="1970-01-01T00:00:00+00:00",
        )


# ---------------------------------------------------------------------------
# Real backend adapters
# ---------------------------------------------------------------------------


#: Form fields the chat endpoint accepts, with every integration forced off.
#:
#: The real endpoint takes roughly forty `Form(...)` fields. Any integration
#: left on pulls more of the operator's connected accounts — Drive, GitHub,
#: Notion, Slack — into a capture that is already one person's private data, for
#: no research value at all. Turning one on requires editing this dict, which is
#: a diff a reviewer sees.
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
    """The real ``POST /api/v1/career/projects/chat`` endpoint.

    Multipart form in, SSE stream out. Everything it returns is
    ``lane=production_observation`` and can never be promoted — the backend
    answers from the authenticated user's own stored projects and memories, so a
    capture is that person's private data whatever the prompt said. It is seed
    material for writing a new generalized scenario, and nothing else.
    """

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


#: JSON endpoints that return a whole view rather than a stream.
JSON_ENDPOINTS: dict[str, str] = {
    "briefing": "/api/v1/mission/briefing",
    "notifications": "/api/v1/notifications",
    "memory": "/api/v1/memory",
}


class KleosJsonAdapter:
    """The plain-JSON views: mission briefing, notifications, memory.

    A different response discipline from the chat endpoint — a whole view, not a
    stream — which is why the adapter boundary exists rather than one client with
    branches. Same lane, same prohibition: these views *are* the user's stored
    data, so a capture can never be promoted.
    """

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

        # Serialized deterministically so the same view produces the same bytes,
        # and so a capture is diffable against a later one.
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


#: Adapter registry. The real adapters register here once they exist.
ADAPTERS: dict[str, type] = {
    "mock": MockBackendAdapter,
    "kleos_chat": KleosChatAdapter,
    "kleos_json": KleosJsonAdapter,
}

#: Adapters that require a transport, and therefore the `collect` extra.
NEEDS_TRANSPORT: frozenset[str] = frozenset({"kleos_chat", "kleos_json"})


def resolve_adapter(name: str, *, transport: Any = None) -> BackendAdapter:
    """Instantiate an adapter by name.

    The real adapters need a transport; the mock refuses one, so a caller cannot
    accidentally point the offline lane at a network client.
    """
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
