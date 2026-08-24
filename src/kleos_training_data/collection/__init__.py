"""Capture candidate material from a backend, or generate it locally.

The collection layer produces *candidates*, never trusted training data.
Everything it emits lands in ``staging/raw`` marked with its capture lane, and
nothing reaches a dataset without passing every promotion gate.
"""

from __future__ import annotations

from kleos_training_data.collection.adapters import (
    ADAPTERS,
    BackendAdapter,
    MockBackendAdapter,
    ScenarioRequest,
    SSEFrame,
    collect_answer,
    parse_sse_stream,
    resolve_adapter,
)
from kleos_training_data.collection.guard import (
    CONFIRM_PHRASE,
    CaptureAuthorization,
    assert_capture_allowed,
    is_local,
)
from kleos_training_data.collection.runner import BatchResult, load_batch, run_batch, scenario_ref

__all__ = [
    "ADAPTERS",
    "CONFIRM_PHRASE",
    "BackendAdapter",
    "BatchResult",
    "CaptureAuthorization",
    "MockBackendAdapter",
    "SSEFrame",
    "ScenarioRequest",
    "assert_capture_allowed",
    "collect_answer",
    "is_local",
    "load_batch",
    "parse_sse_stream",
    "resolve_adapter",
    "run_batch",
    "scenario_ref",
]
