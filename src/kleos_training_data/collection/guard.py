"""The production capture guard.

The KLEOS backend answers from the *authenticated user's own* stored projects,
memories and notifications. It is not a scenario simulator: every capture from a
real deployment is one person's private data, whatever the prompt was.

So capturing against a non-local backend requires **all** of these at once:

1. ``--allow-production`` on the command line
2. ``KLEOS_ALLOW_PRODUCTION_CAPTURE=1`` in the environment
3. The exact confirmation phrase, typed
4. ``CI`` unset

Four independent conditions because any one alone is something a person can do
by accident or a script can inherit. The fourth is **not overridable**: an
automated production capture is never legitimate, and a flag that could disable
that check would eventually be set in a workflow file and forgotten.

The error names which conditions failed. It deliberately never names a way to
turn the guard off.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final
from urllib.parse import urlparse

from kleos_training_data.errors import ProductionGuardError

#: Hosts treated as local development. Anything else is production until proven
#: otherwise — the default has to be the safe one.
LOCAL_HOST_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"^127\.0\.0\.1$"),
    re.compile(r"^localhost$"),
    re.compile(r"^0\.0\.0\.0$"),
    re.compile(r"^\[::1\]$"),
    re.compile(r"^[\w-]+\.local$"),
    re.compile(r"^[\w-]+\.test$"),
)

#: Scheme used by the offline mock adapter, which touches no network at all.
MOCK_SCHEME: Final[str] = "mock"

#: Typed in full, by a human, every time. Long enough that it cannot be muscle
#: memory and specific enough that it states what is actually happening.
CONFIRM_PHRASE: Final[str] = "I understand this captures real personal data"

ENV_ALLOW: Final[str] = "KLEOS_ALLOW_PRODUCTION_CAPTURE"


def is_local(base_url: str) -> bool:
    """Whether a URL points at a local development backend or the mock."""
    parsed = urlparse(base_url)
    if parsed.scheme == MOCK_SCHEME:
        return True
    host = parsed.hostname or ""
    return any(pattern.match(host) for pattern in LOCAL_HOST_PATTERNS)


@dataclass(frozen=True)
class CaptureAuthorization:
    """The audit record written when a production capture is permitted.

    Records the host, never the full URL: a URL can carry a token in its query
    string, and this file sits next to the batch it authorized.
    """

    base_url_host: str
    authorized_at: str
    operator_role: str
    scenario_families: tuple[str, ...]
    expected_captures: int
    conditions_met: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "base_url_host": self.base_url_host,
            "authorized_at": self.authorized_at,
            "operator_role": self.operator_role,
            "scenario_families": list(self.scenario_families),
            "expected_captures": self.expected_captures,
            "conditions_met": list(self.conditions_met),
            "note": (
                "Captures from this batch are lane=production_observation and can "
                "never be promoted. They are seed material for new scenarios only. "
                "See PRIVACY.md, 'Capture lanes'."
            ),
        }


def assert_capture_allowed(
    base_url: str,
    *,
    allow_production: bool,
    confirm: str | None,
    env: dict[str, str],
    operator_role: str = "operator",
    scenario_families: tuple[str, ...] = (),
    expected_captures: int = 0,
) -> CaptureAuthorization | None:
    """Permit or refuse a capture run.

    Returns:
        ``None`` for a local or mock backend, where no authorization is needed.
        A :class:`CaptureAuthorization` when a production capture is permitted.

    Raises:
        ProductionGuardError: If any condition fails.
    """
    if is_local(base_url):
        return None

    failures: list[str] = []
    met: list[str] = []

    if allow_production:
        met.append("--allow-production")
    else:
        failures.append("--allow-production was not passed")

    if env.get(ENV_ALLOW, "").strip() == "1":
        met.append(f"{ENV_ALLOW}=1")
    else:
        failures.append(f"{ENV_ALLOW} is not set to 1")

    if confirm == CONFIRM_PHRASE:
        met.append("confirmation phrase")
    else:
        failures.append("the confirmation phrase was not given exactly")

    # Unconditional and last, so it is the one that shows up in a CI log.
    if env.get("CI"):
        failures.append(
            "CI is set — an automated production capture is never legitimate, "
            "and this condition cannot be overridden"
        )
    else:
        met.append("not running in CI")

    if failures:
        raise ProductionGuardError(
            f"Refusing to capture against a non-local backend ({urlparse(base_url).hostname}).",
            details={
                "unmet_conditions": "; ".join(failures),
                "met_conditions": "; ".join(met) or "(none)",
            },
            suggestions=[
                "Point --base-url at a local development backend instead.",
                "Or use --adapter mock, which touches no network and is what the "
                "test suite and CI use.",
                "A capture from a real deployment is one person's private data and "
                "can never be promoted into a dataset — it is seed material for "
                "writing a new generalized scenario. See PRIVACY.md.",
            ],
        )

    return CaptureAuthorization(
        base_url_host=urlparse(base_url).hostname or "",
        authorized_at=datetime.now(UTC).isoformat(),
        operator_role=operator_role,
        scenario_families=scenario_families,
        expected_captures=expected_captures,
        conditions_met=tuple(met),
    )
