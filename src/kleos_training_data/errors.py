"""Actionable exception types.

An error must tell the operator what happened *and* what to do next. Every
exception here carries an optional list of suggested remedies which the CLI
renders as a numbered list.

The exception type determines the process exit code (see ``scripts/_cli.py``):

===========================  ====  ==============================================
Exception                    Exit  Meaning
===========================  ====  ==============================================
``PrivacyViolationError``       4  Private data was found on a path it must not take
``GateFailure``                 3  A promotion gate rejected the data
``VerificationFailure``         3  A release did not verify
``KleosDataError`` (other)      1  The tool could not do the job
``KeyboardInterrupt``         130  Operator interrupted
===========================  ====  ==============================================

The split between 3 and 1 is load-bearing: CI must be able to tell "the data is
bad" from "the tool crashed", because those need different humans. Exit 4 is
carved out of 3 for the same reason — a privacy failure is not a data-quality
failure, and it should page someone.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Process exit codes. Referenced by scripts/_cli.py and asserted by
#: tests/test_cli_contract.py, so the two cannot drift.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_GATE_FAILED = 3
EXIT_PRIVACY_VIOLATION = 4
EXIT_INTERRUPTED = 130


class KleosDataError(Exception):
    """Base class for all errors raised by this repository.

    Args:
        message: What went wrong, in one sentence.
        details: Optional key/value diagnostics printed verbatim.
        suggestions: Concrete next actions, most likely fix first.
    """

    #: Exit code used when this error reaches the CLI boundary.
    exit_code: int = EXIT_ERROR

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, object] | None = None,
        suggestions: Sequence[str] | None = None,
    ) -> None:
        self.message = message
        self.details = dict(details or {})
        self.suggestions = list(suggestions or [])
        super().__init__(self.render())

    def render(self) -> str:
        """Format the error as a multi-line, human-readable diagnostic."""
        parts = [self.message]
        if self.details:
            parts.append("")
            width = max(len(str(k)) for k in self.details)
            for key, value in self.details.items():
                parts.append(f"  {str(key).ljust(width)} : {value}")
        if self.suggestions:
            parts.append("")
            parts.append("Suggested actions:")
            parts.extend(f"  {i}. {s}" for i, s in enumerate(self.suggestions, start=1))
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Configuration and setup
# ---------------------------------------------------------------------------


class ConfigError(KleosDataError):
    """Configuration is missing, malformed, or internally inconsistent."""


class WorkspaceError(KleosDataError):
    """The staging / vault / releases workspace is missing or misconfigured."""


class MissingDependencyError(KleosDataError):
    """An optional dependency is required for this code path but not installed."""

    def __init__(self, package: str, *, extra: str, purpose: str) -> None:
        super().__init__(
            f"{package!r} is required to {purpose} but is not installed.",
            details={"missing_package": package, "install_extra": extra},
            suggestions=[
                f'Install the extra: pip install -e ".[{extra}]"',
                "The offline pipeline needs none of the optional extras — if you did "
                "not mean to leave the offline path, check the --adapter/--reviewer flag.",
            ],
        )
        self.package = package
        self.extra = extra


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


class ContractViolationError(KleosDataError):
    """A payload failed validation against the mirrored public contract.

    Distinct from a pydantic ``ValidationError``: this is raised when the
    pipeline *constructed* something the public repo would reject, which is a
    bug in this repository rather than bad input.
    """


class CompatibilityDriftError(KleosDataError):
    """The contract mirror no longer matches the pinned public repository.

    This is never a runtime error during a normal build — it can only surface
    from the differential tests or ``scripts/check_contract_compat.py``. When it
    does, a dataset built today may not load in the public repo tomorrow.
    """


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


class ScenarioError(KleosDataError):
    """A scenario definition is invalid or could not be rendered."""


class CaptureError(KleosDataError):
    """A capture could not be obtained: transport, adapter, or backend failure."""


class ProductionGuardError(KleosDataError):
    """A capture against a production backend was refused.

    Raised by ``collection.guard``. Every one of its conditions must hold; this
    error names the ones that did not, but deliberately never names a way to
    disable the check itself.
    """


# ---------------------------------------------------------------------------
# Staging and privacy
# ---------------------------------------------------------------------------


class StagingIntegrityError(KleosDataError):
    """A staged record is missing, unparseable, or its record_hash does not match.

    A hash mismatch means the file was edited outside the pipeline. That is not
    necessarily malicious — it is usually someone fixing a typo by hand — but it
    invalidates every downstream signature, so it has to stop the run.
    """


class PrivacyViolationError(KleosDataError):
    """Private data was found somewhere it must never reach.

    This is the one error that should never be caught and continued past.
    """

    exit_code = EXIT_PRIVACY_VIOLATION


class SanitizationError(KleosDataError):
    """Sanitization could not complete, or produced a result it cannot vouch for."""


class VaultError(KleosDataError):
    """The entity vault is missing, unreadable, or has unsafe permissions."""


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------


class ReviewError(KleosDataError):
    """A review record is missing, malformed, or does not match the candidate."""


class ReviewIntegrityError(ReviewError):
    """A review signature does not match the content it claims to cover.

    Almost always means the candidate was edited after approval. The review does
    not carry over — that is the point of signing it.
    """


# ---------------------------------------------------------------------------
# Promotion and release
# ---------------------------------------------------------------------------


class GateFailure(KleosDataError):
    """One or more promotion gates rejected a candidate.

    The tool worked correctly; the data is not promotable.
    """

    exit_code = EXIT_GATE_FAILED


class PromotionIntegrityError(KleosDataError):
    """A mandatory gate did not execute.

    This is the backstop for a refactor that deletes a gate from the table or
    short-circuits the loop on an early failure. It is a bug in this repository,
    never a property of the data, and it must never be downgraded to a warning.
    """


class ReleaseImmutabilityError(KleosDataError):
    """An existing release version would have been modified or overwritten.

    Dataset versions are immutable. If the content needs to change, the version
    string needs to change — there is deliberately no --force.
    """


class VerificationFailure(KleosDataError):
    """A release did not verify against its own manifest and lock file."""

    exit_code = EXIT_GATE_FAILED
