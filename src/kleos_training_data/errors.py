from __future__ import annotations

from collections.abc import Sequence

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_GATE_FAILED = 3
EXIT_PRIVACY_VIOLATION = 4
EXIT_INTERRUPTED = 130


class KleosDataError(Exception):
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


class ConfigError(KleosDataError):
    pass


class WorkspaceError(KleosDataError):
    pass


class MissingDependencyError(KleosDataError):
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


class ContractViolationError(KleosDataError):
    pass


class CompatibilityDriftError(KleosDataError):
    pass


class ScenarioError(KleosDataError):
    pass


class CaptureError(KleosDataError):
    pass


class ProductionGuardError(KleosDataError):
    pass


class StagingIntegrityError(KleosDataError):
    pass


class PrivacyViolationError(KleosDataError):
    exit_code = EXIT_PRIVACY_VIOLATION


class SanitizationError(KleosDataError):
    pass


class VaultError(KleosDataError):
    pass


class ReviewError(KleosDataError):
    pass


class ReviewIntegrityError(ReviewError):
    pass


class GateFailure(KleosDataError):
    exit_code = EXIT_GATE_FAILED


class PromotionIntegrityError(KleosDataError):
    pass


class ReleaseImmutabilityError(KleosDataError):
    pass


class VerificationFailure(KleosDataError):
    exit_code = EXIT_GATE_FAILED
