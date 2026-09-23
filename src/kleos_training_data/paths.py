from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

from kleos_training_data.errors import WorkspaceError

STAGING_SUBDIRS: tuple[str, ...] = (
    "raw",
    "normalized",
    "sanitized",
    "review",
    "rejected",
    "promoted",
)

REVIEW_SUBDIRS: tuple[str, ...] = ("packets", "llm", "decisions")

VAULT_SUBDIRS: tuple[str, ...] = ("surrogate_maps",)

REPORT_SUBDIRS: tuple[str, ...] = ("coverage", "leakage", "dedup", "reviews", "gates")

VAULT_MODE = 0o700

ENV_ROOT = "KLEOS_TRAINING_DATA_ROOT"
ENV_STAGING = "KLEOS_STAGING_DIR"
ENV_VAULT = "KLEOS_VAULT_DIR"
ENV_RELEASES = "KLEOS_RELEASES_DIR"
ENV_REPORTS = "KLEOS_REPORTS_DIR"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _resolve(root: Path, env_var: str, default: str) -> Path:
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return root / default
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


@dataclass(frozen=True)
class Workspace:
    root: Path
    staging: Path
    vault: Path
    releases: Path
    reports: Path

    @classmethod
    def from_env(cls, root: Path | str | None = None) -> Workspace:
        base = Path(root).expanduser().resolve() if root is not None else repo_root()
        env_root = os.environ.get(ENV_ROOT, "").strip()
        if root is None and env_root:
            base = Path(env_root).expanduser().resolve()
        return cls(
            root=base,
            staging=_resolve(base, ENV_STAGING, "staging"),
            vault=_resolve(base, ENV_VAULT, "vault"),
            releases=_resolve(base, ENV_RELEASES, "releases"),
            reports=_resolve(base, ENV_REPORTS, "reports"),
        )

    def raw_batch(self, batch_id: str) -> Path:
        return self.staging / "raw" / batch_id

    def normalized_batch(self, batch_id: str) -> Path:
        return self.staging / "normalized" / batch_id

    def sanitized_batch(self, batch_id: str) -> Path:
        return self.staging / "sanitized" / batch_id

    def review_packet(self, packet_id: str) -> Path:
        return self.staging / "review" / "packets" / packet_id

    def llm_review(self, candidate_id: str) -> Path:
        return self.staging / "review" / "llm" / f"{candidate_id}.json"

    def human_decision(self, candidate_id: str) -> Path:
        return self.staging / "review" / "decisions" / f"{candidate_id}.json"

    def rejection(self, candidate_id: str) -> Path:
        return self.staging / "rejected" / f"{candidate_id}.json"

    def promoted(self, candidate_id: str) -> Path:
        return self.staging / "promoted" / f"{candidate_id}.json"

    @property
    def promoted_index(self) -> Path:
        return self.staging / "promoted" / "index.jsonl"

    def gate_report(self, run_id: str) -> Path:
        return self.staging / "promoted" / "gate_reports" / f"{run_id}.json"

    @property
    def entity_vault(self) -> Path:
        return self.vault / "entity_vault.json"

    def surrogate_map(self, scenario_family: str) -> Path:
        safe = scenario_family.replace("/", "_")
        return self.vault / "surrogate_maps" / f"{safe}.json"

    def release(self, version: str) -> Path:
        return self.releases / version

    def release_staging(self, token: str) -> Path:
        return self.releases / f".staging-{token}"

    def all_zones(self) -> tuple[Path, ...]:
        return (self.staging, self.vault, self.releases, self.reports)

    def missing_zones(self) -> list[Path]:
        return [path for path in self.all_zones() if not path.is_dir()]

    def assert_initialized(self) -> None:
        missing = self.missing_zones()
        if missing:
            raise WorkspaceError(
                f"{len(missing)} workspace directory/directories are missing.",
                details={"missing": ", ".join(str(p) for p in missing)},
                suggestions=[
                    "Create them: python scripts/init_workspace.py",
                    "If you relocated a zone, check KLEOS_STAGING_DIR / KLEOS_VAULT_DIR "
                    "/ KLEOS_RELEASES_DIR / KLEOS_REPORTS_DIR.",
                ],
            )

    def vault_permissions_ok(self) -> bool:
        if not self.vault.is_dir():
            return False
        mode = stat.S_IMODE(self.vault.stat().st_mode)
        return not (mode & 0o077)

    def assert_vault_secure(self) -> None:
        if not self.vault.is_dir():
            raise WorkspaceError(
                f"The vault directory {self.vault} does not exist.",
                suggestions=["Create it: python scripts/init_workspace.py"],
            )
        if not self.vault_permissions_ok():
            mode = stat.S_IMODE(self.vault.stat().st_mode)
            raise WorkspaceError(
                f"The vault {self.vault} is accessible beyond its owner.",
                details={"mode": f"0o{mode:03o}", "expected": f"0o{VAULT_MODE:03o}"},
                suggestions=[
                    f"Restrict it: chmod {VAULT_MODE:o} {self.vault}",
                    "The vault holds the maps from surrogate names back to real "
                    "people. Treat a permissions change as an incident until you "
                    "know what caused it — see docs/incident-response.md.",
                ],
            )
