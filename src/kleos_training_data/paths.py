"""The three-zone workspace layout.

A reviewer must be able to tell, from the filesystem alone, whether an artifact
is raw and potentially sensitive, reviewed but not yet promoted, or trusted final
training data. That is what these directories are for, and why the boundaries
between them are types rather than string concatenation at call sites.

::

    staging/     UNVERIFIED — raw captures through approved candidates
      raw/         adapter output, never trusted, never committed
      normalized/  contract-shaped, pre-sanitization
      sanitized/   post-redaction candidates + their privacy results
      review/      reviewer packets, LLM reviews, signed human decisions
      rejected/    rejection records with closed-vocabulary reason codes
      promoted/    passed all 14 gates; the pool a release draws from

    vault/       RE-IDENTIFICATION KEYS — entity vault and surrogate maps
                 chmod 700. Needed to re-derive a release, never to consume one.

    releases/    IMMUTABLE PRODUCT — sealed, read-only dataset versions

    reports/     Coverage, dedup and leakage reports. May quote candidate text,
                 so it is git-ignored like the zones above.

Every one of these is deny-by-default in ``.gitignore`` and guarded by
``FORBIDDEN_PATHS`` in ``scripts/check_no_private_data.py``.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

from kleos_training_data.errors import WorkspaceError

#: Subdirectories of ``staging/``, in pipeline order. Order is meaningful: it is
#: the order ``init_workspace.py`` reports them in, and the order a candidate
#: moves through.
STAGING_SUBDIRS: tuple[str, ...] = (
    "raw",
    "normalized",
    "sanitized",
    "review",
    "rejected",
    "promoted",
)

#: Subdirectories of ``staging/review/``.
REVIEW_SUBDIRS: tuple[str, ...] = ("packets", "llm", "decisions")

#: Subdirectories of ``vault/``.
VAULT_SUBDIRS: tuple[str, ...] = ("surrogate_maps",)

#: Subdirectories of ``reports/``.
REPORT_SUBDIRS: tuple[str, ...] = ("coverage", "leakage", "dedup", "reviews", "gates")

#: Mode for the vault. The keys that map a surrogate back to a real person live
#: here; nothing else in the workspace is as directly re-identifying.
VAULT_MODE = 0o700

#: Environment variables that relocate each zone, so a workspace can live on an
#: encrypted volume without editing code.
ENV_ROOT = "KLEOS_TRAINING_DATA_ROOT"
ENV_STAGING = "KLEOS_STAGING_DIR"
ENV_VAULT = "KLEOS_VAULT_DIR"
ENV_RELEASES = "KLEOS_RELEASES_DIR"
ENV_REPORTS = "KLEOS_REPORTS_DIR"


def repo_root() -> Path:
    """The repository root, inferred from this file's location."""
    return Path(__file__).resolve().parent.parent.parent


def _resolve(root: Path, env_var: str, default: str) -> Path:
    """Resolve one zone, honouring an absolute or root-relative override."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return root / default
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


@dataclass(frozen=True)
class Workspace:
    """Typed accessors for the three zones.

    Construct with :meth:`from_env` rather than by hand — that is what applies
    the environment overrides consistently across every script.
    """

    root: Path
    staging: Path
    vault: Path
    releases: Path
    reports: Path

    @classmethod
    def from_env(cls, root: Path | str | None = None) -> Workspace:
        """Build a workspace from the environment, defaulting to the repo root."""
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

    # --- staging -----------------------------------------------------------

    def raw_batch(self, batch_id: str) -> Path:
        """Directory holding one capture batch's raw records."""
        return self.staging / "raw" / batch_id

    def normalized_batch(self, batch_id: str) -> Path:
        """Directory holding one batch's normalized candidates."""
        return self.staging / "normalized" / batch_id

    def sanitized_batch(self, batch_id: str) -> Path:
        """Directory holding one batch's sanitized candidates and privacy results."""
        return self.staging / "sanitized" / batch_id

    def review_packet(self, packet_id: str) -> Path:
        """Directory holding one reviewer packet."""
        return self.staging / "review" / "packets" / packet_id

    def llm_review(self, candidate_id: str) -> Path:
        """Path to one machine review record."""
        return self.staging / "review" / "llm" / f"{candidate_id}.json"

    def human_decision(self, candidate_id: str) -> Path:
        """Path to one signed human decision record."""
        return self.staging / "review" / "decisions" / f"{candidate_id}.json"

    def rejection(self, candidate_id: str) -> Path:
        """Path to one rejection record."""
        return self.staging / "rejected" / f"{candidate_id}.json"

    def promoted(self, candidate_id: str) -> Path:
        """Path to one promoted example plus its audit block."""
        return self.staging / "promoted" / f"{candidate_id}.json"

    @property
    def promoted_index(self) -> Path:
        """Append-only ledger of every promotion, in order."""
        return self.staging / "promoted" / "index.jsonl"

    def gate_report(self, run_id: str) -> Path:
        """Path to one promotion run's full gate report."""
        return self.staging / "promoted" / "gate_reports" / f"{run_id}.json"

    # --- vault -------------------------------------------------------------

    @property
    def entity_vault(self) -> Path:
        """Operator-supplied literals mapped to placeholder slots."""
        return self.vault / "entity_vault.json"

    def surrogate_map(self, scenario_family: str) -> Path:
        """Placeholder-to-surrogate map for one scenario family.

        Keyed on family, not globally: that is what makes the same real person
        appear as different fictional people in different scenarios, so there is
        no cross-example entity for a model to memorize.
        """
        safe = scenario_family.replace("/", "_")
        return self.vault / "surrogate_maps" / f"{safe}.json"

    # --- releases ----------------------------------------------------------

    def release(self, version: str) -> Path:
        """Directory for one sealed dataset version."""
        return self.releases / version

    def release_staging(self, token: str) -> Path:
        """Scratch directory a release is assembled in before being sealed.

        Sealing is an ``os.replace`` of this directory into place, so a crash
        mid-write can never leave a partial release visible under its version
        name.
        """
        return self.releases / f".staging-{token}"

    # --- checks ------------------------------------------------------------

    def all_zones(self) -> tuple[Path, ...]:
        """Every top-level zone, for iteration in checks and reports."""
        return (self.staging, self.vault, self.releases, self.reports)

    def missing_zones(self) -> list[Path]:
        """Zones that do not exist yet."""
        return [path for path in self.all_zones() if not path.is_dir()]

    def assert_initialized(self) -> None:
        """Raise unless every zone exists.

        Raises:
            WorkspaceError: If any zone is missing.
        """
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
        """Whether the vault is not readable or writable by group or other."""
        if not self.vault.is_dir():
            return False
        mode = stat.S_IMODE(self.vault.stat().st_mode)
        return not (mode & 0o077)

    def assert_vault_secure(self) -> None:
        """Raise if the vault is group- or world-accessible.

        Permissions are a speed bump rather than a guarantee — anyone who can
        read the repo as your user can read the vault. The check is here because
        a vault that became world-readable is a signal that something copied or
        recreated it outside the pipeline.

        Raises:
            WorkspaceError: If the vault is missing or too permissive.
        """
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
