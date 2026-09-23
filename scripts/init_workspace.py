from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import add_common_arguments, print_header, print_result, run, setup_logging
from kleos_training_data.errors import EXIT_OK, EXIT_PRIVACY_VIOLATION, PrivacyViolationError
from kleos_training_data.logging_utils import get_logger
from kleos_training_data.paths import (
    REPORT_SUBDIRS,
    REVIEW_SUBDIRS,
    STAGING_SUBDIRS,
    VAULT_MODE,
    VAULT_SUBDIRS,
    Workspace,
)

logger = get_logger(__name__)

IGNORE_PROBES: tuple[str, ...] = (
    "staging/raw/__probe__.json",
    "staging/promoted/index.jsonl",
    "vault/entity_vault.json",
    "vault/surrogate_maps/__probe__.json",
    "releases/kleos-policy-v0.0.0/train.jsonl",
    "releases/kleos-policy-v0.0.0/manifest.json",
    "reports/coverage/__probe__.md",
)

TRACK_PROBES: tuple[str, ...] = (
    "staging/.gitkeep",
    "vault/.gitkeep",
    "releases/.gitkeep",
    "reports/.gitkeep",
)


def _git_available(root: Path) -> bool:
    if shutil.which("git") is None:
        return False
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=root,
            capture_output=True,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return False
    return True


def _is_ignored(root: Path, relative: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", relative],
        cwd=root,
        capture_output=True,
    )
    return result.returncode == 0


def create_directories(workspace: Workspace) -> list[Path]:
    created: list[Path] = []

    targets: list[Path] = [workspace.staging]
    targets += [workspace.staging / name for name in STAGING_SUBDIRS]
    targets += [workspace.staging / "review" / name for name in REVIEW_SUBDIRS]
    targets += [workspace.staging / "promoted" / "gate_reports"]
    targets.append(workspace.vault)
    targets += [workspace.vault / name for name in VAULT_SUBDIRS]
    targets.append(workspace.releases)
    targets.append(workspace.reports)
    targets += [workspace.reports / name for name in REPORT_SUBDIRS]

    for path in targets:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)

    for zone in workspace.all_zones():
        keep = zone / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")
            created.append(keep)

    return created


def secure_vault(workspace: Workspace) -> bool:
    if workspace.vault_permissions_ok():
        return False
    os.chmod(workspace.vault, VAULT_MODE)
    return True


def verify_ignored(workspace: Workspace) -> tuple[list[str], list[str]]:
    root = workspace.root
    leaks = [p for p in IGNORE_PROBES if not _is_ignored(root, p)]
    unreachable = [p for p in TRACK_PROBES if _is_ignored(root, p)]
    return leaks, unreachable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, help="Workspace root (default: repository root).")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the workspace and ignore rules without creating anything.",
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    workspace = Workspace.from_env(args.root)
    print_header("KLEOS TRAINING DATA — WORKSPACE")

    print(f"\n  root     : {workspace.root}")
    print(f"  staging  : {workspace.staging}")
    print(f"  vault    : {workspace.vault}")
    print(f"  releases : {workspace.releases}")
    print(f"  reports  : {workspace.reports}")

    if args.check:
        missing = workspace.missing_zones()
        print(f"\n  missing zones : {len(missing)}")
        for path in missing:
            print(f"    ✗ {path}")
    else:
        created = create_directories(workspace)
        print(f"\n  created  : {len(created)} path(s)")
        for path in created[:12]:
            print(f"    + {path.relative_to(workspace.root)}")
        if len(created) > 12:
            print(f"    … {len(created) - 12} more")

        if secure_vault(workspace):
            print(f"\n  vault mode set to 0o{VAULT_MODE:03o}")

    if not _git_available(workspace.root):
        print("\n  ! git is unavailable or this is not a repository yet.")
        print("    Ignore rules could NOT be verified. Run `git init` and re-run")
        print("    this script before capturing anything.")
        print_result(True, "Workspace ready (ignore rules unverified).")
        return EXIT_OK

    leaks, unreachable = verify_ignored(workspace)

    print(f"\n  ignore probes : {len(IGNORE_PROBES)} checked, {len(leaks)} would be committable")
    print(f"  keep probes   : {len(TRACK_PROBES)} checked, {len(unreachable)} unreachable")

    if unreachable:
        print("\n  ! These must stay trackable or the zones vanish from a clone:")
        for path in unreachable:
            print(f"      {path}")

    if leaks:
        raise PrivacyViolationError(
            f"{len(leaks)} workspace path(s) are NOT ignored by git.",
            details={"committable": ", ".join(leaks)},
            suggestions=[
                "Check .gitignore uses `staging/*` and not `staging/` — git cannot "
                "re-include a file whose parent directory is excluded.",
                "Confirm no later rule re-includes these paths.",
                "Verify by hand: git check-ignore -v " + leaks[0],
                "Do not capture anything until this is fixed. A raw capture is one "
                "person's private data, and git history is forever.",
            ],
        )

    if unreachable:
        print_result(
            False,
            "Workspace directories would not survive a clone.",
            hint="Add `!<zone>/.gitkeep` after the `<zone>/*` rule in .gitignore.",
        )
        return EXIT_PRIVACY_VIOLATION

    print_result(
        True,
        "Workspace ready. Every zone is ignored by git.",
        hint="Next: python scripts/validate_scenarios.py --strict",
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(run(main))
