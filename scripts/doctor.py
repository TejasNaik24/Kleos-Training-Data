#!/usr/bin/env python3
"""Check the environment without printing a single secret value.

    python scripts/doctor.py
    python scripts/doctor.py --check-network      # only if you ask for it

Reports whether each credential is *present*, never what it is. That is the whole
point: the moment a diagnostic prints a token, it becomes the thing people paste
into an issue.

Touches the network only with ``--check-network``. A diagnostic that silently
reaches a backend is a diagnostic that can leak which deployment you are pointed
at, and it makes the command unusable on a plane.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import add_common_arguments, print_header, print_result, run, setup_logging
from kleos_training_data.collection.guard import ENV_ALLOW, is_local
from kleos_training_data.contract.compat import ENV_MODELS_PATH, kleos_models_available
from kleos_training_data.contract.constants import (
    DATASET_SCHEMA_VERSION,
    PIPELINE_VERSION,
    SUPPORTED_TASKS,
)
from kleos_training_data.contract.pin import CONTRACT_SOURCE_COMMIT
from kleos_training_data.errors import EXIT_ERROR, EXIT_OK
from kleos_training_data.logging_utils import SafeSecret
from kleos_training_data.paths import Workspace
from kleos_training_data.privacy.rules import RULESET_VERSION
from kleos_training_data.review.rubric import RUBRIC_VERSION

#: Credentials to report on. Only ever presence, never value.
SECRET_VARS: tuple[str, ...] = ("KLEOS_API_TOKEN", "ANTHROPIC_API_KEY")

#: Non-secret settings, safe to echo.
SETTING_VARS: tuple[str, ...] = (
    "KLEOS_BACKEND_URL",
    "KLEOS_TRAINING_DATA_ENV",
    "KLEOS_DATA_LOG_LEVEL",
    ENV_MODELS_PATH,
)

MIN_PYTHON = (3, 11)


def _row(label: str, ok: bool | None, detail: str = "") -> str:
    icon = {True: "✓", False: "✗", None: "·"}[ok]
    return f"  {icon} {label:<26} {detail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check-network",
        action="store_true",
        help="Probe the configured backend. Off by default, deliberately.",
    )
    parser.add_argument("--workspace", type=Path, help="Workspace root.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("KLEOS TRAINING DATA DOCTOR")
    problems: list[str] = []

    # --- runtime -----------------------------------------------------------
    print("\n  Runtime")
    version_ok = sys.version_info >= MIN_PYTHON
    print(_row("python", version_ok, f"{sys.version_info.major}.{sys.version_info.minor}"))
    if not version_ok:
        problems.append(f"python {'.'.join(map(str, MIN_PYTHON))}+ is required")

    for module, extra in (
        ("pydantic", None),
        ("yaml", None),
        ("jsonschema", None),
        ("httpx", "collect"),
        ("anthropic", "review"),
        ("kleos_models", "compat"),
    ):
        import importlib.util

        present = importlib.util.find_spec(module) is not None
        if extra is None:
            print(_row(module, present, "required"))
            if not present:
                problems.append(f"{module} is missing; run: make install")
        else:
            print(
                _row(
                    module,
                    None if not present else True,
                    f"optional ({extra} extra)" + ("" if present else " — not installed"),
                )
            )

    # --- versions ----------------------------------------------------------
    print("\n  Versions")
    print(_row("pipeline", True, PIPELINE_VERSION))
    print(_row("dataset schema", True, DATASET_SCHEMA_VERSION))
    print(_row("privacy ruleset", True, RULESET_VERSION))
    print(_row("review rubric", True, RUBRIC_VERSION))
    print(_row("pinned contract", True, CONTRACT_SOURCE_COMMIT[:12]))
    print(_row("registered tasks", True, str(len(SUPPORTED_TASKS))))

    # --- secrets -----------------------------------------------------------
    print("\n  Secrets (values are never printed)")
    for name in SECRET_VARS:
        secret = SafeSecret(os.environ.get(name, ""), label=name)
        print(
            _row(
                name,
                None if not secret.is_set else True,
                str(secret) if secret.is_set else "not set (fine for the offline pipeline)",
            )
        )

    print("\n  Settings")
    for name in SETTING_VARS:
        value = os.environ.get(name, "")
        print(_row(name, None if not value else True, value or "not set"))

    # --- production guard --------------------------------------------------
    print("\n  Capture safety")
    backend = os.environ.get("KLEOS_BACKEND_URL", "mock://local")
    local = is_local(backend)
    print(_row("configured backend", True, "local or mock" if local else "NON-LOCAL"))
    allow = os.environ.get(ENV_ALLOW, "").strip() == "1"
    print(
        _row(
            "production capture",
            None if not allow else False,
            "DISABLED" if not allow else f"{ENV_ALLOW}=1 is set — one of four conditions",
        )
    )
    if allow and not local:
        print(
            "      ! A non-local backend and the environment flag are both set.\n"
            "        Two more conditions still gate a capture, and anything it\n"
            "        produces can never be promoted."
        )
    print(
        _row(
            "running in CI",
            None if not os.environ.get("CI") else True,
            "yes — production capture is refused unconditionally" if os.environ.get("CI") else "no",
        )
    )

    # --- workspace ---------------------------------------------------------
    print("\n  Workspace")
    workspace = Workspace.from_env(args.workspace)
    missing = workspace.missing_zones()
    print(
        _row(
            "zones",
            not missing,
            "all present"
            if not missing
            else f"{len(missing)} missing — run scripts/init_workspace.py",
        )
    )
    if missing:
        problems.append("workspace zones are missing")
    if workspace.vault.is_dir():
        secure = workspace.vault_permissions_ok()
        print(
            _row("vault permissions", secure, "owner-only" if secure else "GROUP OR WORLD READABLE")
        )
        if not secure:
            problems.append("the vault is readable beyond its owner")

    releases = sorted(p.name for p in workspace.releases.glob("*") if p.is_dir())
    print(_row("releases", True, ", ".join(releases) if releases else "none yet"))

    # --- git ---------------------------------------------------------------
    print("\n  Git")
    if shutil.which("git") is None:
        print(_row("git", None, "not installed — ignore rules cannot be verified"))
    else:
        inside = (
            subprocess.run(
                ["git", "rev-parse", "--git-dir"], cwd=workspace.root, capture_output=True
            ).returncode
            == 0
        )
        print(
            _row(
                "repository",
                None if not inside else True,
                "initialized" if inside else "not a git repository yet",
            )
        )
        if inside:
            hook = workspace.root / ".git" / "hooks" / "pre-commit"
            print(
                _row(
                    "pre-commit hook",
                    None if not hook.exists() else True,
                    "installed"
                    if hook.exists()
                    else "not installed — run check_no_private_data.py --install-hook",
                )
            )

    # --- compatibility -----------------------------------------------------
    print("\n  Compatibility")
    available = kleos_models_available()
    print(
        _row(
            "kleos-models",
            None if not available else True,
            "importable" if available else "absent (fine — the offline pipeline does not need it)",
        )
    )
    print(
        _row(
            "verification",
            None if not available else True,
            "run: make compat" if available else "would SKIP; CI runs it with --strict",
        )
    )

    # --- network -----------------------------------------------------------
    print("\n  Network")
    if not args.check_network:
        print(_row("backend probe", None, "skipped — pass --check-network to run it"))
    else:
        print(_row("backend probe", None, f"probing {backend} …"))
        try:
            from kleos_training_data.collection.transport import require_httpx

            httpx = require_httpx()
            response = httpx.get(backend, timeout=5.0)
            print(_row("backend probe", True, f"HTTP {response.status_code}"))
        except Exception as exc:
            print(_row("backend probe", False, type(exc).__name__))

    if problems:
        print_result(False, f"{len(problems)} problem(s) found.", hint="; ".join(problems))
        return EXIT_ERROR

    print_result(
        True, "Environment looks healthy.", hint="Prove the pipeline: make slice-clean slice"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(run(main))
