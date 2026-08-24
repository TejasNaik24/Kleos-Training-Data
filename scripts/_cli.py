"""Shared plumbing for the CLI scripts.

Keeps every script thin: argument parsing plus a call into the library. No
pipeline logic lives in ``scripts/`` — that would make it untestable.

Exit-code contract, uniform across every script:

===  ==========================================================================
  0  Success
  1  Domain error — the tool could not do the job
  2  Usage error — argparse rejected the arguments
  3  Gate or verification failure — the tool worked, the data is not acceptable
  4  Privacy violation — private data was found on a path it must not take
130  Interrupted
===  ==========================================================================

The distinction between 1 and 3 is load-bearing. CI must be able to tell "the
data is bad" from "the tool crashed" without reading logs, because those need
different humans. 4 is carved out of 3 for the same reason: a privacy failure is
not a data-quality failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parent.parent

# Allow running the scripts directly from a checkout without installing.
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from kleos_training_data.errors import (
    EXIT_ERROR,
    EXIT_INTERRUPTED,
    KleosDataError,
)
from kleos_training_data.logging_utils import configure_logging, get_logger, resolve_level


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add flags every script accepts."""
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Warnings and errors only.")
    return parser


def add_config_arguments(
    parser: argparse.ArgumentParser, *, required: bool = False
) -> argparse.ArgumentParser:
    """Add the pipeline-config flags shared by the multi-stage scripts.

    Args:
        parser: Parser to extend.
        required: Whether ``--config`` must be supplied. Most scripts have
            working defaults and leave this ``False``.
    """
    parser.add_argument(
        "--config",
        type=Path,
        required=required,
        help="Pipeline config YAML (default: configs/pipeline.yaml).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set collection.concurrency=1",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Workspace root, overriding the repository root and the environment.",
    )
    return parser


def add_batch_argument(
    parser: argparse.ArgumentParser, *, required: bool = True
) -> argparse.ArgumentParser:
    """Add the ``--batch`` flag used by every staging-stage script."""
    parser.add_argument(
        "--batch",
        required=required,
        metavar="ID",
        help="Capture batch identifier, e.g. slice-001.",
    )
    return parser


def setup_logging(args: argparse.Namespace, *, log_file: Path | None = None) -> None:
    """Configure logging from parsed arguments."""
    configure_logging(
        level=resolve_level(
            verbose=getattr(args, "verbose", False), quiet=getattr(args, "quiet", False)
        ),
        log_file=log_file,
    )


def fail(error: BaseException, *, exit_code: int = EXIT_ERROR) -> NoReturn:
    """Print an actionable error and exit.

    ``KleosDataError`` already renders its own diagnostics and suggestions, so it
    is shown verbatim rather than wrapped in a traceback the user cannot act on.
    """
    logger = get_logger("cli")
    if isinstance(error, KleosDataError):
        print(f"\n✗ {error.render()}\n", file=sys.stderr)
    else:
        logger.exception("Unexpected error")
        print(f"\n✗ {type(error).__name__}: {error}\n", file=sys.stderr)
        print(
            "This looks like a bug in the pipeline, not a problem with your data.", file=sys.stderr
        )
        print("Re-run with -v for a full traceback.\n", file=sys.stderr)
    raise SystemExit(exit_code)


def run(main_fn, argv: list[str] | None = None) -> int:
    """Invoke a script entry point with uniform error handling.

    The exit code comes from the exception's own ``exit_code`` attribute rather
    than from a chain of ``except`` arms here. A new error type therefore gets
    its code from its own definition, and cannot silently fall through to 1
    because someone forgot to add an arm.
    """
    try:
        return main_fn(argv)
    except KeyboardInterrupt:
        print("\n✗ Interrupted.\n", file=sys.stderr)
        return EXIT_INTERRUPTED
    except KleosDataError as exc:
        fail(exc, exit_code=exc.exit_code)
    except Exception as exc:
        fail(exc, exit_code=EXIT_ERROR)


def print_header(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_result(ok: bool, message: str, *, hint: str | None = None) -> None:
    """Print a uniform pass/fail footer.

    Failures go to stderr so a caller can separate the verdict from the report
    body without parsing it.
    """
    print()
    if ok:
        print(f"✓ {message}")
        if hint:
            print(f"  {hint}")
        print()
    else:
        print(f"✗ {message}", file=sys.stderr)
        if hint:
            print(f"  {hint}", file=sys.stderr)
        print(file=sys.stderr)
