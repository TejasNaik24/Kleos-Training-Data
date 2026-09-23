from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from kleos_training_data.errors import (
    EXIT_ERROR,
    EXIT_INTERRUPTED,
    KleosDataError,
)
from kleos_training_data.logging_utils import configure_logging, get_logger, resolve_level


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Warnings and errors only.")
    return parser


def add_config_arguments(
    parser: argparse.ArgumentParser, *, required: bool = False
) -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--batch",
        required=required,
        metavar="ID",
        help="Capture batch identifier, e.g. slice-001.",
    )
    return parser


def setup_logging(args: argparse.Namespace, *, log_file: Path | None = None) -> None:
    configure_logging(
        level=resolve_level(
            verbose=getattr(args, "verbose", False), quiet=getattr(args, "quiet", False)
        ),
        log_file=log_file,
    )


def fail(error: BaseException, *, exit_code: int = EXIT_ERROR) -> NoReturn:
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
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_result(ok: bool, message: str, *, hint: str | None = None) -> None:
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
