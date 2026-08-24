"""Structured logging that cannot leak conversation content.

The public repo's logging module has the same shape as this one. This version is
hardened in two places, because this repository actually holds private data:

1. :func:`redact` refuses to show a content preview unless the caller explicitly
   asks for one. In the public repo ``keep=8`` is a convenience; here it is a
   decision, and it has to look like one at the call site.

2. :class:`SafeSecret` wraps credentials so that ``str``, ``repr``, f-strings,
   ``%``-formatting and ``json.dumps(default=str)`` all render a digest instead
   of the value. Nothing reveals a token by accident — you have to call
   ``.reveal()``, and a test asserts that happens in exactly one non-test file.

**Privacy rule:** no logger in this repository ever writes message content,
request bodies or response bodies — not at DEBUG, not behind a flag. If you need
to look at actual candidate text, open the reviewer packet, which exists for
exactly that and is written to a git-ignored directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from kleos_training_data.errors import PrivacyViolationError

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-32s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Root logger name, so repeated configuration is idempotent.
_ROOT_LOGGER_NAME = "kleos_training_data"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a package logger.

    Args:
        name: Dotted suffix, typically ``__name__``. ``None`` returns the root
            package logger.
    """
    if name is None or name == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name.startswith(_ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def configure_logging(
    *,
    level: int | str = logging.INFO,
    log_file: Path | str | None = None,
    quiet_libraries: bool = True,
) -> logging.Logger:
    """Configure the package logger. Safe to call more than once.

    Args:
        level: Threshold for the console handler.
        log_file: Optional path receiving the same records at DEBUG level.
        quiet_libraries: Suppress noisy third-party INFO logs.

    Returns:
        The configured root package logger.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Remove handlers we installed previously so re-configuration does not
    # duplicate every line.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level if isinstance(level, int) else str(level).upper())
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if quiet_libraries:
        for noisy in ("httpx", "httpcore", "urllib3", "anthropic", "filelock"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger


def resolve_level(verbose: bool = False, quiet: bool = False) -> int:
    """Map CLI verbosity flags (and ``KLEOS_DATA_LOG_LEVEL``) to a logging level."""
    env = os.environ.get("KLEOS_DATA_LOG_LEVEL")
    if env:
        resolved = logging.getLevelName(env.upper())
        if isinstance(resolved, int):
            return resolved
    if quiet:
        return logging.WARNING
    if verbose:
        return logging.DEBUG
    return logging.INFO


# ---------------------------------------------------------------------------
# Privacy-preserving helpers
# ---------------------------------------------------------------------------


def redact(text: str, *, keep: int = 0, allow_preview: bool = False) -> str:
    """Render text as a length and a digest instead of its content.

    Args:
        text: Content that may be private. Never rendered verbatim unless the
            caller explicitly opts in.
        keep: Number of leading characters to retain in the preview.
        allow_preview: Required to be ``True`` whenever ``keep > 0``.

    Returns:
        A string such as ``<redacted len=412 sha256=1a2b3c4d>``.

    Raises:
        PrivacyViolationError: If ``keep > 0`` without ``allow_preview=True``.

    The extra flag exists because ``keep=20`` is the kind of thing that gets
    added while debugging and left behind. Requiring a second, unambiguously
    named argument makes it visible in review and greppable afterwards.
    """
    if keep > 0 and not allow_preview:
        raise PrivacyViolationError(
            "redact(keep>0) shows real content and needs allow_preview=True.",
            details={"requested_keep": keep},
            suggestions=[
                "Drop `keep` — the length and digest are usually enough to correlate records.",
                "If you genuinely need a preview, pass allow_preview=True so the "
                "decision is visible at the call site and in review.",
                "To read candidate text, open the reviewer packet instead of the log.",
            ],
        )
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    prefix = ""
    if keep > 0:
        head = text[:keep].replace("\n", " ")
        prefix = f"{head!r}… "
    return f"{prefix}<redacted len={len(text)} sha256={digest}>"


class SafeSecret:
    """A credential that does not render itself.

    ``str``, ``repr``, f-strings, ``%``-formatting and ``json.dumps(default=str)``
    all produce ``<secret len=64 sha256=1a2b3c4d>``. The value is reachable only
    through :meth:`reveal`.

    This is not a security boundary — anything in the process can read the
    attribute. It is a *mistake* boundary: the way a token reaches a log is
    almost never a deliberate print, it is an exception message, a repr of a
    config object, or a dict dumped for debugging. Those paths are exactly the
    ones this closes.
    """

    __slots__ = ("_label", "_value")

    def __init__(self, value: str, *, label: str = "secret") -> None:
        self._value = value
        self._label = label

    def reveal(self) -> str:
        """Return the raw value. Call this at the point of use, never earlier."""
        return self._value

    @property
    def is_set(self) -> bool:
        """Whether a non-empty value is present."""
        return bool(self._value)

    @property
    def fingerprint(self) -> str:
        """Short digest, stable for a given value. Safe to log and compare."""
        if not self._value:
            return "unset"
        return hashlib.sha256(self._value.encode("utf-8")).hexdigest()[:8]

    def __str__(self) -> str:
        if not self._value:
            return f"<{self._label} unset>"
        return f"<{self._label} len={len(self._value)} sha256={self.fingerprint}>"

    __repr__ = __str__

    def __format__(self, spec: str) -> str:
        # Ignore the format spec entirely. `f"{token:>40}"` must not become a
        # padded secret, and `f"{token!r}"` routes through __repr__ above.
        return str(self)

    def __bool__(self) -> bool:
        return bool(self._value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SafeSecret):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)


# ---------------------------------------------------------------------------
# Structured JSONL event stream
# ---------------------------------------------------------------------------


class EventLogger:
    """Append-only JSONL event writer for machine-readable pipeline history.

    Each record carries ``ts``, ``run_id``, ``stage`` and ``event`` plus whatever
    structured fields the caller supplies. Values pass through ``json.dumps``
    with ``default=str`` so unusual types degrade to their repr rather than
    raising mid-run — which is also why :class:`SafeSecret` overrides ``__str__``
    rather than only ``__repr__``.
    """

    def __init__(self, path: Path | str, *, run_id: str, stage: str = "init") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.stage = stage
        self._logger = get_logger("events")

    def set_stage(self, stage: str) -> None:
        """Set the stage label attached to subsequent events."""
        self.stage = stage

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        """Write one event record.

        Never raises: an audit-log failure must not abort a pipeline run and lose
        the work that was already done.
        """
        record: dict[str, Any] = {
            "ts": time.time(),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "run_id": self.run_id,
            "stage": self.stage,
            "event": event,
        }
        record.update(fields)
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:  # pragma: no cover - filesystem dependent
            self._logger.warning("Could not write event log to %s: %s", self.path, exc)
        return record

    def read_all(self) -> list[dict[str, Any]]:
        """Read every event back. Malformed lines are skipped, not fatal."""
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events


@contextmanager
def log_stage(logger: logging.Logger, stage: str) -> Iterator[None]:
    """Log entry/exit of a pipeline stage with elapsed wall time."""
    logger.info("▶ %s", stage)
    started = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - started
        logger.error("✗ %s failed after %.1fs", stage, elapsed)
        raise
    else:
        elapsed = time.perf_counter() - started
        logger.info("✓ %s (%.1fs)", stage, elapsed)


def format_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    """Render rows as a fixed-width text table for console reports.

    Args:
        rows: Records to display.
        columns: Column order. Defaults to the keys of the first row.
    """
    if not rows:
        return "(no rows)"
    cols = columns or list(rows[0].keys())
    widths = {c: len(c) for c in cols}
    rendered: list[dict[str, str]] = []
    for row in rows:
        cells = {}
        for col in cols:
            value = row.get(col, "")
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            cells[col] = text
            widths[col] = max(widths[col], len(text))
        rendered.append(cells)

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    divider = "  ".join("-" * widths[c] for c in cols)
    body = "\n".join("  ".join(cells[c].ljust(widths[c]) for c in cols) for cells in rendered)
    return f"{header}\n{divider}\n{body}"
