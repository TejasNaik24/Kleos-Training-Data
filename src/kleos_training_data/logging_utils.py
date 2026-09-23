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

_ROOT_LOGGER_NAME = "kleos_training_data"


def get_logger(name: str | None = None) -> logging.Logger:
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
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

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


def redact(text: str, *, keep: int = 0, allow_preview: bool = False) -> str:
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
    __slots__ = ("_label", "_value")

    def __init__(self, value: str, *, label: str = "secret") -> None:
        self._value = value
        self._label = label

    def reveal(self) -> str:
        return self._value

    @property
    def is_set(self) -> bool:
        return bool(self._value)

    @property
    def fingerprint(self) -> str:
        if not self._value:
            return "unset"
        return hashlib.sha256(self._value.encode("utf-8")).hexdigest()[:8]

    def __str__(self) -> str:
        if not self._value:
            return f"<{self._label} unset>"
        return f"<{self._label} len={len(self._value)} sha256={self.fingerprint}>"

    __repr__ = __str__

    def __format__(self, spec: str) -> str:
        return str(self)

    def __bool__(self) -> bool:
        return bool(self._value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SafeSecret):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)


class EventLogger:
    def __init__(self, path: Path | str, *, run_id: str, stage: str = "init") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.stage = stage
        self._logger = get_logger("events")

    def set_stage(self, stage: str) -> None:
        self.stage = stage

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
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
        except OSError as exc:
            self._logger.warning("Could not write event log to %s: %s", self.path, exc)
        return record

    def read_all(self) -> list[dict[str, Any]]:
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
