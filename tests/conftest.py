"""Shared test fixtures and builders.

Conventions, mirroring the public repo's test suite:

* Module-level path constants, so no test computes a path itself.
* Plain builder *functions* returning dicts, plus thin fixture wrappers. Tests
  import the functions directly (``from tests.conftest import make_payload``);
  the fixtures exist for tests that prefer injection.
* Optional dependencies are handled by a collection hook, not by
  ``importorskip`` scattered through the suite.
* No mocking library. Fakes are written by hand so their behaviour is readable
  at the point of failure.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
FIXTURES_DIR = REPO_ROOT / "data" / "fixtures"
SURROGATES_DIR = REPO_ROOT / "data" / "surrogates"
SCENARIOS_DIR = REPO_ROOT / "scenarios"
CONFIGS_DIR = REPO_ROOT / "configs"
DOCS_DIR = REPO_ROOT / "docs"


def _available(module: str) -> bool:
    """Whether a module can be imported without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


HAS_KLEOS_MODELS = _available("kleos_models")
HAS_GIT = shutil.which("git") is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests whose optional dependency is absent.

    A skipped differential test is a *degraded* run, not a passing one — CI
    installs the ``compat`` extra and runs ``check_contract_compat.py --strict``
    so that a skip there fails the build. Locally, skipping is the right
    behaviour: the offline pipeline must be workable without the public repo
    checked out.
    """
    skip_public = pytest.mark.skip(
        reason='needs the pinned public contract: pip install -e ".[dev,compat]"'
    )
    skip_network = pytest.mark.skip(reason="performs real network I/O; run explicitly")
    for item in items:
        if "requires_kleos_models" in item.keywords and not HAS_KLEOS_MODELS:
            item.add_marker(skip_public)
        if "requires_network" in item.keywords:
            item.add_marker(skip_network)


# ---------------------------------------------------------------------------
# Git sandbox
# ---------------------------------------------------------------------------


def git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a git command in ``cwd``, raising on failure."""
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


def make_git_sandbox(directory: Path, *, gitignore: Path | None = None) -> Path:
    """Initialize a throwaway git repository carrying the real ``.gitignore``.

    Ignore rules are verified against real git rather than a re-implementation.
    A hand-rolled matcher that disagrees with git is worse than no check: it
    passes while the real thing lets a capture through.
    """
    git("init", "-q", cwd=directory)
    git("config", "user.email", "tests@example.invalid", cwd=directory)
    git("config", "user.name", "Test Runner", cwd=directory)
    source = gitignore or (REPO_ROOT / ".gitignore")
    shutil.copy(source, directory / ".gitignore")
    return directory


def touch(root: Path, relative: str, content: str = "x") -> Path:
    """Create a file (and its parents) under ``root``."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Contract payload builders
# ---------------------------------------------------------------------------


def make_messages(
    *,
    system: str | None = "Rank the items by deadline proximity and evidence strength.",
    user: str = "Two items are open. Item A is due Friday with a confirmed owner. "
    "Item B is due next month and nobody has looked at it.",
    assistant: str = "- Item A first: the deadline is nearest and the evidence is confirmed.\n"
    "- Item B second: distant deadline, and no evidence has been gathered yet.",
) -> list[dict[str, Any]]:
    """A minimal contract-valid conversation: system, user, assistant."""
    messages: list[dict[str, Any]] = []
    if system is not None:
        messages.append({"role": "system", "content": system, "name": None})
    messages.append({"role": "user", "content": user, "name": None})
    messages.append({"role": "assistant", "content": assistant, "name": None})
    return messages


def make_payload(
    *,
    task: str = "notification_prioritization",
    messages: list[dict[str, Any]] | None = None,
    domain: str = "career",
    metadata: dict[str, Any] | None = None,
    **axes: Any,
) -> dict[str, Any]:
    """Build a training-example payload as a plain dict.

    Returns a dict rather than a model so tests can mutate one field into an
    invalid state without fighting validation on the way in.

    Extra keyword arguments spill into ``variation_axes``, so
    ``make_payload(urgency="low")`` reads naturally at the call site.
    """
    variation_axes: dict[str, Any] = {"domain": domain}
    variation_axes.update({k: v for k, v in axes.items() if v is not None})
    payload: dict[str, Any] = {
        "task": task,
        "messages": messages if messages is not None else make_messages(),
        "variation_axes": variation_axes,
    }
    payload["metadata"] = (
        metadata
        if metadata is not None
        else {
            "source": "synthetic",
            "quality_status": "reviewed",
        }
    )
    return payload


@pytest.fixture
def payload_factory():
    """The :func:`make_payload` builder, for tests that prefer injection."""
    return make_payload


@pytest.fixture
def message_factory():
    """The :func:`make_messages` builder."""
    return make_messages


@pytest.fixture
def workspace(tmp_path: Path):
    """An initialized workspace rooted in a temporary directory."""
    import sys

    from kleos_training_data.paths import Workspace

    sys.path.insert(0, str(SCRIPTS_DIR))
    from init_workspace import create_directories, secure_vault

    space = Workspace.from_env(tmp_path)
    create_directories(space)
    secure_vault(space)
    return space
