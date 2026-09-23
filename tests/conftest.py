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
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


HAS_KLEOS_MODELS = _available("kleos_models")
HAS_GIT = shutil.which("git") is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_public = pytest.mark.skip(
        reason='needs the pinned public contract: pip install -e ".[dev,compat]"'
    )
    skip_network = pytest.mark.skip(reason="performs real network I/O; run explicitly")
    for item in items:
        if "requires_kleos_models" in item.keywords and not HAS_KLEOS_MODELS:
            item.add_marker(skip_public)
        if "requires_network" in item.keywords:
            item.add_marker(skip_network)


def git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


def make_git_sandbox(directory: Path, *, gitignore: Path | None = None) -> Path:
    git("init", "-q", cwd=directory)
    git("config", "user.email", "tests@example.invalid", cwd=directory)
    git("config", "user.name", "Test Runner", cwd=directory)
    source = gitignore or (REPO_ROOT / ".gitignore")
    shutil.copy(source, directory / ".gitignore")
    return directory


def touch(root: Path, relative: str, content: str = "x") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def make_messages(
    *,
    system: str | None = "Rank the items by deadline proximity and evidence strength.",
    user: str = "Two items are open. Item A is due Friday with a confirmed owner. "
    "Item B is due next month and nobody has looked at it.",
    assistant: str = "- Item A first: the deadline is nearest and the evidence is confirmed.\n"
    "- Item B second: distant deadline, and no evidence has been gathered yet.",
) -> list[dict[str, Any]]:
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
    return make_payload


@pytest.fixture
def message_factory():
    return make_messages


@pytest.fixture
def workspace(tmp_path: Path):
    import sys

    from kleos_training_data.paths import Workspace

    sys.path.insert(0, str(SCRIPTS_DIR))
    from init_workspace import create_directories, secure_vault

    space = Workspace.from_env(tmp_path)
    create_directories(space)
    secure_vault(space)
    return space
