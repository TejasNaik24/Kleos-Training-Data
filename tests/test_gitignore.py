from __future__ import annotations

import subprocess

import pytest
from tests.conftest import HAS_GIT, REPO_ROOT, git, make_git_sandbox, touch

pytestmark = pytest.mark.skipif(not HAS_GIT, reason="git is not installed")


MUST_BE_IGNORED = [
    "staging/raw/slice-001/cap_9d0e6a2c.json",
    "staging/raw/slice-001/batch.json",
    "staging/normalized/slice-001/kx-npr-3f9a1c8e2b7d0456.json",
    "staging/sanitized/slice-001/kx-npr-a7c30fe19b2d4468.json",
    "staging/sanitized/slice-001/kx-npr-a7c30fe19b2d4468.privacy.json",
    "staging/review/packets/pk-001/packet.md",
    "staging/review/llm/kx-npr-a7c30fe19b2d4468.json",
    "staging/review/decisions/kx-npr-a7c30fe19b2d4468.json",
    "staging/rejected/kx-npr-deadbeef12345678.json",
    "staging/promoted/index.jsonl",
    "staging/promoted/gate_reports/pr-001.json",
    "vault/entity_vault.json",
    "vault/surrogate_maps/notif.deadline_vs_evidence.json",
    "releases/kleos-policy-v0.1.0/train.jsonl",
    "releases/kleos-policy-v0.1.0/validation.jsonl",
    "releases/kleos-policy-v0.1.0/test.jsonl",
    "releases/kleos-policy-v0.1.0/manifest.json",
    "releases/kleos-policy-v0.1.0/provenance.json",
    "releases/kleos-policy-v0.1.0/RELEASE.lock",
    "reports/coverage/kleos-policy-v0.1.0.md",
    "reports/leakage/leakage_report.json",
    "reports/dedup/near_duplicates.json",
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "data/my_export.jsonl",
    "data/memories.json",
    "data/raw_conversations/2026-08-21.json",
    ".env",
    ".env.local",
    ".env.production",
    "server.key",
    "client.pem",
    "bundle.p12",
    "credentials.json",
    "service-account-kleos.json",
    ".DS_Store",
    "__pycache__/module.cpython-311.pyc",
    ".pytest_cache/CACHEDIR.TAG",
    ".mypy_cache/index.json",
]

MUST_BE_TRACKED = [
    "staging/.gitkeep",
    "vault/.gitkeep",
    "releases/.gitkeep",
    "reports/.gitkeep",
    ".gitignore",
    ".gitattributes",
    ".env.example",
    "pyproject.toml",
    "Makefile",
    "README.md",
    "PRIVACY.md",
    "SECURITY.md",
    "src/kleos_training_data/__init__.py",
    "src/kleos_training_data/privacy/rules.py",
    "scripts/check_no_private_data.py",
    "scripts/build_release.py",
    "tests/conftest.py",
    "scenarios/notification_prioritization/deadline_vs_evidence.yaml",
    "scenarios/_shared/system_prompts/reasoning_layer.md",
    "configs/pipeline.yaml",
    "data/README.md",
    "data/surrogates/generic_pool_a.yaml",
    "data/fixtures/valid/minimal.json",
    "data/fixtures/invalid/consecutive_assistant.json",
    "data/id_ledger.json",
    "docs/privacy.md",
    ".github/workflows/ci.yml",
]


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    directory = tmp_path_factory.mktemp("gitignore_sandbox")
    for relative in MUST_BE_IGNORED + MUST_BE_TRACKED:
        touch(directory, relative)
    make_git_sandbox(directory)

    copied = (directory / ".gitignore").read_text(encoding="utf-8")
    assert copied == (REPO_ROOT / ".gitignore").read_text(encoding="utf-8"), (
        "The sandbox .gitignore is not the repository's. Every assertion in this "
        "module would be testing the wrong file."
    )
    return directory


def is_ignored(sandbox, relative: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", relative], cwd=sandbox, capture_output=True
    )
    return result.returncode == 0


@pytest.mark.parametrize("relative", MUST_BE_IGNORED)
def test_private_paths_are_ignored(sandbox, relative: str) -> None:
    assert is_ignored(sandbox, relative), (
        f"{relative} would be COMMITTED. This repository is private, but private "
        f"git is not a vault — a clone, a fork or a collaborator sees it, and "
        f"history is forever. Check .gitignore uses `<zone>/*` rather than "
        f"`<zone>/`: git cannot re-include a file whose parent directory is "
        f"excluded. Debug with: git check-ignore -v {relative}"
    )


@pytest.mark.parametrize("relative", MUST_BE_TRACKED)
def test_repository_content_stays_trackable(sandbox, relative: str) -> None:
    assert not is_ignored(sandbox, relative), (
        f"{relative} is ignored, so it would vanish from a fresh clone. The "
        f"ignore rules are too broad. Debug with: git check-ignore -v {relative}"
    )


def test_staging_everything_admits_only_the_allowlist(sandbox) -> None:
    git("add", "-A", cwd=sandbox)
    staged = set(git("diff", "--cached", "--name-only", cwd=sandbox).stdout.split())

    leaked = staged & set(MUST_BE_IGNORED)
    assert not leaked, (
        f"`git add -A` staged {len(leaked)} path(s) that must never be committed: {sorted(leaked)}"
    )

    missing = set(MUST_BE_TRACKED) - staged
    assert not missing, (
        f"`git add -A` did NOT stage {len(missing)} path(s) that the repository "
        f"needs: {sorted(missing)}"
    )


def test_every_runtime_zone_is_covered() -> None:
    from kleos_training_data.paths import Workspace

    space = Workspace.from_env(REPO_ROOT)
    zones = [path.name for path in space.all_zones()]
    assert zones, "Workspace.all_zones() returned nothing"

    for zone in zones:
        assert any(p.startswith(f"{zone}/") for p in MUST_BE_IGNORED), (
            f"No MUST_BE_IGNORED probe covers the {zone!r} zone."
        )
        assert f"{zone}/.gitkeep" in MUST_BE_TRACKED, (
            f"No MUST_BE_TRACKED probe keeps {zone}/.gitkeep alive."
        )


def test_gitignore_has_no_trailing_comments_on_patterns() -> None:
    offenders = []
    for number, line in enumerate((REPO_ROOT / ".gitignore").read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "#" in stripped:
            offenders.append(f"line {number}: {stripped}")

    assert not offenders, (
        "These .gitignore lines have trailing comments, which git treats as part "
        "of the pattern:\n  " + "\n  ".join(offenders)
    )
