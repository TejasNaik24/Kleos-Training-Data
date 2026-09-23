from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from typing import ClassVar

import pytest
from tests.conftest import HAS_GIT, REPO_ROOT, make_git_sandbox, touch

SCANNER_PATH = REPO_ROOT / "scripts" / "check_no_private_data.py"


def _load_scanner():
    spec = importlib.util.spec_from_file_location("_check_no_private_data", SCANNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scanner = _load_scanner()


class TestDependencyFreedom:
    def test_imports_only_the_standard_library(self) -> None:
        tree = ast.parse(SCANNER_PATH.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])

        non_stdlib = sorted(imported - set(sys.stdlib_module_names))
        assert not non_stdlib, (
            f"{SCANNER_PATH.name} imports non-stdlib module(s): {non_stdlib}. "
            f"It must run before `pip install` on a bare CI runner."
        )

    def test_does_not_import_the_package_it_guards(self) -> None:
        tree = ast.parse(SCANNER_PATH.read_text(encoding="utf-8"))
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)

        offenders = sorted(m for m in modules if m.split(".")[0] == "kleos_training_data")
        assert not offenders, f"The scanner must not import the library it guards: {offenders}"


class TestSecretDetection:
    SAMPLES: ClassVar[list[tuple[str, str]]] = [
        ("aws_access_key", "AKIAIOSFODNN7EXAMPLE"),
        ("openai_key", "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD"),
        ("anthropic_key", "sk-ant-api03-notarealkey_abcdefghijklmnop"),
        ("hf_token", "hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        ("github_token", "ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        ("slack_token", "xoxb-1111111111-notarealtoken"),
        ("google_api_key", "AIzaSy" + "A" * 33),
        ("jwt", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijk"),
        ("private_key_block", "-----BEGIN RSA PRIVATE KEY-----"),
        ("supabase_url", "https://abcdefghijklmnopqrst.supabase.co"),
        ("bearer_token", "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345"),
        ("assigned_secret", 'api_key = "abcdefghijklmnopqrstuvwx"'),
        ("session_cookie", "sb-abcdefghij-auth-token=abcdefghijklmnopqrst"),
        ("database_url_with_password", "postgresql+asyncpg://postgres:hunter2xyz@db.host:5432/x"),
    ]

    @pytest.mark.parametrize(("name", "sample"), SAMPLES, ids=[n for n, _ in SAMPLES])
    def test_pattern_fires(self, name: str, sample: str) -> None:
        matched = {
            pattern_name
            for pattern_name, pattern, _severity in scanner.PATTERNS
            if pattern.search(sample)
        }
        assert name in matched, (
            f"Pattern {name!r} did not match its own sample. Detection that does "
            f"not fire is worse than no detection: it produces a green build."
        )

    @pytest.mark.parametrize(("name", "sample"), SAMPLES, ids=[n for n, _ in SAMPLES])
    def test_sample_is_reported_as_an_error(self, name: str, sample: str, tmp_path) -> None:
        target = touch(tmp_path, "candidate.py", f"value = {sample!r}\n")
        findings = scanner.scan_file(target, tmp_path)
        severities = {f.severity for f in findings}
        assert "error" in severities, f"{name} produced {severities or 'no findings'}, not an error"

    def test_clean_source_produces_nothing(self, tmp_path) -> None:
        target = touch(
            tmp_path,
            "clean.py",
            "def rank(items):\n    return sorted(items, key=lambda i: i.deadline)\n",
        )
        assert scanner.scan_file(target, tmp_path) == []


class TestAllowlists:
    @pytest.mark.parametrize(
        "address",
        [
            "noreply@example.com",
            "alice@example.invalid",
            "user@host",
            "your-email@example.org",
        ],
    )
    def test_placeholder_emails_are_allowed(self, address: str, tmp_path) -> None:
        target = touch(tmp_path, "doc.md", f"Contact {address} for details.\n")
        emails = [f for f in scanner.scan_file(target, tmp_path) if f.pattern == "email_address"]
        assert not emails, f"{address} should be allowlisted; it is a documentation placeholder"

    def test_a_real_looking_email_is_still_reported(self, tmp_path) -> None:
        target = touch(tmp_path, "doc.md", "Contact j.doe@somecollege.edu for details.\n")
        emails = [f for f in scanner.scan_file(target, tmp_path) if f.pattern == "email_address"]
        assert emails, "A plausible personal address must still be surfaced"
        assert emails[0].severity == "warn"

    def test_the_nil_uuid_is_allowed(self, tmp_path) -> None:
        target = touch(tmp_path, "fixture.json", '{"id": "00000000-0000-0000-0000-000000000000"}\n')
        uuids = [f for f in scanner.scan_file(target, tmp_path) if f.pattern == "bare_uuid"]
        assert not uuids

    def test_a_real_looking_uuid_is_reported(self, tmp_path) -> None:
        target = touch(tmp_path, "note.md", "user 9d0e6a2c-1f4b-4a55-9a8e-0b3f1c77e2d1\n")
        uuids = [f for f in scanner.scan_file(target, tmp_path) if f.pattern == "bare_uuid"]
        assert uuids
        assert uuids[0].severity == "warn", "Too common to block on, too load-bearing to ignore"


class TestRedaction:
    def test_the_secret_is_never_echoed_in_full(self, tmp_path) -> None:
        secret = "ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        target = touch(tmp_path, "leak.py", f'token = "{secret}"\n')
        findings = scanner.scan_file(target, tmp_path)
        assert findings

        rendered = "\n".join(f.render(tmp_path) for f in findings)
        assert secret not in rendered, "The scanner reprinted the secret it found"

    def test_enough_context_survives_to_locate_the_hit(self, tmp_path) -> None:
        target = touch(
            tmp_path, "leak.py", 'token = "ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n'
        )
        finding = scanner.scan_file(target, tmp_path)[0]
        assert finding.line_number == 1
        assert "leak.py" in finding.render(tmp_path)
        assert "ghp_aa" in finding.excerpt, "A prefix is needed to tell two hits apart"


class TestSelfExemption:
    def test_every_exempt_path_exists(self) -> None:
        missing = [p for p in scanner.SELF_EXEMPT if not (REPO_ROOT / p).exists()]
        assert not missing, (
            f"SELF_EXEMPT names {len(missing)} path(s) that do not exist: {sorted(missing)}. "
            f"Remove them, or create the file the exemption was written for."
        )

    def test_the_list_stays_short(self) -> None:
        assert len(scanner.SELF_EXEMPT) <= 10, (
            "SELF_EXEMPT has grown. Each entry is a file the scanner no longer "
            "checks at all — prefer rewriting the offending line so it cannot be "
            "mistaken for a real secret."
        )


@pytest.mark.skipif(not HAS_GIT, reason="git is not installed")
class TestForbiddenPaths:
    @pytest.fixture
    def repo(self, tmp_path):
        return make_git_sandbox(tmp_path)

    def test_private_data_on_disk_is_not_a_finding(self, repo) -> None:
        touch(repo, "staging/raw/cap.json", '{"answer": "..."}')
        touch(repo, "vault/entity_vault.json", "{}")
        touch(repo, "releases/v1/train.jsonl", "{}")
        assert scanner.check_forbidden_paths(repo) == []

    def test_a_private_path_git_does_not_ignore_is_a_finding(self, repo) -> None:
        (repo / ".gitignore").write_text("# rules accidentally removed\n", encoding="utf-8")
        touch(repo, "staging/raw/cap.json", '{"answer": "..."}')

        problems = scanner.check_forbidden_paths(repo)
        assert [name for name, _ in problems] == ["staging/raw/cap.json"]

    @pytest.mark.parametrize(
        "relative",
        [
            "staging/raw/cap.json",
            "vault/entity_vault.json",
            "releases/kleos-policy-v0.1.0/train.jsonl",
            "reports/coverage/report.md",
            ".env",
            "server.key",
        ],
    )
    def test_tracking_a_private_path_is_a_finding(self, repo, relative: str) -> None:
        touch(repo, relative, "x")
        subprocess.run(["git", "add", "-f", relative], cwd=repo, check=True, capture_output=True)

        problems = scanner.check_forbidden_paths(repo, staged_only=True)
        assert [name for name, _ in problems] == [relative], (
            f"git is tracking {relative} and the scanner did not object"
        )

    def test_the_gitkeep_files_are_permitted(self, repo) -> None:
        for zone in ("staging", "vault", "releases", "reports"):
            touch(repo, f"{zone}/.gitkeep", "")
        subprocess.run(["git", "add", "-Af"], cwd=repo, check=True, capture_output=True)
        assert scanner.check_forbidden_paths(repo, staged_only=True) == []


class TestVirtualenvDetection:
    def test_a_nonstandard_venv_name_is_skipped(self, tmp_path) -> None:
        touch(tmp_path, "env311/pyvenv.cfg", "home = /usr/bin\n")
        touch(tmp_path, "env311/lib/site/cert.pem", "-----BEGIN RSA PRIVATE KEY-----\n")
        touch(tmp_path, "src/app.py", "x = 1\n")

        scanned = {p.name for p in scanner._iter_files(tmp_path)}
        assert "app.py" in scanned
        assert "cert.pem" not in scanned

    def test_a_real_key_outside_a_venv_is_still_scanned(self, tmp_path) -> None:
        touch(tmp_path, "src/app.py", "x = 1\n")
        touch(tmp_path, "src/leaked.txt", "-----BEGIN OPENSSH PRIVATE KEY-----\n")
        scanned = {p.name for p in scanner._iter_files(tmp_path)}
        assert "leaked.txt" in scanned

    def test_runtime_zones_are_not_content_scanned(self, tmp_path) -> None:
        touch(tmp_path, "staging/raw/cap.json", '{"x": 1}')
        touch(tmp_path, "vault/entity_vault.json", "{}")
        touch(tmp_path, "src/app.py", "x = 1\n")
        scanned = {str(p.relative_to(tmp_path)) for p in scanner._iter_files(tmp_path)}
        assert "src/app.py" in scanned
        assert not any(s.startswith(("staging/", "vault/")) for s in scanned)


class TestExitCodes:
    def test_a_clean_tree_exits_zero(self, tmp_path) -> None:
        touch(tmp_path, "app.py", "x = 1\n")
        assert scanner.main([str(tmp_path)]) == 0

    def test_a_secret_exits_with_the_privacy_code(self, tmp_path) -> None:
        touch(tmp_path, "app.py", 'k = "AKIAIOSFODNN7EXAMPLE"\n')
        assert scanner.main([str(tmp_path)]) == scanner.EXIT_PRIVACY_VIOLATION == 4

    def test_warnings_alone_pass_unless_strict(self, tmp_path) -> None:
        touch(tmp_path, "notes.md", "reach me at j.doe@somecollege.edu\n")
        assert scanner.main([str(tmp_path)]) == 0
        assert scanner.main([str(tmp_path), "--strict"]) == scanner.EXIT_PRIVACY_VIOLATION

    def test_the_repository_itself_is_clean(self) -> None:
        assert scanner.main([str(REPO_ROOT)]) == 0
