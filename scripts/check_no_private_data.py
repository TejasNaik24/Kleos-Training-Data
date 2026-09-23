from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EXIT_PRIVACY_VIOLATION = 4

PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "error"),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{32,}"), "error"),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"), "error"),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}"), "error"),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"), "error"),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"), "error"),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "error"),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
        "error",
    ),
    (
        "private_key_block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
        "error",
    ),
    ("supabase_url", re.compile(r"https://[a-z0-9]{20}\.supabase\.co"), "error"),
    (
        "supabase_service_key",
        re.compile(r"\bservice_role\b.{0,40}\beyJ", re.DOTALL),
        "error",
    ),
    (
        "assigned_secret",
        re.compile(
            r"\b(?:api[_-]?key|secret|password|passwd|token|credential)\s*[:=]\s*"
            r"['\"][A-Za-z0-9_\-./+]{16,}['\"]",
            re.IGNORECASE,
        ),
        "error",
    ),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{24,}"), "error"),
    (
        "session_cookie",
        re.compile(r"\b(?:sb-[a-z0-9-]+-auth-token|kleos_session)\s*=\s*[A-Za-z0-9._-]{16,}"),
        "error",
    ),
    (
        "database_url_with_password",
        re.compile(r"\bpostgres(?:ql)?(?:\+\w+)?://[^\s:/]+:[^\s@]{4,}@"),
        "error",
    ),
    ("email_address", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"), "warn"),
    (
        "us_phone",
        re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
        "warn",
    ),
    (
        "bare_uuid",
        re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"),
        "warn",
    ),
]

EMAIL_ALLOWLIST = re.compile(
    r"(?:noreply@|example\.com|example\.org|example\.invalid|\.invalid|"
    r"your[-_]?email|user@host|\.png|\.jpg|@\{|@example|name@domain)",
    re.IGNORECASE,
)

UUID_ALLOWLIST = re.compile(
    r"(?:00000000-0000-0000-0000-000000000000|^0{8}-|-0{12}$|deadbeef|f{8}-f{4})",
    re.IGNORECASE,
)

SKIP_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "node_modules",
        "build",
        "dist",
        ".eggs",
        "htmlcov",
        "site-packages",
        "staging",
        "vault",
        "releases",
        "reports",
    }
)


def _virtualenv_roots(root: Path) -> set[Path]:
    roots: set[Path] = set()
    for marker in root.rglob("pyvenv.cfg"):
        if any(parent in roots for parent in marker.parents):
            continue
        roots.add(marker.parent)
    return roots


SKIP_SUFFIXES = frozenset(
    {
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".gguf",
        ".ckpt",
        ".onnx",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".pdf",
        ".zip",
        ".gz",
        ".tar",
        ".woff",
        ".woff2",
        ".ttf",
        ".ico",
        ".lock",
    }
)

FORBIDDEN_PATHS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\.env$"), ".env must never be committed"),
    (re.compile(r"^\.env\.(?!example)"), "environment files must never be committed"),
    (
        re.compile(r"^staging/(?!\.gitkeep$)"),
        "staging/ holds unverified captures and must never be committed",
    ),
    (
        re.compile(r"^vault/(?!\.gitkeep$)"),
        "vault/ holds re-identification keys and must never be committed",
    ),
    (
        re.compile(r"^releases/(?!\.gitkeep$)"),
        "releases/ is shipped by path, not by git (see DATA_GOVERNANCE.md)",
    ),
    (
        re.compile(r"^reports/(?!\.gitkeep$)"),
        "reports/ may quote candidate content and must not be committed",
    ),
    (re.compile(r".*\.pem$|.*\.key$|.*\.p12$"), "key material must never be committed"),
)

SELF_EXEMPT = frozenset(
    {
        "scripts/check_no_private_data.py",
        "src/kleos_training_data/privacy/rules.py",
        "tests/test_privacy_scanner.py",
        "tests/test_privacy.py",
        "PRIVACY.md",
        "SECURITY.md",
        ".env.example",
    }
)

MAX_FILE_BYTES = 2 * 1024 * 1024


@dataclass
class Finding:
    path: Path
    line_number: int
    pattern: str
    severity: str
    excerpt: str

    def render(self, root: Path) -> str:
        try:
            location = self.path.relative_to(root)
        except ValueError:
            location = self.path
        icon = "✗" if self.severity == "error" else "!"
        return f"  {icon} {location}:{self.line_number}  [{self.pattern}]  {self.excerpt}"


def _iter_files(root: Path, paths: list[Path] | None = None):
    if paths:
        for path in paths:
            if path.is_file():
                yield path
        return

    venvs = _virtualenv_roots(root)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if any(venv in path.parents for venv in venvs):
            continue
        yield path


def _redact(line: str, match: re.Match[str]) -> str:
    text = match.group(0)
    shown = text[:6] + "…" if len(text) > 8 else "…"

    start, end = match.span()
    before = line[:start].strip()[-30:]
    after = line[end:].strip()[:30]
    snippet = f"{before}<{len(text)} chars redacted>{after}"

    if text in snippet:
        snippet = "<line withheld>"
    return f"{shown!r} in {snippet!r}"


def scan_file(path: Path, root: Path) -> list[Finding]:
    try:
        relative = str(path.relative_to(root))
    except ValueError:
        relative = str(path)
    if relative in SELF_EXEMPT:
        return []

    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return []
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    findings: list[Finding] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        for name, pattern, severity in PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            if name == "email_address" and EMAIL_ALLOWLIST.search(match.group(0)):
                continue
            if name == "bare_uuid" and UUID_ALLOWLIST.search(match.group(0)):
                continue
            findings.append(
                Finding(
                    path=path,
                    line_number=line_number,
                    pattern=name,
                    severity=severity,
                    excerpt=_redact(line, match),
                )
            )
    return findings


def _git_paths(root: Path, *, staged_only: bool) -> list[str] | None:
    if staged_only:
        command = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
    else:
        command = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    try:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    return [name for name in result.stdout.splitlines() if name]


def check_forbidden_paths(root: Path, *, staged_only: bool = False) -> list[tuple[str, str]]:
    names = _git_paths(root, staged_only=staged_only)
    if not names:
        return []
    problems: list[tuple[str, str]] = []
    for name in names:
        for pattern, reason in FORBIDDEN_PATHS:
            if pattern.match(name):
                problems.append((name, reason))
                break
    return problems


def _staged_files(root: Path) -> list[Path]:
    names = _git_paths(root, staged_only=True) or []
    return [root / name for name in names if (root / name).is_file()]


HOOK_SCRIPT = """#!/bin/sh
# KLEOS training-data pre-commit hook: block secrets and private data.
exec python3 scripts/check_no_private_data.py --staged
"""


def install_hook(root: Path) -> int:
    hooks_dir = root / ".git" / "hooks"
    if not hooks_dir.exists():
        print("✗ No .git/hooks directory. Run `git init` first.", file=sys.stderr)
        return 1
    hook_path = hooks_dir / "pre-commit"
    if hook_path.exists():
        print(f"! {hook_path} already exists; not overwriting.")
        print("  Add this line to it manually:")
        print("    python3 scripts/check_no_private_data.py --staged")
        return 0
    hook_path.write_text(HOOK_SCRIPT, encoding="utf-8")
    hook_path.chmod(0o755)
    print(f"✓ Installed pre-commit hook at {hook_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=REPO_ROOT, help="Directory to scan.")
    parser.add_argument("--staged", action="store_true", help="Scan only git-staged files.")
    parser.add_argument("--install-hook", action="store_true", help="Install the pre-commit hook.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures too.")
    args = parser.parse_args(argv)

    root = args.path.resolve()

    if args.install_hook:
        return install_hook(root)

    targets = _staged_files(root) if args.staged else None
    if args.staged and not targets:
        print("✓ No staged files to scan.")
        return 0

    print()
    print("=" * 72)
    print(f"Private-data scan — {root}")
    print("=" * 72)

    findings: list[Finding] = []
    scanned = 0
    for path in _iter_files(root, targets):
        scanned += 1
        findings.extend(scan_file(path, root))

    forbidden = check_forbidden_paths(root, staged_only=args.staged)
    git_available = _git_paths(root, staged_only=False) is not None

    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warn"]

    print(f"\n  scanned  : {scanned} file(s)")
    print(f"  errors   : {len(errors)}")
    print(f"  warnings : {len(warnings)}")
    print(f"  forbidden paths : {len(forbidden)}")

    if not git_available:
        print("\n  ! git is unavailable, so tracked-path checks were skipped.")
        print("    staging/, vault/ and releases/ could not be verified.")

    if forbidden:
        print("\nPaths git is tracking that must never be committed:")
        for name, reason in forbidden:
            print(f"  ✗ {name}: {reason}")

    if errors:
        print("\nSecrets or credentials detected:")
        for finding in errors[:40]:
            print(finding.render(root))

    if warnings:
        print("\nPossible personal data (review these):")
        for finding in warnings[:20]:
            print(finding.render(root))
        if len(warnings) > 20:
            print(f"  … {len(warnings) - 20} more")

    failed = bool(errors) or bool(forbidden) or (args.strict and bool(warnings))

    print()
    if failed:
        print("✗ Private-data scan FAILED.", file=sys.stderr)
        print(
            "\n  This repository is PRIVATE, but private git is not a secure vault.",
            file=sys.stderr,
        )
        print("  A clone, a fork, or a future collaborator sees everything in", file=sys.stderr)
        print("  history. Do not commit until these are resolved.\n", file=sys.stderr)
        print("  If a hit is a false positive, rewrite the line so it cannot be", file=sys.stderr)
        print("  mistaken for a real secret. Adding a file to SELF_EXEMPT turns", file=sys.stderr)
        print("  this scanner off for that file permanently.\n", file=sys.stderr)
        print("  If a secret was already committed, rotate it immediately —", file=sys.stderr)
        print(
            "  removing it from the working tree does not remove it from history.", file=sys.stderr
        )
        print("  See SECURITY.md and docs/incident-response.md.\n", file=sys.stderr)
        return EXIT_PRIVACY_VIOLATION

    print("✓ No secrets or private data detected.")
    if warnings:
        print(f"  ({len(warnings)} warning(s) above are worth a look.)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
