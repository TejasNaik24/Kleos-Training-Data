# Security policy

This document explains how to report a security or privacy issue, and how the
project keeps credentials and personal data out of the repository.

## Contents

- [Supported versions](#supported-versions)
- [Reporting a vulnerability](#reporting-a-vulnerability)
- [What must never be committed](#what-must-never-be-committed)
- [Secret scanning](#secret-scanning)
- [Credential handling](#credential-handling)
- [Production capture](#production-capture)
- [Responding to an exposed secret](#responding-to-an-exposed-secret)

## Supported versions

| Version | Supported |
| --- | --- |
| The `main` branch | Yes |
| The latest dataset release, `kleos-policy-v0.0.7` | Yes |
| Earlier dataset releases | No. Releases are immutable, so fixes ship in a new version. |

## Reporting a vulnerability

Report vulnerabilities and privacy concerns privately through GitHub: open the
repository's **Security** tab and choose **Report a vulnerability**. Do not open
a public issue for a security problem.

Include the affected file or commit, a description of the problem, steps to
reproduce it, and its likely impact. If the report concerns an exposed
credential, give its location and a hash of it, never the credential itself.

These are in scope:

- Secrets or personal data anywhere in the repository or its history
- A way to promote an example past a mandatory promotion gate
- A way for a secret or personal data to reach a release without being detected
- A way to bypass the production capture guard

## What must never be committed

| Category | Examples |
| --- | --- |
| Credentials | API keys, bearer tokens, JWTs, session cookies, OAuth refresh tokens |
| Database access | Connection URLs that contain a password |
| Key material | `*.pem`, `*.key` and `*.p12` files, private key blocks |
| Environment files | `.env`, `.env.local`, `.env.production` |
| Runtime zones | Anything under `staging/`, `vault/`, `releases/` or `reports/` other than the top-level `.gitkeep` |

The runtime zones are git-ignored with deny-by-default rules, and
`tests/test_gitignore.py` checks those rules against a real git repository,
including what `git add -A` would stage.

## Secret scanning

```bash
python scripts/check_no_private_data.py .
python scripts/check_no_private_data.py --install-hook
```

`check_no_private_data.py` uses only the Python standard library, so it runs
before any dependency is installed. It is the first job in CI.

| Behavior | Detail |
| --- | --- |
| Content patterns | 15 error-level patterns for credentials and key material, and 3 warning-level patterns for email addresses, phone numbers and UUIDs |
| Forbidden paths | Files that git would commit under the runtime zones, `.env` files and key files |
| Exit code | 4 for any error-level finding or forbidden path, and for warnings when `--strict` is given. 0 otherwise. |
| `--staged` | Scans only the files staged for commit. The pre-commit hook uses this mode. |
| `--install-hook` | Installs a pre-commit hook. An existing hook is never overwritten. |

The scanner skips the content of the files listed in `SELF_EXEMPT`: the scanner
itself, the privacy rule definitions and their tests, this file, `PRIVACY.md`
and `.env.example`. A test requires every entry to exist and limits the list to
ten entries. Rewrite a line that triggers a false positive rather than adding
its file to the list.

## Credential handling

- Configuration is read from environment variables. `.env.example` lists them,
  and nothing loads a `.env` file automatically. To load one into the current
  shell, run `set -a; . ./.env; set +a`.
- Every variable is optional. The offline pipeline, from generation to
  verification, runs without any credential.
- Credentials are wrapped in `SafeSecret`, which renders as
  `<secret len=64 sha256=1a2b3c4d>` in `str()`, `repr()` and format strings.
  Only `.reveal()` returns the value, and the one call site is the HTTP
  transport that sets the `Authorization` header. The wrapper prevents
  accidental leaks through logs, exception messages and object representations.
  It does not protect against code that calls `.reveal()` on purpose.
- `python scripts/doctor.py` reports whether each credential is set without
  printing any value.
- For backend capture, use a scoped, read-only credential created for that
  purpose, and revoke it afterward. Never use production credentials for a data
  export.

## Production capture

Capturing from a non-local backend requires four independent conditions,
including an unset `CI` variable that no option can override. The adapters for
the live KLEOS backend label their captures `production_observation`, and
promotion gate G10 rejects that lane, so such captures can never be promoted
into a dataset. The capture CLI currently runs only the mock adapter. The guard
is described in
[docs/collection.md](docs/collection.md#production-capture-guard).

## Responding to an exposed secret

Rotate the credential first, before any cleanup. The complete procedure,
including history rewriting and verification that the old credential is invalid,
is in [docs/incident-response.md](docs/incident-response.md#exposed-secret).

## Related documentation

- [PRIVACY.md](PRIVACY.md): the privacy policy
- [docs/incident-response.md](docs/incident-response.md): incident runbooks
- [docs/collection.md](docs/collection.md): capture lanes and the production guard
- [CONTRIBUTING.md](CONTRIBUTING.md): the development workflow
