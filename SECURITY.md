# Security

This repository is **private**. Private git is not a secure vault: a clone, a
fork, a compromised laptop or a future collaborator sees everything, and history
is forever. Treat the contents accordingly.

## What must never be committed

| Category | Examples |
| --- | --- |
| Credentials | API keys, bearer tokens, JWTs, session cookies, OAuth refresh tokens |
| Database access | Connection URLs containing a password |
| Key material | `*.pem`, `*.key`, `*.p12`, private key blocks |
| Environment files | `.env`, `.env.local`, `.env.production` |
| Raw captures | Anything under `staging/` |
| Re-identification keys | Anything under `vault/` |
| Datasets | Anything under `releases/` |
| Reports | Anything under `reports/` — they may quote candidate text |

The four runtime zones are deny-by-default in `.gitignore`, with only a
top-level `.gitkeep` tracked. `tests/test_gitignore.py` proves this against real
git rather than a re-implementation, and its decisive assertion is what
`git add -A` actually stages.

## Before every commit

```bash
python scripts/check_no_private_data.py .          # exit 4 if anything is found
```

Install it as a hook so you cannot forget:

```bash
python scripts/check_no_private_data.py --install-hook
```

The scanner imports nothing outside the standard library so it runs on a bare
machine and as the first CI job — a leaked secret should fail the build in ten
seconds, not after a five-minute dependency install.

`SELF_EXEMPT` turns the scanner **off** for a file. Prefer rewriting the
offending line so it cannot be mistaken for a real secret. Every entry must name
a file that exists; a test enforces that, because a stale entry silently exempts
whatever gets created at that path later.

## Credential handling

- Secrets come from environment variables, loaded from a git-ignored `.env`.
  Start from `.env.example`.
- Every variable is optional. The entire offline pipeline — generation, mock
  capture, sanitization, review, promotion, release, verification — runs with
  none of them set. A pipeline that needs a live credential to be tested is one
  whose first end-to-end run creates private data before any gate exists to
  catch it.
- Credentials are wrapped in `SafeSecret`, which renders as
  `<secret len=64 sha256=1a2b3c4d>` under `str`, `repr`, f-strings, `%`
  formatting and `json.dumps(default=str)`. Only `.reveal()` returns the value.
  This is a *mistake* boundary, not a security boundary: the way a token reaches
  a log is almost never a deliberate print, it is an exception message or a repr
  of a config object.
- Use a scoped, read-only, purpose-created credential for any backend capture,
  and revoke it afterwards. **Never use production credentials for a data
  export.**

## Backend capture safety

Capturing against a non-local backend requires **all** of the following, and the
guard names the ones that failed without ever naming a way to disable itself:

1. `--allow-production` on the command line
2. `KLEOS_ALLOW_PRODUCTION_CAPTURE=1` in the environment
3. The exact confirmation phrase
4. `CI` **unset** — this one is not overridable, because an automated production
   capture is never legitimate

Captures obtained this way are marked `production_observation` and can never be
promoted into a dataset. See `PRIVACY.md`.

## Logging

No logger in this repository writes message content, request bodies or response
bodies — not at DEBUG, not behind a flag. `redact()` refuses to show a preview
unless the caller passes `allow_preview=True`, so the decision is visible at the
call site and greppable afterwards.

To read candidate text, open the reviewer packet. That is what it is for, and it
is written to a git-ignored directory.

## If a secret is exposed

Deleting the file in a later commit does **not** remove it from history.

1. **Rotate the credential immediately.** Assume it is compromised. Do this
   before anything else — the cleanup can wait, the rotation cannot.
2. Stop distribution: do not push, do not share the branch.
3. Remove it from the working tree.
4. Rewrite history with an approved tool (`git filter-repo`), and force-push
   only after coordinating with anyone who has a clone.
5. **Verify the old credential is actually invalid** by trying to use it.
6. Audit access logs for use of the credential during the exposure window.
7. Write it up in the incident record.

The same applies to private data, with one addition: identify which dataset
releases and which trained checkpoints are affected. A release cannot be edited
— it is immutable — so remediation means a new version plus an assessment of
whatever was already trained. See `docs/incident-response.md`.

## Reporting

This is a private repository with a single maintainer. Report anything you find
directly to the repository owner. Do not open a public issue anywhere, and do
not include the secret itself in the report — a digest and a file path are
enough to act on.
