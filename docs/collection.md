# Collection

Collection turns scenario requests into raw capture records in `staging/raw/`.
This document covers the capture lanes, the adapters, the guard around
production capture, and the HTTP transport used by the backend adapters.

## Contents

- [Capture lanes](#capture-lanes)
- [Adapters](#adapters)
- [Running a capture](#running-a-capture)
- [Production capture guard](#production-capture-guard)
- [HTTP transport](#http-transport)
- [Logging](#logging)

## Capture lanes

Every capture is stamped with a lane, and the lane decides whether its content
can ever reach a dataset.

| Lane | Source | Contract source | Promotable |
| --- | --- | --- | --- |
| `mock_backend` | The deterministic mock adapter | `synthetic` | Yes |
| `synthetic` | Scenarios rendered locally without a backend. Defined, but not produced by the current tools. | `synthetic` | Yes |
| `production_observation` | The live KLEOS backend | `real_sanitized` | No |

Every release so far was built from the `mock_backend` lane.

The live KLEOS backend answers from the authenticated user's own projects,
memories and notifications, so a capture from it is that user's personal data
whatever the prompt asked. Promotion gate G10 rejects the `production_observation`
lane outright. Such a capture can serve only as reference material: a person reads
it, learns what situations occur in real use, and writes a new, generalized
scenario by hand. The contract vocabulary includes a `synthetic_seeded` source
type for examples written this way. The current tools do not produce it.

## Adapters

| Adapter | Lane | Behavior |
| --- | --- | --- |
| `mock` | `mock_backend` | Renders the scenario's expected answer as a server-sent event stream: a keepalive comment, `answer_start`, the answer in 48-character `answer_delta` chunks, a `citation` frame and `done`. The answer is wrapped in a `<think>` reasoning span and uses CRLF line endings, so every mock capture exercises normalization. The same scenario and batch always produce the same capture ID and bytes. |
| `kleos_chat` | `production_observation` | Sends `POST /api/v1/career/projects/chat` as a multipart form built from `CHAT_FORM_DEFAULTS` plus the question. |
| `kleos_json` | `production_observation` | Reads a plain-JSON view: `briefing` (`/api/v1/mission/briefing`), `notifications` (`/api/v1/notifications`) or `memory` (`/api/v1/memory`). |

`CHAT_FORM_DEFAULTS` turns off every optional feature of the chat endpoint.
`thinking`, `knowledge_base`, `web_search`, `deep_research` and
`ask_tool_permission` are `false`, `model_mode` is `small`, and the Drive,
GitHub, GitLab, Notion, Slack, Discord, Dropbox and OneDrive integrations are
disabled. An enabled integration would pull more of the user's connected
accounts into the capture. There is no option or flag to change these values,
and a test asserts that `capture_backend.py` does not reference them.

The `kleos_chat` and `kleos_json` adapters require an HTTP transport from the
`collect` extra and raise `CaptureError` without one. `capture_backend.py` does
not construct a transport yet, so the capture CLI currently runs only the `mock`
adapter.

## Running a capture

```bash
python scripts/capture_backend.py --adapter mock --out-batch batch-001
```

| Option | Effect |
| --- | --- |
| `--out-batch <id>` | Required. Batch directory under `staging/raw/`. |
| `--adapter <name>` | Adapter to use. Default `mock`. |
| `--family <id>` | Limit to one family. Can be repeated. |
| `--limit <n>` | Stop after `n` requests |
| `--scenarios <path>` | Catalog directory. Default `scenarios/`. |
| `--dry-run` | Run the production guard and stop |

Each batch directory contains one record per capture and a `_batch.json` summary
with the batch ID, lane, adapter, endpoint, pipeline version, families and
request, capture and failure counts. A capture ID that repeats within a batch is
reported as a failure instead of overwriting the earlier record. The script
exits with code 1 if any capture fails.

## Production capture guard

The guard in `collection/guard.py` applies when `--base-url` points to a
non-local target. Local targets are the `mock://` scheme and the hosts
`127.0.0.1`, `localhost`, `0.0.0.0`, `*.local` and `*.test`.

A non-local capture requires all four conditions:

| Condition | Provided by |
| --- | --- |
| Explicit opt-in on the command line | `--allow-production` |
| Explicit opt-in in the environment | `KLEOS_ALLOW_PRODUCTION_CAPTURE=1` |
| Typed confirmation | `--confirm "I understand this captures real personal data"` |
| Not running in automation | The `CI` environment variable is unset or empty. No option overrides this condition. |

Each condition guards against a different accident, such as a flag copied into a
script or an environment variable inherited from a shell profile. If any
condition fails, the guard raises `ProductionGuardError`, lists which conditions
were met and which were not, and exits with code 1. The error message never
explains how to disable the guard.

An authorized run writes `_capture_authorization.json` into the batch directory
with the target host, the time, the operator role, the scenario families, the
expected capture count and the conditions met. The file records the host rather
than the full URL, because a URL can carry a token in its query string.

`--base-url` is currently read only by the guard. The mock adapter ignores it,
so an authorized run of the current CLI still captures through the mock adapter.

## HTTP transport

`HttpTransport` in `collection/transport.py` is the HTTP layer for the backend
adapters. It requires the `collect` extra (`httpx`).

| Behavior | Setting |
| --- | --- |
| Retried | HTTP 408, 429, 500, 502, 503, 504, timeouts and connection errors |
| Never retried | HTTP 400, 401, 403, 404, 405, 422 |
| Attempts | Up to 4 per request |
| Retry budget | 20 retries shared by all requests through one transport instance |
| Backoff | Full jitter, a random delay between 0 and `min(0.5 × 2^attempt, 20)` seconds |
| `Retry-After` | Honored in seconds or HTTP-date form, capped at 20 seconds |
| Rate limit | Token bucket at 20 requests per minute with a burst of 5, applied to every attempt |
| Timeout | 60 seconds |
| Redirects | Not followed |
| Tracing | Each request carries an `X-Request-Id` header |

Status codes that fail identically on a second attempt are not retried, and a
shared budget stops a degraded backend from multiplying the request load.
Server-sent event responses are parsed into frames once received, and an `error`
frame or a missing `done` frame raises `CaptureError`.

## Logging

| Source | Logged fields |
| --- | --- |
| Capture runner | Capture ID, family, point, status, latency, frame count and byte count |
| HTTP transport, on a retryable failure | Request ID, attempt, status, wait time and remaining retry budget |

Request and response bodies are not logged. Credentials are wrapped in
`SafeSecret`, described in [SECURITY.md](../SECURITY.md#credential-handling).

## Related documentation

- [../PRIVACY.md](../PRIVACY.md): why production captures are never promoted
- [staging.md](staging.md): normalization and the records that follow capture
- [scenarios.md](scenarios.md): the catalog that capture requests come from
- [../SECURITY.md](../SECURITY.md): credential handling
