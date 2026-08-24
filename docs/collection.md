# Collection

## The three lanes

This is the most consequential distinction in the repository.

| Lane | Content | Promotable |
| --- | --- | --- |
| `synthetic` | Rendered locally from a scenario | yes |
| `mock_backend` | Deterministic offline adapter | yes |
| `production_observation` | The real KLEOS backend | **never** |

`POST /api/v1/career/projects/chat` answers from the *authenticated user's own*
stored projects, memories and notifications. It is not a scenario simulator: a
capture from it is that person's private data whatever the prompt asked. Gate
`G10_PROVENANCE` rejects the lane outright.

Such a capture is **seed material**. A human reads it in a reviewer packet,
learns what situation genuinely arises, and writes a *new* generalized scenario.
The resulting example is `synthetic_seeded`, has a different content hash, and
has no textual descent from the capture. That keeps the research value — knowing
which situations actually occur — without training on one.

## The mock adapter

Not a pass-through. It emits real SSE frames, chunks the answer across
`answer_delta` events, wraps it in a `<think>` span and uses CRLF line endings —
so normalization has genuine work to do. A mock that gives the next stage nothing
to do proves nothing about it.

It is deterministic: the same scenario and batch produce the same capture id and
the same bytes, so re-running a batch does not fill staging with near-duplicates.

```bash
python scripts/capture_backend.py --adapter mock --out-batch slice-001
```

## The real adapters

Behind the `collect` extra, and they refuse to instantiate without a transport —
so the offline lane cannot accidentally be pointed at a network client.

`kleos_chat` sends a multipart form with **every integration forced off**:
`web_search`, `deep_research`, Drive, GitHub, GitLab, Notion, Slack, Discord,
Dropbox, OneDrive. Any integration that fires pulls more of the operator's
connected accounts into a capture that is already private data, for no research
value. Turning one on means editing `CHAT_FORM_DEFAULTS`, which is a diff a
reviewer sees.

`kleos_json` covers the plain-JSON views — mission briefing, notifications,
memory. A different response discipline, which is why the adapter boundary exists
rather than one client full of branches.

## The production guard

Capturing against a non-local URL requires **all four**:

1. `--allow-production`
2. `KLEOS_ALLOW_PRODUCTION_CAPTURE=1`
3. the exact confirmation phrase, typed
4. `CI` unset

Four, because any one alone is something a person can do by accident or a script
can inherit. The fourth is **not overridable**: an automated production capture is
never legitimate, and a flag that could disable that check would end up set in a
workflow file and forgotten.

The refusal names which conditions failed. It deliberately never names a way to
turn the guard off.

A permitted run writes `_capture_authorization.json` beside the batch, recording
the host — never the URL, which can carry a token in its query string.

## Transport

- Retry only `{408, 429, 500, 502, 503, 504}` and timeouts. A 401 or 422 fails
  identically on a second attempt; retrying wastes the budget and delays the real
  error.
- A **per-batch** retry budget, not per-request. Per-request limits let a
  degraded backend turn 50 scenarios into 200 requests against a service already
  struggling.
- Full-jitter backoff, honouring `Retry-After` in both integer and HTTP-date form.
- Token-bucket rate limiting, conservative by default.
- SSE streams are **never** retried: a stream that failed midway already
  delivered a partial answer, and re-requesting produces a second, differently
  truncated one.

## Logging

Counts and timings only. Never a request body, never a response body, at any
level, behind any flag. A log line is the easiest way for private content to
escape, because logs get pasted into issues.

```
request_id=... attempt=1 status=200 waited_ms=48 budget_left=20
```

Credentials are wrapped in `SafeSecret`, which renders as
`<secret len=64 sha256=1a2b3c4d>` under `str`, `repr`, f-strings and
`json.dumps(default=str)`. `.reveal()` is called in exactly one non-test file.
