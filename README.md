# kleos-training-data

**PRIVATE.** Never publish this repository, its history, or the contents of
`staging/`, `vault/` or `releases/`.

Collection, review, sanitization, promotion and versioned release of training
datasets for the public [`kleos-models`](https://github.com/TejasNaik24/Kleos-Models)
repository.

---

## 1. What is this repository?

A pipeline that turns deliberately-constructed KLEOS scenarios into auditable,
sanitized, schema-valid, **immutable** training datasets.

It exists to serve one research principle:

> **Train a generalizable decision policy, not private facts about a person.**

A good training target:

> Given several competing opportunities, prioritize the one with the closest
> meaningful deadline, the strongest evidence of impact, and the highest
> consequence of delay.

A bad one:

> Tejas should prioritize his Motorola project because he has an internship there.

The first transfers to anyone. The second is a fact about one individual that a
model can memorize and later surface. Every gate in this repository exists to
keep the second kind out.

## 2. Why is it private?

Because captures can contain real personal data, and because the sanitization
keys that map a fictional surrogate back to a real person live here in `vault/`.

Private git is **not** a secure vault. A clone, a fork, or a future collaborator
sees everything in history, and history is forever. The `.gitignore`, the
pre-commit scanner and the promotion gates are a safety net under human
judgement, not a replacement for it.

## 3. What is safe to commit?

| Safe | Never |
| --- | --- |
| `src/`, `scripts/`, `tests/`, `docs/` | Anything under `staging/`, `vault/`, `releases/`, `reports/` |
| `scenarios/**/*.yaml` | `.env`, tokens, cookies, JWTs, database URLs with passwords |
| `configs/*.yaml` | Raw or sanitized captures |
| `data/surrogates/` (fictional names) | Real user exports, production data |
| `data/fixtures/` (obviously fake) | `*.pem`, `*.key`, `*.p12` |

The four runtime zones are deny-by-default in `.gitignore` — only a top-level
`.gitkeep` is tracked. Verify before every commit:

```bash
python scripts/check_no_private_data.py .        # exit 4 if anything is found
python scripts/check_no_private_data.py --install-hook
```

Finished datasets are shipped **by path**, not by git. See `DATA_GOVERNANCE.md`.

## 4. How do I collect data?

Three lanes, with different promotion rights. This distinction is the most
important one in the repository.

| Lane | Source | Can promote? |
| --- | --- | --- |
| `synthetic` | Scenario YAML, no backend | Yes |
| `mock_backend` | Deterministic offline adapter | Yes |
| `production_observation` | The real KLEOS backend | **Never** |

The KLEOS backend answers from the *authenticated user's own* stored projects,
memories and notifications. It is not a scenario simulator — every capture from
it is one person's private data. Such a capture is readable in a reviewer packet
as **seed material only**: a human reads it, learns what situation actually
occurs, and writes a *new* generalized scenario. The resulting example is
`synthetic_seeded` and has no textual descent from the capture.

```bash
python scripts/init_workspace.py
python scripts/validate_scenarios.py --strict
python scripts/capture_backend.py --adapter mock --out-batch slice-001
python scripts/normalize_captures.py --batch slice-001
```

Capturing against production requires four independent conditions to hold at
once and is refused outright in CI. See `docs/collection.md`.

## 5. How do I review data?

Sanitize first, then review. Deterministic gates are authoritative; an LLM
assists, and a human owns the final decision.

```bash
python scripts/sanitize_candidates.py --batch slice-001
python scripts/build_review_packet.py --batch slice-001
python scripts/run_llm_review.py --packet-id pk-001 --reviewer mock
python scripts/record_decision.py --candidate <id> --decision approve
```

Six scored dimensions (0–4) and four PASS/FAIL hard gates. A failing hard gate
overrides any score — that is enforced as a model invariant, so a verdict with a
failing gate and `decision="approved"` cannot be constructed at all. A human may
override scores; a human may **not** approve over a privacy or private-fact
failure. See `docs/review.md`.

## 6. How do I promote data?

```bash
python scripts/promote_examples.py --batch slice-001
```

Fourteen ordered gates, all of which run — no short-circuit, so one run tells you
everything wrong with a candidate. Thirteen are non-bypassable, including every
privacy, schema, review, dedup and leakage gate. `--force` maps to a named
constant covering only the one bypassable gate (`G11_COVERAGE_AXES`, for piloting
an unregistered axis); a policy object that bypasses a mandatory gate cannot be
constructed. See `docs/staging.md`.

## 7. How do I build a dataset?

```bash
python scripts/build_release.py --version kleos-policy-v0.1.0 \
    --strategy scenario_family_holdout --seed 42
```

Group-aware splitting keeps every scenario family and its perturbations on one
side of the boundary. OOD holdouts are **declared in advance**, not discovered by
a seed — an OOD claim you cannot state in advance is not an OOD claim.

Releases are immutable. There is deliberately no `--force`: if the content needs
to change, the version string needs to change. See `docs/dataset-lifecycle.md`.

## 8. How do I verify compatibility with kleos-models?

```bash
python scripts/verify_release.py --release releases/kleos-policy-v0.1.0 --strict
python scripts/check_contract_compat.py --release releases/kleos-policy-v0.1.0 --strict
```

Then, against the real public repo:

```bash
python ../Kleos-Models/scripts/validate_dataset.py \
    --dataset releases/kleos-policy-v0.1.0 --require-reviewed
```

This repository does **not** import `kleos-models` at runtime. It mirrors the
contract in `src/kleos_training_data/contract/`, pinned to a specific public
commit, and proves the mirror correct with differential tests that compare our
JSONL bytes, split assignments and validation decisions against the real thing.
See `docs/compatibility.md`.

## 9. How do I run tests?

```bash
make install        # creates .venv, installs the offline dev environment
make check          # privacy + lint + format + typecheck + test
make test-fast      # skips the end-to-end pipeline test
```

Differential tests need the pinned public package and skip without it:

```bash
make install-compat
```

The entire pipeline and test suite runs offline with no credentials. That is
deliberate — a pipeline that needs a live backend to be tested is one whose first
end-to-end run creates private data before any gate exists to catch it.

## 10. Where are secrets stored?

Environment variables, loaded from a git-ignored `.env`. Start from
`.env.example`; every variable in it is optional, and none is needed for the
offline pipeline.

Credentials are wrapped in `SafeSecret`, which renders as
`<secret len=64 sha256=1a2b3c4d>` under `str`, `repr`, f-strings and
`json.dumps(default=str)`. Only `.reveal()` returns the value, and a test asserts
that is called in exactly one non-test file.

Check what would be verified without printing any value:

```bash
python scripts/check_contract_compat.py
python scripts/init_workspace.py --check
```

(`scripts/doctor.py`, a single consolidated diagnostic, arrives in Phase J.)

---

## The pipeline

```
scenarios/                    declared situations, not literal conversations
    ↓  generate | capture (mock or real backend)
staging/raw/                  UNVERIFIED — never trusted, never committed
    ↓  normalize
staging/normalized/           contract-shaped, pre-sanitization
    ↓  sanitize               secrets → reject; PII → redact → fictional surrogate
staging/sanitized/            + a privacy result naming every detection
    ↓  review                 LLM assists, deterministic gates decide, human owns it
staging/review/               packets, machine reviews, signed human decisions
    ↓  promote                14 gates, 11 non-bypassable
staging/promoted/             the pool a release draws from
    ↓  build_release          dedup → leakage → group-aware split → manifest → seal
releases/<version>/           IMMUTABLE: manifest.json + train.jsonl [+ val/test]
    ↓
kleos-models/scripts/train.py --dataset <that directory>
```

Rejections go to `staging/rejected/` with a closed-vocabulary reason code, so
"which failure dominates?" is a question you can actually answer.

## Status

The vertical slice is complete and provable offline:

```bash
make slice-clean slice
```

That runs scenario → mock capture → normalize → sanitize → review → promote →
split → seal → verify → compatibility check, with no credentials, no network and
no private data. CI runs the same target.

| Built | |
| --- | --- |
| Workspace, three zones, deny-by-default ignore rules | proven against real `git` |
| Zero-dependency private-data scanner | exits 4 on a planted key |
| Contract mirror (schemas, writer, splitter, dedup) | byte-identical to the pinned public repo |
| Content-derived stable ids | position- and provenance-independent |
| Scenario system with policy-derived targets | perturbations proven decision-preserving |
| Mock capture lane + production guard | four independent conditions, un-overridable in CI |
| Four-layer privacy detection + surrogates | zero false positives on the synthetic corpus |
| Review rubric with structural gate dominance | a failing gate cannot be approved over |
| 14 promotion gates | 13 mandatory, `--force` cannot reach them |
| Immutable sealed releases | no `--force`, verified by independent re-derivation |

Not yet built (Phase J): the real backend adapters, the `AnthropicReviewer`, the
full seven-task scenario catalog, coverage reporting, and the remaining docs.
See `CHANGELOG.md`.

## Documentation

| Document | Covers |
| --- | --- |
| `PRIVACY.md` | Capture lanes, consent basis, what sanitization does and does not fix |
| `SECURITY.md` | Secret handling, incident response entry point |
| `DATA_GOVERNANCE.md` | Retention, deletion requests, release custody |
| `DATASET_CONTRACT.md` | The exact shape a release must have |
| `docs/architecture.md` | Why the zones are separate |
| `docs/collection.md` | Adapters, rate limits, the production guard |
| `docs/staging.md` | Record formats, the promotion gates |
| `docs/review.md` | The rubric and why hard gates dominate |
| `docs/privacy.md` | Detection layers, surrogates, private-fact review |
| `docs/dataset-lifecycle.md` | Versioning, splitting, immutability |
| `docs/compatibility.md` | The contract mirror and the differential tests |
| `docs/research-protocol.md` | Coverage targets, OOD design, consistency groups |
| `docs/incident-response.md` | Exposed secret, promoted PII, consent problem |
