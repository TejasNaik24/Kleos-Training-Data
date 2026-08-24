# Changelog

Dataset releases and pipeline changes. Dataset versions are immutable — a
correction is a new version, never an edit.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added — pipeline

- Three-zone workspace (`staging/`, `vault/`, `releases/`) with deny-by-default
  ignore rules, proven against real `git` rather than a re-implementation.
- Zero-dependency private-data scanner, runnable before any install and as a
  pre-commit hook.
- Contract mirror — schemas, JSONL writer, splitter, duplicate detection — pinned
  to kleos-models `12361d53` and verified byte-identical by differential tests.
- Content-derived example ids: position-independent, provenance-independent, and
  sensitive to a one-character content change.
- Scenario system where the training target is **computed from a registered
  policy**, never authored beside the prompt and never taken from a model's own
  output.
- Seven scenario families, one per registered task.
- Mock capture lane emitting realistic SSE frames, and a production guard
  requiring four independent conditions — one of which cannot be overridden.
- Four-layer privacy detection, redaction, and per-family fictional surrogates.
- Private-fact assessment, modelled separately from PII because it has no
  mechanical fix.
- Review rubric where a failing hard gate cannot be approved over — enforced as a
  model invariant on every construction path.
- Fourteen promotion gates, thirteen mandatory, with four independent mechanisms
  preventing `--force` from reaching one.
- Immutable sealed releases with `RELEASE.lock`, verified by independent
  re-derivation.
- Coverage reporting over joint axis structure.
- Real backend adapters (`kleos_chat`, `kleos_json`) and an Anthropic reviewer,
  both behind optional extras and both refused in CI.
- `doctor.py`, which reports credential *presence* and never a value.

### Fixed — during the initial build

Each of these produced plausible-looking output, which is why they are recorded:

- Privacy detection excerpts reprinted their *neighbours* — the context window
  around one hit contained the next one. Found in `detect.py`, then again in
  `facts.py`.
- A vault entry matching inside an email address applied two overlapping
  replacements and corrupted the text.
- 32 false PII flags and 12 false private-fact accusations on our own synthetic
  corpus, from flagging the fictional names we generated.
- Sanitization's own surrogates re-triggered the rules that produced them: a
  redacted `@dana` became `@rowan.baxter`, which is still a handle.
- Sanitization and the reviewer disagreed about the same bytes, because a shared
  allowlist was opt-in rather than the default.
- Perturbations that did not perturb — 24 candidates yielding 19 unique ids.
- A `context_order` perturbation whose shuffle landed on the order the base
  already used, producing an identical prompt that silently overwrote its sibling
  at normalization.
- `sampling: stratified` did not stratify: six points over three formats gave
  20 json, 4 prose, 0 bullets.
- A gate marked bypassable "for near-duplicates only", which would in fact have
  skipped exact duplicates too, since a bypassed gate does not run at all.
- Two tests passing vacuously: identical unicode literals, and group ids that
  were all singletons.

### Known upstream

- `kleos-models/scripts/check_no_private_data.py:234-239` reprints short lines
  containing the secret it just found, into CI logs. Recorded in
  `contract/pin.py:KNOWN_UPSTREAM_ISSUES`; our port excises the matched span.

## Dataset releases

### kleos-policy-v0.0.1

The vertical-slice proof, not a research dataset.

- 150 examples across seven scenario families, one per registered task
- split `format_holdout` on `json`, seed 42 — 90 train / 10 validation / 50 test
- all `source: synthetic`, all `quality_status: reviewed`
- `contains_private_data: false`, computed from a byte-level scan
- verified by `verify_release --strict` (32 checks) and loaded by the public
  `validate_dataset.py` with zero errors
