# Changelog

Dataset releases and pipeline changes. Dataset versions are immutable — a
correction is a new version, never an edit.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [kleos-policy-v0.0.2] — 2026-09-01

Candidate release, pending external review. 1,152 examples across 17 scenario
families and all seven tasks; 721 train / 157 validation / 274 test, held out on
`format=json`. Audit: `reports/kleos-policy-v0.0.2-audit.md`.

### Fixed — the content layer

Six defects that produced examples which were schema-valid, privacy-clean and
gate-passing while teaching something other than their label. Nothing in the
pipeline objected, because nothing was checking whether rendered content matched
the task it was filed under.

- `respect_workspace_scope` tested for the literal substring `"out of scope"` in
  an item's `detail`; no generated detail ever contained it, so the scope branch
  was unreachable and **every** `workspace_reasoning` example fell through to
  plain deadline ranking. Items now carry an explicit `scope` field.
- `render_system_prompt` accepted a situation and ignored it, giving all seven
  tasks the prioritization instruction — `tool_routing` examples were told "You
  help prioritize competing work" and handed candidate sources. Replaced with a
  registered `Framing` per family.
- `deadline_days` rendered as a due date under every framing, so memory records
  read "recorded as of in 4 days". Time now renders by the framing's sense:
  `due`, `age` or `staleness`.
- Distractors drawn by hashing each slot independently collided, printing the
  same sentence twice in one prompt on 29 of 150 examples.
- `difficulty` was inverted: `easy` mapped to the spread that bunches candidates
  together, so "easy" points were the hardest to separate.
- `"due in about 1 weeks"`.

### Fixed — silent data loss

- The mock adapter derived `capture_id` from the axes, which a `paraphrase` does
  not change, so both members of every `count: 2` paraphrase group produced one
  id and the second overwrote the first. A 1,152-request batch wrote 948 files
  while reporting 1,152 captured. The prompt is now in the seed, and the runner
  treats a repeated `capture_id` as a failure rather than a write.
- Normalization's request lookup was keyed on
  `family|point|kind|perturbation_of` — also missing the ordinal — so 204
  captures reported "no prompt in the catalog matches this capture". Now keyed
  on `(family, point)` with every request at that point retained.
- The `person_name` bigram rule used `\s+`, which matches a newline, so
  `"Active workspace: Research\nBlue Harbor"` matched `"Research Blue"` and
  redaction corrupted 97 prompts. Narrowed to `[ \t]+`; no scanner exception was
  added.

### Added

- Three registered policies: `defer_to_explicit_statement` (an explicit
  statement outranks an inference), `ask_when_underdetermined` (a missing
  variable is not a close call), `prefer_least_privilege_source` (among sources
  that can answer, the one reaching least far).
- Six framings — `priority`, `briefing`, `routing`, `memory`, `context`,
  `workspace` — each with its own instruction, nouns and time sense.
- Ten scenario families, taking the catalog from 7 to 17 and closing the
  `task × domain` grid to 28/28.
- `tests/test_framing_and_content.py` — 23 tests, each pinned to one of the
  defects above and verified by mutation to fail when it is reintroduced.

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
