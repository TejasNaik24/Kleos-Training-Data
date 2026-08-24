"""The fourteen promotion gates.

Promotion is the only path into a dataset. Every gate runs — there is no
short-circuit — so one run tells you everything wrong with a candidate rather
than one thing at a time. The cost is a few wasted checks on a doomed candidate;
the benefit is that fixing three problems takes one cycle instead of three.

**Mandatory-ness is computed from this table, not declared beside it.**
``GateSpec.bypassable`` defaults to ``False``, and ``MANDATORY_GATE_IDS`` is
derived from :data:`GATES`. A gate added without thinking about it is therefore
mandatory, and making one bypassable requires an explicit edit that shows up in
review. The alternative — a hand-maintained list of mandatory ids — drifts the
first time somebody adds a gate and forgets to update it, and it drifts silently
in the permissive direction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Final

from kleos_training_data.promotion.context import PromotionContext


class GateStatus(str, Enum):
    """Outcome of one gate."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    #: Bypassed by an explicit policy. Only ever possible for a bypassable gate.
    BYPASSED = "BYPASSED"


@dataclass(frozen=True)
class GateOutcome:
    """What a gate concluded."""

    status: GateStatus
    message: str = ""
    evidence: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.evidence is None:
            object.__setattr__(self, "evidence", {})

    @property
    def failed(self) -> bool:
        return self.status is GateStatus.FAIL


def ok(message: str = "") -> GateOutcome:
    return GateOutcome(GateStatus.PASS, message)


def fail(message: str, **evidence: str) -> GateOutcome:
    return GateOutcome(GateStatus.FAIL, message, evidence)


def warn(message: str, **evidence: str) -> GateOutcome:
    return GateOutcome(GateStatus.WARN, message, evidence)


GateFn = Callable[[PromotionContext], GateOutcome]


@dataclass(frozen=True)
class GateSpec:
    """One gate: what it checks, and whether it may ever be skipped."""

    id: str
    description: str
    check: GateFn
    #: Defaults to False. A new gate is mandatory unless somebody deliberately
    #: says otherwise, in a diff a reviewer will see.
    bypassable: bool = False
    #: Whether a failure here is a privacy incident rather than a quality issue.
    #: Drives exit code 4 and the incident-response path.
    privacy: bool = False


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------


def _g01_staging_integrity(ctx: PromotionContext) -> GateOutcome:
    if not ctx.candidate.hash_matches():
        return fail(
            "the staged candidate was modified outside the pipeline",
            candidate=ctx.candidate.candidate_id,
        )
    if ctx.privacy is None:
        return fail(
            "no privacy record accompanies this candidate",
            hint="run scripts/sanitize_candidates.py",
        )
    return ok(f"record hash verified, privacy record present ({ctx.privacy_status})")


def _g02_schema_valid(ctx: PromotionContext) -> GateOutcome:
    error = ctx.schema_error()
    if error:
        return fail("the payload does not satisfy the public contract", error=error)
    return ok("validates against the contract mirror")


def _g03_id_integrity(ctx: PromotionContext) -> GateOutcome:
    if not ctx.ledger.verify(ctx.payload, ctx.candidate.candidate_id):
        return fail(
            "the id does not match the content it names",
            candidate=ctx.candidate.candidate_id,
            hint="the content changed after the id was minted",
        )
    return ok("id re-derives from the content")


def _g04_secret_scan(ctx: PromotionContext) -> GateOutcome:
    blocking = [d for d in ctx.residual_detections if d.severity == "block"]
    if blocking:
        return fail(
            f"{len(blocking)} secret(s) survive in the payload",
            rules=", ".join(sorted({d.rule_id for d in blocking})),
        )
    return ok("no secret detected")


def _g05_pii_scan(ctx: PromotionContext) -> GateOutcome:
    unresolved = [d for d in ctx.residual_detections if d.severity in {"redact", "review"}]
    if unresolved:
        return fail(
            f"{len(unresolved)} unresolved PII detection(s) survive sanitization",
            rules=", ".join(sorted({d.rule_id for d in unresolved})),
        )
    return ok("no unresolved PII")


def _g06_surrogate_integrity(ctx: PromotionContext) -> GateOutcome:
    problems = ctx.surrogate_problems()
    if problems:
        return fail("sanitization left something behind", problems="; ".join(problems))
    return ok("no vault literal or placeholder residue")


def _g07_private_fact(ctx: PromotionContext) -> GateOutcome:
    verdict = ctx.fact_verdict
    if verdict == "policy_like":
        return ok("no private-fact signal")
    if verdict == "fact_teaching":
        return fail(
            "the example appears to teach a fact about a real person",
            verdict=verdict,
            signals=ctx.fact_signal_ids(),
        )
    # needs_fact_review: a human must have decided it, on the record.
    if ctx.human is not None and ctx.human.gates.policy_not_facts == "PASS":
        return ok(f"verdict {verdict}, cleared by a human on the record")
    return fail(
        "private-fact risk needs an explicit human decision",
        verdict=verdict,
        signals=ctx.fact_signal_ids(),
    )


def _g08_review_present(ctx: PromotionContext) -> GateOutcome:
    if ctx.human is None:
        return fail("no human decision exists for this candidate")
    if not ctx.human.signature_valid():
        return fail("the human decision's signature does not verify")
    if not ctx.human.applies_to(ctx.candidate.content_hash):
        return fail(
            "the human decision was made about different content",
            reviewed=ctx.human.content_hash[:16],
            candidate=ctx.candidate.content_hash[:16],
            hint="the candidate was edited after approval; it needs a fresh review",
        )
    return ok("a signature-valid decision covers this exact content")


def _g09_review_approved(ctx: PromotionContext) -> GateOutcome:
    if ctx.human is None:
        return fail("no human decision exists")
    if ctx.human.decision != "approve":
        return fail(
            f"the human decision is {ctx.human.decision!r}",
            reasons=", ".join(r.value for r in ctx.human.reason_codes) or "(none given)",
        )
    verdict = ctx.verdict()
    if not verdict.approved:
        return fail(f"the combined verdict is {verdict.decision!r}", explain=verdict.explain())
    return ok(verdict.explain())


def _g10_provenance(ctx: PromotionContext) -> GateOutcome:
    lane = ctx.candidate.lane
    if not lane.promotable:
        return fail(
            f"lane {lane.value!r} can never be promoted",
            hint=(
                "the KLEOS backend answers from the authenticated user's own "
                "stored data, so a production capture is one person's private "
                "data. Use it as seed material for a new scenario instead."
            ),
        )
    if ctx.source not in ctx.allowed_sources:
        return fail(
            f"source {ctx.source!r} is not valid for lane {lane.value!r}",
            expected=lane.contract_source,
        )
    return ok(f"lane {lane.value}, source {ctx.source}")


def _g11_coverage_axes(ctx: PromotionContext) -> GateOutcome:
    axes = ctx.payload.get("variation_axes") or {}
    if not axes.get("domain"):
        return fail("variation_axes.domain is required by the public contract")
    unregistered = ctx.unregistered_axes()
    if unregistered:
        return warn(
            f"piloting unregistered axis/axes: {', '.join(unregistered)}",
            axes=", ".join(unregistered),
        )
    return ok(f"{len(axes)} axis/axes declared")


def _g12_corpus_dedup(ctx: PromotionContext) -> GateOutcome:
    findings = ctx.duplicate_findings()
    fatal = [f for f in findings if f.fatal]
    if fatal:
        return fail(
            f"{len(fatal)} exact or normalized duplicate(s) already promoted",
            existing=", ".join(sorted({f.existing_id for f in fatal})),
        )
    near = [f for f in findings if not f.fatal]
    if near:
        return warn(
            f"{len(near)} near-duplicate(s) above threshold",
            existing=", ".join(f"{f.existing_id}@{f.similarity:.2f}" for f in near[:5]),
        )
    return ok(f"no duplicate among {len(ctx.corpus)} promoted example(s)")


def _g13_eval_leakage(ctx: PromotionContext) -> GateOutcome:
    findings = ctx.leakage_findings()
    if findings:
        return fail(
            f"{len(findings)} overlap(s) with held-out evaluation material",
            kinds=", ".join(sorted({f.kind.value for f in findings})),
            existing=", ".join(sorted({f.existing_id for f in findings})[:5]),
        )
    return ok(f"no overlap with {len(ctx.eval_corpus)} evaluation example(s)")


def _g14_contract_render(ctx: PromotionContext) -> GateOutcome:
    extras = ctx.disallowed_metadata_extras()
    if extras:
        return fail(
            "metadata carries keys that would ship inside train.jsonl",
            keys=", ".join(extras),
            hint="a pointer back into staging/ must not survive into a release",
        )
    if not ctx.round_trips():
        return fail("the example does not survive a serialize/re-parse round trip")
    return ok("renders and re-parses identically")


#: The table. Order is execution order, and it is meaningful: cheap structural
#: checks first, so a malformed candidate fails before an expensive corpus scan.
GATES: Final[tuple[GateSpec, ...]] = (
    GateSpec("G01_STAGING_INTEGRITY", "The staged record is intact.", _g01_staging_integrity),
    GateSpec("G02_SCHEMA_VALID", "The payload satisfies the public contract.", _g02_schema_valid),
    GateSpec("G03_ID_INTEGRITY", "The id re-derives from the content.", _g03_id_integrity),
    GateSpec("G04_SECRET_SCAN", "No secret survives.", _g04_secret_scan, privacy=True),
    GateSpec("G05_PII_SCAN", "No unresolved PII survives.", _g05_pii_scan, privacy=True),
    GateSpec(
        "G06_SURROGATE_INTEGRITY",
        "No vault literal or placeholder residue survives.",
        _g06_surrogate_integrity,
        privacy=True,
    ),
    GateSpec(
        "G07_PRIVATE_FACT",
        "The example teaches a policy, not a fact about a person.",
        _g07_private_fact,
        privacy=True,
    ),
    GateSpec(
        "G08_REVIEW_PRESENT", "A valid review covers this exact content.", _g08_review_present
    ),
    GateSpec("G09_REVIEW_APPROVED", "The review approves it.", _g09_review_approved),
    GateSpec("G10_PROVENANCE", "The lane and source permit promotion.", _g10_provenance),
    # The one genuinely bypassable gate. Piloting an unregistered axis is a
    # legitimate thing to do while deciding whether to register it upstream, and
    # a missing `domain` is caught by G02 anyway since the contract requires it.
    GateSpec(
        "G11_COVERAGE_AXES",
        "Required axes are present.",
        _g11_coverage_axes,
        bypassable=True,
    ),
    # Mandatory, despite an earlier draft marking it bypassable "for near
    # duplicates only". That reasoning was wrong about its own mechanism: a
    # bypassed gate does not run at all, so the bypass would have skipped exact
    # and normalized duplicates too.
    #
    # No bypass is needed. The gate already distinguishes them itself — exact and
    # normalized duplicates FAIL, near-duplicates WARN, and a WARN does not block
    # unless `strict_warnings` is set. A human weighs the near-duplicate; nobody
    # gets to wave through an exact one.
    GateSpec(
        "G12_CORPUS_DEDUP",
        "Not a duplicate of something already promoted.",
        _g12_corpus_dedup,
    ),
    GateSpec("G13_EVAL_LEAKAGE", "No overlap with evaluation material.", _g13_eval_leakage),
    GateSpec("G14_CONTRACT_RENDER", "Renders to the exact public contract.", _g14_contract_render),
)

#: Derived from the table, never hand-maintained. This is mechanism #1 of four.
MANDATORY_GATE_IDS: Final[frozenset[str]] = frozenset(
    spec.id for spec in GATES if not spec.bypassable
)

#: The only ids ``--force`` may ever cover.
BYPASSABLE_GATE_IDS: Final[frozenset[str]] = frozenset(spec.id for spec in GATES if spec.bypassable)

#: Gates whose failure is a privacy incident.
PRIVACY_GATE_IDS: Final[frozenset[str]] = frozenset(spec.id for spec in GATES if spec.privacy)

GATES_BY_ID: Final[dict[str, GateSpec]] = {spec.id: spec for spec in GATES}
