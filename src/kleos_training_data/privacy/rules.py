"""Detection rules: patterns, severities and allowlists.

Four layers, in the order they run. The order is the design:

1. **secrets** — severity ``block``. Never auto-redacted, always a hard reject.
   A credential in a candidate means the capture path itself is compromised;
   quietly swapping the string for a placeholder would hide that and let the
   pipeline carry on as though nothing happened.
2. **pii** — severity ``redact``. Mechanically replaceable: an email is an email
   wherever it appears.
3. **vault** — severity ``redact``. Operator-supplied literals that regex cannot
   find, such as a real employer whose name reads like an ordinary noun.
4. **structural** — severity ``review``. Heuristics with real false-positive
   rates (capitalized bigrams, organization suffixes). They flag for a human
   rather than deciding.

The secret patterns are duplicated from ``scripts/check_no_private_data.py``,
which imports nothing so it can run before any install.
``tests/test_privacy_rules.py`` asserts this list is a superset of that one — a
library scanner looser than the pre-commit scanner is how a secret reaches a
release.
"""

from __future__ import annotations

import re
from typing import Final, NamedTuple

#: Bumped whenever a rule changes in a way that would produce different output
#: from identical input. Recorded on every sanitized candidate, so a release can
#: name the ruleset that produced it.
RULESET_VERSION: Final[str] = "privacy-rules-v1"

#: Severity ordering, most severe first.
SEVERITIES: Final[tuple[str, ...]] = ("block", "redact", "review", "warn")


class Rule(NamedTuple):
    """One detection rule."""

    rule_id: str
    kind: str
    layer: str
    severity: str
    pattern: re.Pattern[str]
    #: Placeholder slot this rule redacts into. ``None`` for block/warn rules,
    #: which are never replaced.
    slot: str | None = None


# ---------------------------------------------------------------------------
# Layer 1 — secrets. Severity `block`.
# ---------------------------------------------------------------------------

SECRET_RULES: Final[tuple[Rule, ...]] = (
    Rule(
        "secret.aws_access_key.v1",
        "aws_access_key",
        "secret",
        "block",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    ),
    Rule(
        "secret.openai_key.v1",
        "openai_key",
        "secret",
        "block",
        re.compile(r"\bsk-[A-Za-z0-9]{32,}"),
    ),
    Rule(
        "secret.anthropic_key.v1",
        "anthropic_key",
        "secret",
        "block",
        re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    ),
    Rule("secret.hf_token.v1", "hf_token", "secret", "block", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    Rule(
        "secret.github_token.v1",
        "github_token",
        "secret",
        "block",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"),
    ),
    Rule(
        "secret.slack_token.v1",
        "slack_token",
        "secret",
        "block",
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    ),
    Rule(
        "secret.google_api_key.v1",
        "google_api_key",
        "secret",
        "block",
        re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    ),
    Rule(
        "secret.jwt.v1",
        "jwt",
        "secret",
        "block",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    ),
    Rule(
        "secret.private_key_block.v1",
        "private_key_block",
        "secret",
        "block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
    ),
    Rule(
        "secret.supabase_url.v1",
        "supabase_url",
        "secret",
        "block",
        re.compile(r"https://[a-z0-9]{20}\.supabase\.co"),
    ),
    Rule(
        "secret.supabase_service_key.v1",
        "supabase_service_key",
        "secret",
        "block",
        re.compile(r"\bservice_role\b.{0,40}\beyJ", re.DOTALL),
    ),
    Rule(
        "secret.assigned_secret.v1",
        "assigned_secret",
        "secret",
        "block",
        re.compile(
            r"\b(?:api[_-]?key|secret|password|passwd|token|credential)\s*[:=]\s*"
            r"['\"][A-Za-z0-9_\-./+]{16,}['\"]",
            re.IGNORECASE,
        ),
    ),
    Rule(
        "secret.bearer_token.v1",
        "bearer_token",
        "secret",
        "block",
        re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{24,}"),
    ),
    Rule(
        "secret.session_cookie.v1",
        "session_cookie",
        "secret",
        "block",
        re.compile(r"\b(?:sb-[a-z0-9-]+-auth-token|kleos_session)\s*=\s*[A-Za-z0-9._-]{16,}"),
    ),
    Rule(
        "secret.database_url.v1",
        "database_url_with_password",
        "secret",
        "block",
        re.compile(r"\bpostgres(?:ql)?(?:\+\w+)?://[^\s:/]+:[^\s@]{4,}@"),
    ),
)


# ---------------------------------------------------------------------------
# Layer 2 — PII. Severity `redact`.
# ---------------------------------------------------------------------------

PII_RULES: Final[tuple[Rule, ...]] = (
    Rule(
        "pii.email.v1",
        "email",
        "pii",
        "redact",
        re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
        slot="EMAIL",
    ),
    Rule(
        "pii.us_phone.v1",
        "phone",
        "pii",
        "redact",
        re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
        slot="PHONE",
    ),
    Rule(
        "pii.ssn.v1", "ssn_like", "pii", "redact", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), slot="ID"
    ),
    # A home directory path names its owner, and these appear constantly in
    # anything pasted from a terminal.
    Rule(
        "pii.home_path.v1",
        "home_path",
        "pii",
        "redact",
        re.compile(r"(?:/(?:Users|home)/|[A-Z]:\\Users\\)[A-Za-z0-9._-]+"),
        slot="PATH",
    ),
    Rule(
        "pii.url_with_token.v1",
        "url_with_token",
        "pii",
        "redact",
        re.compile(r"https?://\S*[?&](?:token|key|access_token|api_key)=[^\s&]+"),
        slot="URL",
    ),
    Rule(
        "pii.street_address.v1",
        "street_address",
        "pii",
        "redact",
        re.compile(
            r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct)\b"
        ),
        slot="ADDRESS",
    ),
    Rule(
        "pii.postal_code.v1",
        "postal_code",
        "pii",
        "redact",
        re.compile(r"\b\d{5}(?:-\d{4})?\b(?=\s*(?:,|\.|$))"),
        slot="POSTAL",
    ),
    Rule(
        "pii.student_id.v1",
        "student_id",
        "pii",
        "redact",
        re.compile(
            r"\b(?:student|employee|emp|badge)\s*(?:id|#|number)?\s*[:#]?\s*\d{5,10}\b",
            re.IGNORECASE,
        ),
        slot="ID",
    ),
    Rule(
        "pii.social_handle.v1",
        "social_handle",
        "pii",
        "redact",
        re.compile(r"(?<![\w/])@[A-Za-z][A-Za-z0-9_]{2,29}\b"),
        slot="HANDLE",
    ),
    Rule(
        "pii.uuid.v1",
        "uuid",
        "pii",
        "redact",
        re.compile(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
            re.IGNORECASE,
        ),
        slot="ID",
    ),
    # An absolute timestamp pins an event to a real moment in someone's life.
    # Relative phrasing ("due Friday") carries the same decision information
    # without the anchor, which is why scenarios render deadlines that way.
    Rule(
        "pii.absolute_datetime.v1",
        "absolute_datetime",
        "pii",
        "redact",
        re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?(?:Z|[+-]\d{2}:?\d{2})?\b"),
        slot="DATE",
    ),
)


# ---------------------------------------------------------------------------
# Layer 4 — structural heuristics. Severity `review`.
# ---------------------------------------------------------------------------

#: Words whose capitalization is grammatical rather than a name. Shared with the
#: private-fact heuristics so the two layers agree about what a name looks like.
SENTENCE_STARTERS: Final[tuple[str, ...]] = (
    "The",
    "This",
    "That",
    "These",
    "Those",
    "It",
    "If",
    "When",
    "With",
    "Start",
    "Then",
    "What",
    "Given",
    "Where",
    "Which",
    "Also",
    "Nothing",
    "Most",
    "There",
    "Rank",
    "Do",
    "Confirm",
    "And",
    "But",
    "Both",
    "First",
    "Second",
    "Third",
    "Deadlines",
    "Timing",
    "Evidence",
    "Item",
    "Next",
    "Email",
    "Call",
    "Contact",
    "See",
    "Note",
    "Its",
    "Their",
    "Your",
    "Our",
    "Everything",
    "Either",
    "Neither",
    "Because",
    "Since",
    "While",
    "After",
    "Before",
    "Once",
    "Unless",
)

#: Organization suffixes strong enough to name an entity.
ORG_SUFFIXES: Final[tuple[str, ...]] = (
    "Inc",
    "LLC",
    "Ltd",
    "Corp",
    "Corporation",
    "Company",
    "University",
    "College",
    "Institute",
    "Laboratory",
    "Labs",
    "Foundation",
    "Hospital",
    "Clinic",
    "Bank",
    "Group",
    "Partners",
    "Holdings",
)

STRUCTURAL_RULES: Final[tuple[Rule, ...]] = (
    Rule(
        "structural.org_name.v1",
        "org_name",
        "structural",
        "review",
        re.compile(r"\b(?:[A-Z][A-Za-z0-9&.-]+\s+){1,3}(?:" + "|".join(ORG_SUFFIXES) + r")\b"),
        slot="ORG",
    ),
    # A capitalized bigram is a weak signal on its own — "Blue Harbor" is a
    # fictional project, "Dana Whitfield" is a person, and nothing in the string
    # distinguishes them. Hence `review`, not `redact`.
    #
    # The negative lookahead stops the bigram spanning a sentence boundary:
    # "…first. Then Silverbrook…" is not a person called "Then Silverbrook",
    # and flagging it teaches a reviewer to skim.
    Rule(
        "structural.person_name.v1",
        "person_name",
        "structural",
        "review",
        # `[ \t]+`, not `\s+`: a person's name is written on one line. Allowing
        # the separator to match a newline joined the last word of one line to
        # the first word of the next, so "Active workspace: Research\nBlue
        # Harbor is due..." matched "Research Blue" and redaction rewrote it to
        # "Active workspace: Sam Ridley Harbor" — corrupting a prompt whose
        # assistant turn still named Blue Harbor, which the no_unsupported_claims
        # gate then correctly rejected. Narrowing the separator removes the false
        # positive without weakening the rule on any name a person would write.
        re.compile(
            r"\b(?!(?:" + "|".join(SENTENCE_STARTERS) + r")\b)"
            r"[A-Z][a-z]{2,}[ \t]+[A-Z][a-z]{2,}\b"
        ),
        slot="PERSON",
    ),
)


ALL_RULES: Final[tuple[Rule, ...]] = SECRET_RULES + PII_RULES + STRUCTURAL_RULES


# ---------------------------------------------------------------------------
# Allowlists
# ---------------------------------------------------------------------------

#: Addresses that are documentation placeholders. ``.invalid`` is reserved by
#: RFC 2606 and is what every committed fixture uses.
EMAIL_ALLOWLIST: Final[re.Pattern[str]] = re.compile(
    r"(?:noreply@|example\.com|example\.org|example\.invalid|\.invalid|"
    r"your[-_]?email|user@host|@example|name@domain)",
    re.IGNORECASE,
)

UUID_ALLOWLIST: Final[re.Pattern[str]] = re.compile(
    r"(?:00000000-0000-0000-0000-000000000000|deadbeef)", re.IGNORECASE
)

#: Capitalized bigrams that are ordinary language rather than names. Kept short
#: on purpose: every entry is a case the structural layer stops looking at.
PHRASE_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {
        "Item A",
        "Item B",
        "Item C",
        "Next Steps",
        "Due Date",
        "In Progress",
        "Not Started",
        "Pull Request",
        "Code Review",
        "Weekly Update",
        "Office Hours",
        "New York",  # a place in a generated distractor, not a person
    }
)

#: Rules whose matches are never redacted, only reported.
NON_REDACTING_SEVERITIES: Final[frozenset[str]] = frozenset({"block", "warn"})


def rules_for_layer(layer: str) -> tuple[Rule, ...]:
    """Every rule belonging to one layer."""
    return tuple(rule for rule in ALL_RULES if rule.layer == layer)


def is_allowlisted(kind: str, matched: str) -> bool:
    """Whether a match is a known false positive."""
    if kind == "email":
        return bool(EMAIL_ALLOWLIST.search(matched))
    if kind == "uuid":
        return bool(UUID_ALLOWLIST.search(matched))
    if kind == "person_name":
        return matched in PHRASE_ALLOWLIST
    if kind == "org_name":
        return matched in PHRASE_ALLOWLIST
    return False
