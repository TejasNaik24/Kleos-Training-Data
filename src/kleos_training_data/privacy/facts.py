"""Private-fact risk: the part sanitization cannot fix.

PII is a **string** problem with a mechanical fix. An email is an email; replace
it and you are done.

A private fact is a **semantic** problem with no mechanical fix at all. Consider:

    Prioritize the Motorola project — your internship there ends in three weeks
    and your manager already flagged the deadline.

Strip every name and it still teaches that a specific person had a specific
internship ending on a specific timeline. There is no substitution that repairs
it, because the *situation* is the identifying thing. The example is unusable
however thoroughly it is scrubbed.

So this module produces **signals, never redactions**. It surfaces spans into
the reviewer packet and sets a verdict; the decision belongs to review, as the
``policy_not_facts`` hard gate. That division is deliberate — a heuristic
confident enough to auto-reject would also be confident enough to auto-approve,
and neither is warranted here.

The question a reviewer is answering:

    If a model memorized this example and surfaced it later, would it reveal
    something about a real individual?
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Final, NamedTuple

#: Ratio of assistant proper nouns absent from the prompt that forces review.
UNSUPPORTED_ENTITY_THRESHOLD: Final[float] = 0.0

#: Score at or above which a candidate needs a human fact decision.
NEEDS_REVIEW_SCORE: Final[float] = 0.4

#: Score at or above which a candidate is presumed to be teaching a fact.
FACT_TEACHING_SCORE: Final[float] = 0.75


class FactRule(NamedTuple):
    """One private-fact heuristic."""

    rule_id: str
    pattern: re.Pattern[str]
    score: float
    rationale: str


#: Superset of the public repo's `_FACT_TEACHING_HINTS` (validation.py:349-353).
#: Each phrasing asserts something about the reader that the prompt did not
#: supply — which is either a hallucination or a memory, and both disqualify.
FACT_RULES: Final[tuple[FactRule, ...]] = (
    FactRule(
        "fact.attributive_justification",
        re.compile(
            r"\bbecause (?:he|she|they|you|I) (?:work|works|worked|intern|interns) at\b",
            re.IGNORECASE,
        ),
        0.9,
        "justifies a decision by who someone works for, which is a fact about a person",
    ),
    FactRule(
        "fact.document_recall",
        re.compile(
            r"\byour (?:resume|CV|transcript|profile) (?:says|states|lists|shows)\b", re.IGNORECASE
        ),
        0.85,
        "recalls the contents of a personal document",
    ),
    FactRule(
        "fact.session_recall",
        re.compile(
            r"\bas (?:we|you) (?:discussed|mentioned|agreed) (?:last|on|in) \w+", re.IGNORECASE
        ),
        0.7,
        "refers to a prior conversation the prompt does not contain",
    ),
    FactRule(
        "fact.biographical_possessive",
        re.compile(
            r"\byour (?:advisor|manager|supervisor|landlord|GPA|salary|employer|"
            r"internship|roommate|therapist|doctor)\b",
            re.IGNORECASE,
        ),
        0.8,
        "asserts a biographical relationship or attribute",
    ),
    FactRule(
        "fact.employment_claim",
        re.compile(r"\b(?:your|their|his|her) (?:internship|job|role) at [A-Z]\w+", re.IGNORECASE),
        0.9,
        "names a specific employment relationship",
    ),
)

#: Words that begin a sentence and so are capitalized for grammatical reasons.
_SENTENCE_STARTERS: Final[frozenset[str]] = frozenset(
    {
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
        "Deadlines",
        "Timing",
        "Evidence",
        "Item",
        "Also",
        "Nothing",
        "Most",
        "There",
        "Rank",
        "Do",
        "Confirm",
        "I",
        "A",
        "An",
        "And",
        "But",
        "Both",
        "First",
    }
)

_PROPER_NOUN = re.compile(r"\b[A-Z][a-z]{2,}\b")
_SPECIFIC_NUMBER = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\$\d[\d,.]*\b|\b\d{4}-\d{2}-\d{2}\b")


@dataclass(frozen=True)
class FactRiskSignal:
    """One heuristic firing."""

    rule_id: str
    message_index: int
    role: str
    span: tuple[int, int]
    score: float
    rationale: str
    #: A redacted excerpt, safe to print in a reviewer packet.
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "message_index": self.message_index,
            "role": self.role,
            "span": list(self.span),
            "score": self.score,
            "rationale": self.rationale,
            "excerpt": self.excerpt,
        }


@dataclass
class FactRiskAssessment:
    """What the heuristics found, and what it implies."""

    signals: list[FactRiskSignal] = field(default_factory=list)

    @property
    def max_score(self) -> float:
        return max((s.score for s in self.signals), default=0.0)

    @property
    def verdict(self) -> str:
        """``policy_like`` | ``needs_fact_review`` | ``fact_teaching``."""
        if self.max_score >= FACT_TEACHING_SCORE:
            return "fact_teaching"
        if self.max_score >= NEEDS_REVIEW_SCORE:
            return "needs_fact_review"
        return "policy_like"

    @property
    def requires_human(self) -> bool:
        """Whether a human must decide the ``policy_not_facts`` gate."""
        return self.verdict != "policy_like"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "max_score": self.max_score,
            "signals": [s.to_dict() for s in self.signals],
        }


def _supported_nouns(text: str) -> set[str]:
    """Names the prompt can be said to supply. Deliberately **permissive**.

    Includes sentence-initial words. Anything missing from this set becomes a
    false accusation against the assistant, so the failure mode of being too
    generous here is merely a missed signal, while the failure mode of being too
    strict is a reviewer taught to ignore the strongest signal we have.

    Concretely: a prose prompt reading "Silverbrook is due in 5 days." must
    register Silverbrook, even though its capital is sentence-initial.
    """
    return {match.group(0) for match in _PROPER_NOUN.finditer(text)}


def _asserted_nouns(text: str) -> set[str]:
    """Names the assistant can be said to assert. Deliberately **conservative**.

    Excludes both stopwords and sentence-initial words, because a capital at the
    start of a sentence is grammatical and therefore ambiguous — and an ambiguous
    signal is not enough to accuse an answer of inventing an entity. "Take the
    confirmed one…" opens with a capitalized verb, not a company.
    """
    found: set[str] = set()
    for match in _PROPER_NOUN.finditer(text):
        word = match.group(0)
        if word in _SENTENCE_STARTERS:
            continue
        preceding = text[: match.start()].rstrip()
        if not preceding or preceding[-1] in ".!?:\n":
            continue
        found.add(word)
    return found


def assess(payload: dict[str, Any]) -> FactRiskAssessment:
    """Score a candidate's private-fact risk."""
    from kleos_training_data.privacy.detect import redacted_excerpt

    messages = payload.get("messages") or []
    signals: list[FactRiskSignal] = []

    prompt_text = " ".join(
        str(m.get("content", "")) for m in messages if m.get("role") != "assistant"
    )
    prompt_nouns = _supported_nouns(prompt_text)
    prompt_numbers = set(_SPECIFIC_NUMBER.findall(prompt_text))

    for index, message in enumerate(messages):
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))

        for rule in FACT_RULES:
            for match in rule.pattern.finditer(content):
                signals.append(
                    FactRiskSignal(
                        rule_id=rule.rule_id,
                        message_index=index,
                        role=role,
                        span=(match.start(), match.end()),
                        score=rule.score,
                        rationale=rule.rationale,
                        excerpt=redacted_excerpt(content, match.start(), match.end()),
                    )
                )

        if role != "assistant":
            continue

        # The strongest single signal: the assistant names something the prompt
        # never mentioned. That is either a hallucination or a memory, and both
        # are disqualifying for different reasons.
        for noun in sorted(_asserted_nouns(content) - prompt_nouns):
            position = content.find(noun)
            signals.append(
                FactRiskSignal(
                    rule_id="fact.unsupported_entity",
                    message_index=index,
                    role=role,
                    span=(position, position + len(noun)),
                    score=0.6,
                    rationale=(
                        f"the assistant names an entity absent from the prompt "
                        f"({len(noun)} chars); either invented or remembered"
                    ),
                    excerpt=redacted_excerpt(content, position, position + len(noun)),
                )
            )

        for number in sorted(set(_SPECIFIC_NUMBER.findall(content)) - prompt_numbers):
            position = content.find(number)
            signals.append(
                FactRiskSignal(
                    rule_id="fact.unsupported_number",
                    message_index=index,
                    role=role,
                    span=(position, position + len(number)),
                    score=0.5,
                    rationale="the assistant asserts a specific figure absent from the prompt",
                    excerpt=redacted_excerpt(content, position, position + len(number)),
                )
            )

    return FactRiskAssessment(signals=_with_masked_excerpts(signals, messages))


def _with_masked_excerpts(
    signals: list[FactRiskSignal], messages: list[dict[str, Any]]
) -> list[FactRiskSignal]:
    """Rebuild every excerpt masking all spans found in the same message.

    The same defect the detection layer had: an excerpt that masks only its own
    span reprints its neighbours. Here it is worse, because a fact signal's whole
    purpose is to name a *suspected private fact* — quoting the one next to it
    into a reviewer packet defeats the point entirely.
    """
    from kleos_training_data.privacy.detect import redacted_excerpt

    spans_by_message: dict[int, tuple[tuple[int, int], ...]] = {}
    for signal in signals:
        spans_by_message.setdefault(signal.message_index, ())
    for index in spans_by_message:
        spans_by_message[index] = tuple(s.span for s in signals if s.message_index == index)

    rebuilt: list[FactRiskSignal] = []
    for signal in signals:
        content = str(messages[signal.message_index].get("content", ""))
        rebuilt.append(
            FactRiskSignal(
                rule_id=signal.rule_id,
                message_index=signal.message_index,
                role=signal.role,
                span=signal.span,
                score=signal.score,
                rationale=signal.rationale,
                excerpt=redacted_excerpt(
                    content,
                    signal.span[0],
                    signal.span[1],
                    mask_spans=spans_by_message[signal.message_index],
                ),
            )
        )
    return rebuilt
