from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Final, NamedTuple

UNSUPPORTED_ENTITY_THRESHOLD: Final[float] = 0.0

NEEDS_REVIEW_SCORE: Final[float] = 0.4

FACT_TEACHING_SCORE: Final[float] = 0.75


class FactRule(NamedTuple):
    rule_id: str
    pattern: re.Pattern[str]
    score: float
    rationale: str


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
    rule_id: str
    message_index: int
    role: str
    span: tuple[int, int]
    score: float
    rationale: str
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
    signals: list[FactRiskSignal] = field(default_factory=list)

    @property
    def max_score(self) -> float:
        return max((s.score for s in self.signals), default=0.0)

    @property
    def verdict(self) -> str:
        if self.max_score >= FACT_TEACHING_SCORE:
            return "fact_teaching"
        if self.max_score >= NEEDS_REVIEW_SCORE:
            return "needs_fact_review"
        return "policy_like"

    @property
    def requires_human(self) -> bool:
        return self.verdict != "policy_like"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "max_score": self.max_score,
            "signals": [s.to_dict() for s in self.signals],
        }


def _supported_nouns(text: str) -> set[str]:
    return {match.group(0) for match in _PROPER_NOUN.finditer(text)}


def _asserted_nouns(text: str) -> set[str]:
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
