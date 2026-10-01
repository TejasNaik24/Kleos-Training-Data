from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kleos_training_data.hashing import jaccard, normalize_text, shingles
from kleos_training_data.privacy.detect import Detection
from kleos_training_data.privacy.facts import FactRiskAssessment
from kleos_training_data.review.llm_schema import REVIEW_RESPONSE_SCHEMA
from kleos_training_data.review.reviewers import render_instructions
from kleos_training_data.review.rubric import DIMENSIONS, HARD_GATES

DEFAULT_NEIGHBOURS: int = 3

NEIGHBOUR_FLOOR: float = 0.35


@dataclass
class PacketItem:
    candidate_id: str
    content_hash: str
    task: str
    scenario_family: str
    policy_claim: str
    anti_claim: str
    payload: dict[str, Any]
    variation_axes: dict[str, str]
    detections: list[Detection] = field(default_factory=list)
    fact_risk: FactRiskAssessment = field(default_factory=FactRiskAssessment)
    neighbours: list[tuple[str, float]] = field(default_factory=list)
    perturbation_kind: str | None = None
    contract_valid: bool = True

    def to_machine_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "task": self.task,
            "policy_claim": self.policy_claim,
            "anti_claim": self.anti_claim,
            "payload": self.payload,
            "variation_axes": self.variation_axes,
            "contract_valid": self.contract_valid,
            "privacy_summary": {
                "counts": {d.kind: 1 for d in self.detections},
                "residual_rules": sorted({d.rule_id for d in self.detections}),
            },
            "fact_signals": [s.to_dict() for s in self.fact_risk.signals],
        }


@dataclass
class ReviewPacket:
    packet_id: str
    batch_id: str
    items: list[PacketItem] = field(default_factory=list)

    def write(self, directory: Path | str) -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)

        (root / "packet.md").write_text(self.render_markdown(), encoding="utf-8")
        (root / "packet.jsonl").write_text(
            "".join(
                json.dumps(item.to_machine_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                for item in self.items
            ),
            encoding="utf-8",
        )
        (root / "packet.meta.json").write_text(
            json.dumps(
                {
                    "packet_id": self.packet_id,
                    "batch_id": self.batch_id,
                    "candidates": [i.candidate_id for i in self.items],
                    "response_schema": REVIEW_RESPONSE_SCHEMA,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return root

    def render_markdown(self) -> str:
        lines: list[str] = [
            f"# Review packet {self.packet_id}",
            "",
            f"Batch `{self.batch_id}` — {len(self.items)} candidate(s).",
            "",
            "## How to review",
            "",
            "For each candidate, the question is **not** whether the answer looks",
            "reasonable. It is whether the example teaches the stated policy, and",
            "whether a model that memorized it would reveal anything about a real",
            "person.",
            "",
            "Score each dimension 0–4:",
            "",
        ]
        lines += [f"- **{name}** — {question}" for name, question in DIMENSIONS.items()]
        lines += ["", "Then decide each hard gate, PASS or FAIL:", ""]
        lines += [f"- **{name}** — {question}" for name, question in HARD_GATES.items()]
        lines += [
            "",
            "A failing gate rejects the example whatever the scores say. You may",
            "override a score with a justification; you may **not** approve over a",
            "failing `no_private_data` or `policy_not_facts` gate. Fix the content",
            "instead — that changes its id and requires a fresh review, which is",
            "the point.",
            "",
        ]

        for index, item in enumerate(self.items, start=1):
            lines += self._render_item(index, item)

        lines += [
            "---",
            "",
            "## Recording a decision",
            "",
            "```bash",
            "python scripts/record_decision.py \\",
            "    --candidate <id> --decision approve \\",
            "    --gate no_private_data=PASS --gate policy_not_facts=PASS \\",
            "    --gate no_unsupported_claims=PASS --gate schema_and_contract_valid=PASS",
            "```",
            "",
        ]
        return "\n".join(lines)

    def _render_item(self, index: int, item: PacketItem) -> list[str]:
        lines = [
            "---",
            "",
            f"## {index}. `{item.candidate_id}`",
            "",
            f"- task: `{item.task}`",
            f"- scenario family: `{item.scenario_family}`",
            f"- perturbation: `{item.perturbation_kind or 'none (base example)'}`",
            "",
            f"**Should teach:** {item.policy_claim}",
            "",
            f"**Must not teach:** {item.anti_claim}",
            "",
            "### Variation axes",
            "",
            "| axis | value |",
            "| --- | --- |",
        ]
        lines += [f"| {k} | {v} |" for k, v in sorted(item.variation_axes.items())]
        lines += ["", "### Conversation", ""]

        for message in item.payload.get("messages", []):
            if message.get("reasoning"):
                lines += ["**reasoning:**", "", "```", str(message["reasoning"]), "```", ""]
            lines += [
                f"**{message.get('role')}:**",
                "",
                "```",
                str(message.get("content", "")),
                "```",
                "",
            ]

        if item.detections:
            lines += ["### Privacy detections", ""]
            lines += [
                f"- `{d.rule_id}` ({d.severity}) at `{d.field_path}` — {d.excerpt}"
                for d in item.detections
            ]
            lines += [""]
        else:
            lines += ["### Privacy detections", "", "None.", ""]

        lines += [
            "### Private-fact risk",
            "",
            f"Verdict: **{item.fact_risk.verdict}** (max score {item.fact_risk.max_score:.2f})",
            "",
        ]
        if item.fact_risk.signals:
            lines += [
                f"- `{s.rule_id}` in message {s.message_index} ({s.role}) — "
                f"{s.rationale}: {s.excerpt}"
                for s in item.fact_risk.signals
            ]
            lines += [""]

        if item.neighbours:
            lines += [
                "### Nearest already-promoted examples",
                "",
                "Redundancy is worth seeing *before* approval, not at deduplication after.",
                "",
            ]
            lines += [f"- `{other}` — similarity {score:.2f}" for other, score in item.neighbours]
            lines += [""]

        return lines


def nearest_neighbours(
    text: str, corpus: dict[str, str], *, limit: int = DEFAULT_NEIGHBOURS
) -> list[tuple[str, float]]:
    if not corpus:
        return []
    target = shingles(normalize_text(text))
    scored = [
        (other_id, jaccard(target, shingles(normalize_text(other_text))))
        for other_id, other_text in corpus.items()
    ]
    ranked = sorted(
        (pair for pair in scored if pair[1] >= NEIGHBOUR_FLOOR),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return ranked[:limit]


def machine_prompt(item: PacketItem) -> str:
    return (
        render_instructions()
        + "\n\nCandidate:\n"
        + json.dumps(item.to_machine_dict(), indent=2, ensure_ascii=False, sort_keys=True)
    )
